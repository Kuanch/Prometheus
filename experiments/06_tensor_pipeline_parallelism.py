#!/usr/bin/env python3
"""
Experiment 6: Tensor Parallelism & Pipeline Parallelism Demo

Demonstrates two key distributed parallelism strategies in PyTorch:

1. TENSOR PARALLELISM (TP):
   - Splits individual layers/tensors ACROSS devices
   - Each device holds a SHARD of every layer
   - Requires AllReduce communication within each forward/backward pass

2. PIPELINE PARALLELISM (PP):
   - Splits the model LAYER-WISE across devices
   - Each device holds a SUBSET of consecutive layers
   - Micro-batching hides the pipeline bubble

3. HYBRID TP + PP for maximum scalability

Usage:
   torchrun --nproc_per_node=<N> experiments/06_tensor_pipeline_parallelism.py
   python experiments/06_tensor_pipeline_parallelism.py --demo-mode
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.distributed as dist


# ============================================================================
#  PART 1: MODEL DEFINITION
# ============================================================================

class TransformerBlock(nn.Module):
    """A simplified Transformer block for demonstration."""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ffn_up = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.ffn_down = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, S, D = x.shape
        head_dim = D // self.num_heads
        residual = x
        x = self.ln1(x)
        q = self.q_proj(x).view(B, S, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        x = residual + self.dropout(self.out_proj(attn_output))
        residual = x
        x = self.ln2(x)
        x = self.ffn_up(x)
        x = torch.relu(x)
        x = self.ffn_down(x)
        x = residual + self.dropout(x)
        return x


class SimpleTransformerLM(nn.Module):
    """A simple Transformer Language Model for parallelism demos."""

    def __init__(self, vocab_size=32000, hidden_dim=512, num_layers=8, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)


# ============================================================================
#  PART 2: TENSOR PARALLELISM (TP)
# ============================================================================
#
# Column-Parallel Linear: split output dim across GPUs
#   GPU 0 gets W[:, :D/2], GPU 1 gets W[:, D/2:]
#   Each GPU: Y_i = X @ W_i  (different output columns)
#
# Row-Parallel Linear: split input dim across GPUs
#   GPU 0 gets W[:D/2, :], GPU 1 gets W[D/2:, :]
#   Each GPU: Y_i = X_i @ W_i, then AllReduce Y = sum(Y_i)

def apply_tensor_parallelism_manual(model, rank, world_size, device):
    """Manually apply Column-Parallel and Row-Parallel sharding."""

    def _shard_linear_column(linear, rank, world_size):
        out_features = linear.out_features // world_size
        weight_chunk = linear.weight.data.chunk(world_size, dim=0)[rank]
        new_linear = nn.Linear(linear.in_features, out_features, bias=False, device=device)
        new_linear.weight = nn.Parameter(weight_chunk.to(device))
        return new_linear

    def _shard_linear_row(linear, rank, world_size):
        in_features = linear.in_features // world_size
        weight_chunk = linear.weight.data.chunk(world_size, dim=1)[rank]
        new_linear = nn.Linear(in_features, linear.out_features, bias=False, device=device)
        new_linear.weight = nn.Parameter(weight_chunk.to(device))
        return new_linear

    for layer in model.layers:
        layer.q_proj = _shard_linear_column(layer.q_proj, rank, world_size)
        layer.k_proj = _shard_linear_column(layer.k_proj, rank, world_size)
        layer.v_proj = _shard_linear_column(layer.v_proj, rank, world_size)
        layer.out_proj = _shard_linear_row(layer.out_proj, rank, world_size)
        layer.ffn_up = _shard_linear_column(layer.ffn_up, rank, world_size)
        layer.ffn_down = _shard_linear_row(layer.ffn_down, rank, world_size)
        layer.num_heads = layer.num_heads // world_size
    return model


class TensorParallelTransformerBlock(nn.Module):
    """
    Transformer block with explicit Tensor Parallelism.

    Communication pattern per layer:
      Input X (replicated) -> ColParallel QKV -> Attention(local heads)
      -> RowParallel OutProj -> *AllReduce* -> ColParallel FFN_up
      -> RowParallel FFN_down -> *AllReduce* -> Output (replicated)

    Total: 2 AllReduce ops per layer
    """

    def __init__(self, hidden_dim, num_heads, rank, world_size, device, process_group=None):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group
        assert hidden_dim % world_size == 0
        assert num_heads % world_size == 0
        self.local_heads = num_heads // world_size
        self.head_dim = hidden_dim // num_heads
        self.local_hidden = self.local_heads * self.head_dim
        self.q_proj = nn.Linear(hidden_dim, self.local_hidden, bias=False, device=device)
        self.k_proj = nn.Linear(hidden_dim, self.local_hidden, bias=False, device=device)
        self.v_proj = nn.Linear(hidden_dim, self.local_hidden, bias=False, device=device)
        self.out_proj = nn.Linear(self.local_hidden, hidden_dim, bias=False, device=device)
        local_ffn_dim = (hidden_dim * 4) // world_size
        self.ffn_up = nn.Linear(hidden_dim, local_ffn_dim, bias=False, device=device)
        self.ffn_down = nn.Linear(local_ffn_dim, hidden_dim, bias=False, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, device=device)

    def forward(self, x):
        B, S, D = x.shape
        residual = x
        x = self.ln1(x)
        q = self.q_proj(x).view(B, S, self.local_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.local_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.local_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.local_hidden)
        out = self.out_proj(out)
        if self.process_group is not None:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.process_group)
        x = residual + out
        residual = x
        x = self.ln2(x)
        x = self.ffn_up(x)
        x = torch.relu(x)
        x = self.ffn_down(x)
        if self.process_group is not None:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.process_group)
        x = residual + x
        return x


# ============================================================================
#  PART 3: PIPELINE PARALLELISM (PP)
# ============================================================================

class PipelineStage(nn.Module):
    """A pipeline stage holding a subset of transformer layers."""

    def __init__(self, layers, stage_id, num_stages, embedding=None, ln_final=None, lm_head=None):
        super().__init__()
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.layers = layers
        self.embedding = embedding
        self.ln_final = ln_final
        self.lm_head = lm_head

    def forward(self, x):
        if self.embedding is not None:
            x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        if self.ln_final is not None:
            x = self.ln_final(x)
        if self.lm_head is not None:
            x = self.lm_head(x)
        return x


def create_pipeline_stages(model, num_stages, devices):
    """Split a model into pipeline stages across devices."""
    num_layers = len(model.layers)
    layers_per_stage = num_layers // num_stages
    remainder = num_layers % num_stages
    stages = []
    layer_idx = 0
    for stage_id in range(num_stages):
        n_layers = layers_per_stage + (1 if stage_id < remainder else 0)
        stage_layers = nn.ModuleList([model.layers[layer_idx + i] for i in range(n_layers)])
        embedding = model.embedding if stage_id == 0 else None
        ln_final = model.ln_final if stage_id == num_stages - 1 else None
        lm_head = model.lm_head if stage_id == num_stages - 1 else None
        device = devices[stage_id]
        stage = PipelineStage(stage_layers, stage_id, num_stages,
                              embedding, ln_final, lm_head).to(device)
        stages.append(stage)
        layer_idx += n_layers
    return stages


def gpipe_forward(stages, input_ids, num_microbatches=4, devices=None):
    """
    GPipe-style pipeline forward pass with micro-batching.

    Schedule (F=forward, B=backward):
      GPU 0: F0  F1  F2  F3  B3  B2  B1  B0
      GPU 1:     F0  F1  F2  F3  B3  B2  B1  B0
      Bubble ratio = (stages-1) / (microbatches + stages - 1)
    """
    batch_size = input_ids.size(0)
    assert batch_size % num_microbatches == 0
    microbatches = input_ids.chunk(num_microbatches, dim=0)
    all_outputs = []
    for mb in microbatches:
        x = mb
        for stage_idx, stage in enumerate(stages):
            device = devices[stage_idx] if devices else next(stage.parameters()).device
            x = x.to(device)
            x = stage(x)
        all_outputs.append(x)
    return torch.cat(all_outputs, dim=0)


# ============================================================================
#  PART 4: 1F1B SCHEDULE DESCRIPTION
# ============================================================================

def one_f1b_schedule_description():
    """Print the 1F1B schedule description."""
    print("""
    ================================================================
     1F1B Pipeline Schedule (PipeDream / Megatron-LM style)
    ================================================================
    Key idea: Interleave forward and backward passes to reduce
    peak memory usage compared to GPipe.

    Time step:  0    1    2    3    4    5    6    7    8    9
    GPU 0:     F0   F1   F2   F3   B0   ..   B1   ..   B2   B3
    GPU 1:          F0   F1   F2   B0   F3   B1        B2   B3
    GPU 2:               F0   F1   B0   F2   B1   F3   B2   B3

    Advantages over GPipe:
    - Lower peak memory (activations freed earlier)
    - Same bubble ratio
    - Better compute/communication overlap

    Memory: GPipe peak = M * act_size, 1F1B peak = P * act_size
    When M >> P, 1F1B uses significantly less memory!
    """)


# ============================================================================
#  PART 5: DEMO MODE (Single-GPU / CPU Simulation)
# ============================================================================

def demo_tensor_parallelism(device):
    """Demonstrate Tensor Parallelism concepts on a single device."""
    print("\n" + "=" * 70)
    print(" DEMO: Tensor Parallelism (Simulated on single device)")
    print("=" * 70)

    hidden_dim = 512
    seq_len = 64
    batch_size = 4
    simulated_tp_size = 2

    print(f"\n  Config: hidden={hidden_dim}, seq={seq_len}, batch={batch_size}, TP={simulated_tp_size}")

    full_weight = torch.randn(hidden_dim * 4, hidden_dim, device=device)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    print(f"\n  Full FFN weight: {list(full_weight.shape)}, Input: {list(x.shape)}")

    # Column-Parallel Split
    print(f"\n  --- Column-Parallel (split output dim) ---")
    col_shards = full_weight.chunk(simulated_tp_size, dim=0)
    for i, shard in enumerate(col_shards):
        print(f"    GPU {i}: weight shard = {list(shard.shape)}")
    col_outputs = [x @ shard.T for shard in col_shards]
    for i, out in enumerate(col_outputs):
        print(f"    GPU {i}: output shard = {list(out.shape)}")
    col_full = torch.cat(col_outputs, dim=-1)
    reference = x @ full_weight.T
    print(f"    Concat output: {list(col_full.shape)}")
    print(f"    Matches full:  {torch.allclose(col_full, reference, atol=1e-5)}")

    # Row-Parallel Split
    print(f"\n  --- Row-Parallel (split input dim) ---")
    row_weight = torch.randn(hidden_dim, hidden_dim * 4, device=device)
    print(f"  Full weight: {list(row_weight.shape)}")
    row_weight_shards = row_weight.chunk(simulated_tp_size, dim=1)
    input_shards = col_outputs
    for i, (w, xs) in enumerate(zip(row_weight_shards, input_shards)):
        print(f"    GPU {i}: weight={list(w.shape)}, input={list(xs.shape)}")
    row_outputs = [xs @ w.T for xs, w in zip(input_shards, row_weight_shards)]
    allreduce_result = sum(row_outputs)
    reference_row = col_full @ row_weight.T
    print(f"    Partial output: {list(row_outputs[0].shape)}")
    print(f"    AllReduce result: {list(allreduce_result.shape)}")
    print(f"    Matches full: {torch.allclose(allreduce_result, reference_row, atol=1e-4)}")

    # Communication cost
    print(f"\n  --- Communication Cost ---")
    elems = batch_size * seq_len * hidden_dim
    bytes_ar = elems * 4
    total = bytes_ar * 2 * 8  # 2 allreduce/layer * 8 layers
    print(f"    Per AllReduce: {elems:,} elements = {bytes_ar/1e6:.2f} MB")
    print(f"    Per layer: 2 AllReduces")
    print(f"    Total (8 layers): {total/1e6:.2f} MB")
    print(f"    Ring cost: 2*(TP-1)/TP * data_size per AllReduce")


def demo_pipeline_parallelism(device):
    """Demonstrate Pipeline Parallelism on a single device."""
    print("\n" + "=" * 70)
    print(" DEMO: Pipeline Parallelism (Simulated on single device)")
    print("=" * 70)

    vocab_size, hidden_dim, num_layers, num_heads = 1000, 256, 8, 8
    batch_size, seq_len, num_stages = 16, 32, 4

    print(f"\n  Config: vocab={vocab_size}, hidden={hidden_dim}, layers={num_layers}")
    print(f"  PP stages={num_stages}, batch={batch_size}, seq={seq_len}")

    model = SimpleTransformerLM(vocab_size, hidden_dim, num_layers, num_heads)
    devices = [device] * num_stages
    stages = create_pipeline_stages(model, num_stages, devices)

    print(f"\n  Pipeline Stages:")
    for i, stage in enumerate(stages):
        n_params = sum(p.numel() for p in stage.parameters())
        parts = []
        if stage.embedding: parts.append("Embed")
        parts.extend([f"L{j}" for j in range(len(stage.layers))])
        if stage.ln_final: parts.append("LN")
        if stage.lm_head: parts.append("Head")
        print(f"    Stage {i}: [{', '.join(parts)}] params={n_params:,} ({n_params*4/1e6:.2f}MB)")

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    print(f"\n  --- Micro-batch Analysis ---")
    for num_mb in [1, 2, 4, 8]:
        if batch_size % num_mb != 0:
            continue
        bubble = (num_stages - 1) / (num_mb + num_stages - 1)
        t0 = time.perf_counter()
        with torch.no_grad():
            output = gpipe_forward(stages, input_ids, num_microbatches=num_mb, devices=devices)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"    uBatch={num_mb:2d} (size {batch_size//num_mb:2d}): "
              f"out={list(output.shape)}, bubble={bubble:.1%}, time={elapsed:.1f}ms")

    model = model.to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        ref = model(input_ids)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"    No-pipeline ref: out={list(ref.shape)}, time={elapsed:.1f}ms")

    print(f"\n  --- Pipeline Bubble Analysis ---")
    print(f"  {'Stages':>8} {'uBatch':>8} {'Bubble':>8} {'Efficiency':>10}")
    for pp in [2, 4, 8]:
        for mb in [2, 4, 8, 16, 32]:
            b = (pp-1) / (mb+pp-1)
            print(f"  {pp:>8} {mb:>8} {b:>8.1%} {1-b:>10.1%}")
        print(f"  {'---':>8} {'---':>8} {'---':>8} {'---':>10}")


# ============================================================================
#  PART 6: MULTI-GPU DISTRIBUTED
# ============================================================================

def run_distributed_tp(rank, world_size):
    """Run real multi-GPU Tensor Parallelism."""
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if rank == 0:
        print("\n" + "=" * 70)
        print(f" DISTRIBUTED: Tensor Parallelism ({world_size} GPUs)")
        print("=" * 70)

    hidden_dim, num_heads, num_layers, vocab_size = 512, 8, 4, 32000
    embedding = nn.Embedding(vocab_size, hidden_dim).to(device)
    tp_layers = nn.ModuleList([
        TensorParallelTransformerBlock(hidden_dim, num_heads, rank, world_size, device)
        for _ in range(num_layers)
    ])
    ln_final = nn.LayerNorm(hidden_dim).to(device)
    lm_head = nn.Linear(hidden_dim, vocab_size, bias=False).to(device)

    input_ids = torch.randint(0, vocab_size, (4, 64), device=device)
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    with torch.no_grad():
        x = embedding(input_ids)
        for layer in tp_layers:
            x = layer(x)
        x = ln_final(x)
        logits = lm_head(x)
    torch.cuda.synchronize(); dist.barrier()
    elapsed = (time.perf_counter() - t0) * 1000

    if rank == 0:
        print(f"  Input: {list(input_ids.shape)}, Output: {list(logits.shape)}")
        print(f"  Forward: {elapsed:.2f} ms")
        for i in range(world_size):
            print(f"  GPU {i} peak mem: {torch.cuda.max_memory_allocated(i)/1e6:.1f} MB")
    dist.destroy_process_group()


def run_distributed_pp(rank, world_size):
    """Run real multi-GPU Pipeline Parallelism."""
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if rank == 0:
        print("\n" + "=" * 70)
        print(f" DISTRIBUTED: Pipeline Parallelism ({world_size} stages)")
        print("=" * 70)

    hidden_dim, num_heads, num_layers, vocab_size = 512, 8, 8, 32000
    batch_size, seq_len, num_microbatches = 16, 64, 4

    if rank == 0:
        model = SimpleTransformerLM(vocab_size, hidden_dim, num_layers, num_heads)
        devices = [torch.device(f"cuda:{i}") for i in range(world_size)]
        stages = create_pipeline_stages(model, world_size, devices)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=devices[0])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = gpipe_forward(stages, input_ids, num_microbatches, devices)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        bubble = (world_size-1) / (num_microbatches+world_size-1)
        print(f"  Input: {list(input_ids.shape)}, Output: {list(output.shape)}")
        print(f"  Forward: {elapsed:.2f} ms, bubble: {bubble:.1%}")
        for i in range(world_size):
            print(f"  GPU {i} peak mem: {torch.cuda.max_memory_allocated(i)/1e6:.1f} MB")
    dist.destroy_process_group()


# ============================================================================
#  PART 7: PyTorch Native APIs Reference
# ============================================================================

def demo_pytorch_native_apis():
    """Show PyTorch built-in parallelism APIs (torch >= 2.0)."""
    print("\n" + "=" * 70)
    print(" PyTorch Native Parallelism APIs (Reference)")
    print("=" * 70)
    print("""
    1. DeviceMesh - organize GPUs into a logical grid:

       from torch.distributed.device_mesh import init_device_mesh
       mesh = init_device_mesh("cuda", (2, 4), dim_names=("pp", "tp"))
       # PP Stage 0: GPU0 GPU1 GPU2 GPU3  (TP group)
       # PP Stage 1: GPU4 GPU5 GPU6 GPU7  (TP group)

    2. Tensor Parallel API:

       from torch.distributed.tensor.parallel import (
           parallelize_module, ColwiseParallel, RowwiseParallel)

       model = parallelize_module(model, mesh["tp"], {
           "q_proj": ColwiseParallel(),
           "k_proj": ColwiseParallel(),
           "v_proj": ColwiseParallel(),
           "out_proj": RowwiseParallel(),
           "ffn_up": ColwiseParallel(),
           "ffn_down": RowwiseParallel(),
       })

    3. Pipeline Parallel API:

       from torch.distributed.pipelining import (
           SplitPoint, pipeline, ScheduleGPipe, Schedule1F1B)

       pipe = pipeline(model, mb_args=(input,),
           split_spec={"layers.3": SplitPoint.END})
       schedule = ScheduleGPipe(pipe, n_microbatches=4)
       output = schedule.step(input)

    4. Hybrid TP + PP:

       for stage in pipe.stages:
           parallelize_module(stage, mesh["tp"], {...})
    """)


# ============================================================================
#  PART 8: COMPARISON TABLE
# ============================================================================

def print_parallelism_comparison():
    """Print comparison of parallelism strategies."""
    print("\n" + "=" * 70)
    print(" COMPARISON: Parallelism Strategies")
    print("=" * 70)
    print("""
    +------------------+---------------------+---------------------+
    |                  | Tensor Parallelism  | Pipeline Parallelism|
    +------------------+---------------------+---------------------+
    | What's split     | Individual layers   | Groups of layers    |
    | Granularity      | Intra-layer         | Inter-layer         |
    | Communication    | AllReduce per layer | P2P between stages  |
    | Comm frequency   | High (every layer)  | Low (between stages)|
    | Bandwidth need   | Very high (NVLink)  | Moderate            |
    | Memory savings   | Linear in TP degree | Linear in PP degree |
    | Bubble overhead  | None                | Yes (use more uB)   |
    | Best for         | Within-node (fast)  | Cross-node (slower) |
    | Scalability      | 2-8 GPUs typically  | 2-64+ stages        |
    +------------------+---------------------+---------------------+

    Hybrid examples (Megatron-LM):
      GPT-3 175B:  TP=8 x PP=16 x DP=8  = 1024 GPUs
      LLaMA-2 70B: TP=8 x PP=4  x DP=2  =   64 GPUs
    """)


# ============================================================================
#  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TP & PP Parallelism Demo")
    parser.add_argument("--demo-mode", action="store_true",
                        help="Single-device simulation (no multi-GPU needed)")
    parser.add_argument("--mode", choices=["tp", "pp", "both", "all"], default="all")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if is_distributed and not args.demo_mode:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if args.mode in ("tp", "both", "all"):
            run_distributed_tp(rank, world_size)
        if args.mode in ("pp", "both", "all"):
            run_distributed_pp(rank, world_size)
        if rank == 0 and args.mode == "all":
            print_parallelism_comparison()
            demo_pytorch_native_apis()
    else:
        print("\n" + "=" * 70)
        print("  PYTORCH TENSOR & PIPELINE PARALLELISM DEMO")
        print(f"  Device: {device}")
        if not torch.cuda.is_available():
            print("  WARNING: No GPU -- running conceptual demo on CPU")
        print("=" * 70)

        if args.mode in ("tp", "both", "all"):
            demo_tensor_parallelism(device)
        if args.mode in ("pp", "both", "all"):
            demo_pipeline_parallelism(device)
        if args.mode == "all":
            one_f1b_schedule_description()
            print_parallelism_comparison()
            demo_pytorch_native_apis()

        print("\n" + "=" * 70)
        print(" NEXT STEPS")
        print("=" * 70)
        print("""
    Multi-GPU Tensor Parallelism:
      torchrun --nproc_per_node=2 experiments/06_tensor_pipeline_parallelism.py --mode tp

    Multi-GPU Pipeline Parallelism:
      torchrun --nproc_per_node=4 experiments/06_tensor_pipeline_parallelism.py --mode pp

    Demo mode (single device):
      python experiments/06_tensor_pipeline_parallelism.py --demo-mode --mode all
        """)


if __name__ == "__main__":
    main()
