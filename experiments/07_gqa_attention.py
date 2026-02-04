#!/usr/bin/env python3
"""
Experiment 7: Grouped Query Attention (GQA)

This experiment implements GQA from scratch matching Llama 3's attention design,
with every fundamental operation decoupled for individual Nsight profiling.

Architecture (Llama 3 8B defaults):
- d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128
- RoPE with theta=500,000
- Causal masking
- No bias in projections

Decoupled operations (each wrapped in its own NVTX range):
  1. Q projection       (linear: d_model -> n_heads * head_dim)
  2. K projection       (linear: d_model -> n_kv_heads * head_dim)
  3. V projection       (linear: d_model -> n_kv_heads * head_dim)
  4. RoPE application   (rotary embeddings on Q and K)
  5. KV head expansion  (repeat KV heads to match Q head count)
  6. Q @ K^T            (attention score computation)
  7. Causal masking      (apply causal mask)
  8. Softmax            (normalize attention weights)
  9. Attn @ V           (weighted value aggregation)
  10. Output projection  (linear: n_heads * head_dim -> d_model)

Usage:
    python experiments/07_gqa_attention.py

    nsys profile --trace=cuda,nvtx -o /output/gqa_nsys python experiments/07_gqa_attention.py

    ncu --set full --launch-count 50 --export /output/gqa_ncu python experiments/07_gqa_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from contextlib import contextmanager

sys.path.insert(0, '/workspace')
from utils.profiler import GPUProfiler, print_comparison_table, get_gpu_info


# ============================================================================
# NVTX Helpers
# ============================================================================

@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


# ============================================================================
# Rotary Position Embeddings (RoPE)
# ============================================================================

def precompute_rope_frequencies(head_dim: int, max_seq_len: int, theta: float = 500_000.0, device='cuda'):
    """
    Precompute sin/cos tables for RoPE.

    RoPE encodes position by rotating pairs of dimensions:
      freq_i = 1 / (theta ^ (2i / head_dim))   for i = 0, 1, ..., head_dim/2 - 1
      cos(pos * freq_i), sin(pos * freq_i)

    Returns:
        cos_table: [max_seq_len, head_dim/2]
        sin_table: [max_seq_len, head_dim/2]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    # [max_seq_len, head_dim/2]
    angles = torch.outer(positions, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to tensor x.

    x: [batch, n_heads, seq_len, head_dim]
    cos, sin: [seq_len, head_dim/2]

    Splits head_dim into first half (x1) and second half (x2), then rotates:
      (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    This is the half-rotation variant used by Llama.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # Broadcast cos/sin to [1, 1, seq_len, head_dim/2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    out = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    return out


# ============================================================================
# GQA Attention (Decoupled Steps)
# ============================================================================

class GQAAttention(nn.Module):
    """
    Grouped Query Attention matching Llama 3 design.

    Every operation is a separate method so each can be profiled independently.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_groups = n_heads // n_kv_heads   # Q heads per KV head

        # Projections (no bias, matching Llama 3)
        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # Precompute RoPE tables
        cos, sin = precompute_rope_frequencies(self.head_dim, max_seq_len)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    # ---------- Decoupled operations ----------

    def proj_q(self, x: torch.Tensor) -> torch.Tensor:
        """Q projection: [B, S, d] -> [B, n_heads, S, head_dim]"""
        B, S, _ = x.shape
        q = self.W_q(x)                                     # [B, S, n_heads * head_dim]
        return q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def proj_k(self, x: torch.Tensor) -> torch.Tensor:
        """K projection: [B, S, d] -> [B, n_kv_heads, S, head_dim]"""
        B, S, _ = x.shape
        k = self.W_k(x)                                     # [B, S, n_kv_heads * head_dim]
        return k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

    def proj_v(self, x: torch.Tensor) -> torch.Tensor:
        """V projection: [B, S, d] -> [B, n_kv_heads, S, head_dim]"""
        B, S, _ = x.shape
        v = self.W_v(x)                                     # [B, S, n_kv_heads * head_dim]
        return v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

    def apply_rope_q(self, q: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply RoPE to Q: rotates each head's vector by position-dependent angle."""
        return apply_rope(q, self.rope_cos[:seq_len], self.rope_sin[:seq_len])

    def apply_rope_k(self, k: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply RoPE to K."""
        return apply_rope(k, self.rope_cos[:seq_len], self.rope_sin[:seq_len])

    def expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """
        Expand KV heads to match Q head count.
        [B, n_kv_heads, S, head_dim] -> [B, n_heads, S, head_dim]

        Each KV head is repeated n_groups times.
        """
        B, n_kv, S, D = kv.shape
        # expand: insert a group dim, repeat, then flatten
        return (
            kv[:, :, None, :, :]              # [B, n_kv, 1, S, D]
            .expand(B, n_kv, self.n_groups, S, D)
            .reshape(B, self.n_heads, S, D)
        )

    def compute_attn_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Q @ K^T scaled dot-product.
        q: [B, n_heads, S, head_dim]
        k: [B, n_heads, S, head_dim]
        Returns: [B, n_heads, S, S]
        """
        return torch.matmul(q, k.transpose(-2, -1)) * self.scale

    def apply_causal_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply causal (lower-triangular) mask: future positions get -inf."""
        S = scores.shape[-1]
        mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
        return scores.masked_fill(mask, float('-inf'))

    def attn_softmax(self, scores: torch.Tensor) -> torch.Tensor:
        """Softmax over the key dimension."""
        return F.softmax(scores, dim=-1)

    def attn_v_multiply(self, weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Attention weights @ V.
        weights: [B, n_heads, S, S]
        v: [B, n_heads, S, head_dim]
        Returns: [B, n_heads, S, head_dim]
        """
        return torch.matmul(weights, v)

    def proj_output(self, attn_out: torch.Tensor) -> torch.Tensor:
        """
        Output projection: concat heads then project back to d_model.
        attn_out: [B, n_heads, S, head_dim]
        Returns: [B, S, d_model]
        """
        B, _, S, _ = attn_out.shape
        # [B, n_heads, S, head_dim] -> [B, S, n_heads * head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.head_dim)
        return self.W_o(attn_out)

    # ---------- Full forward with NVTX annotations ----------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full GQA forward pass with NVTX ranges on every operation.
        x: [batch, seq_len, d_model]
        Returns: [batch, seq_len, d_model]
        """
        B, S, _ = x.shape

        with nvtx_range("01_Q_Projection"):
            q = self.proj_q(x)

        with nvtx_range("02_K_Projection"):
            k = self.proj_k(x)

        with nvtx_range("03_V_Projection"):
            v = self.proj_v(x)

        with nvtx_range("04_RoPE_Q"):
            q = self.apply_rope_q(q, S)

        with nvtx_range("05_RoPE_K"):
            k = self.apply_rope_k(k, S)

        with nvtx_range("06_KV_Expand"):
            k = self.expand_kv(k)
            v = self.expand_kv(v)

        with nvtx_range("07_QK_Matmul"):
            scores = self.compute_attn_scores(q, k)

        with nvtx_range("08_Causal_Mask"):
            scores = self.apply_causal_mask(scores)

        with nvtx_range("09_Softmax"):
            weights = self.attn_softmax(scores)

        with nvtx_range("10_AttnV_Matmul"):
            attn_out = self.attn_v_multiply(weights, v)

        with nvtx_range("11_Output_Projection"):
            out = self.proj_output(attn_out)

        return out


# ============================================================================
# Profiling Experiments
# ============================================================================

def profile_individual_ops(attn: GQAAttention, x: torch.Tensor, profiler: GPUProfiler):
    """Profile each GQA operation independently."""
    B, S, _ = x.shape
    print(f"\n{'='*70}")
    print(f" Individual Operation Profiling  (B={B}, S={S}, d={attn.d_model})")
    print(f"{'='*70}")

    results = []
    d = attn.d_model
    nh = attn.n_heads
    nkv = attn.n_kv_heads
    hd = attn.head_dim

    # 1. Q projection  —  GEMM: [B*S, d] @ [d, nh*hd]
    q_flops = 2 * B * S * d * (nh * hd)
    r = profiler.benchmark_compute_bound("01 Q proj", lambda: attn.proj_q(x), num_flops=q_flops)
    results.append(r)
    q = attn.proj_q(x)

    # 2. K projection  —  GEMM: [B*S, d] @ [d, nkv*hd]
    k_flops = 2 * B * S * d * (nkv * hd)
    r = profiler.benchmark_compute_bound("02 K proj", lambda: attn.proj_k(x), num_flops=k_flops)
    results.append(r)
    k = attn.proj_k(x)

    # 3. V projection  —  same shape as K
    r = profiler.benchmark_compute_bound("03 V proj", lambda: attn.proj_v(x), num_flops=k_flops)
    results.append(r)
    v = attn.proj_v(x)

    # 4. RoPE on Q
    rope_bytes = 2 * B * nh * S * hd * x.element_size()   # read + write
    r = profiler.benchmark_memory_bound("04 RoPE Q", lambda: attn.apply_rope_q(q, S), bytes_accessed=rope_bytes)
    results.append(r)
    q = attn.apply_rope_q(q, S)

    # 5. RoPE on K
    rope_k_bytes = 2 * B * nkv * S * hd * x.element_size()
    r = profiler.benchmark_memory_bound("05 RoPE K", lambda: attn.apply_rope_k(k, S), bytes_accessed=rope_k_bytes)
    results.append(r)
    k = attn.apply_rope_k(k, S)

    # 6. KV expansion
    expand_bytes = B * nkv * S * hd * x.element_size()   # read only (expand is virtual or copy)
    r = profiler.benchmark_memory_bound("06 KV expand",
                                        lambda: (attn.expand_kv(k), attn.expand_kv(v)),
                                        bytes_accessed=2 * expand_bytes)
    results.append(r)
    k = attn.expand_kv(k)
    v = attn.expand_kv(v)

    # 7. Q @ K^T  —  batched GEMM: [B*nh, S, hd] @ [B*nh, hd, S] -> [B*nh, S, S]
    qk_flops = 2 * B * nh * S * S * hd
    r = profiler.benchmark_compute_bound("07 Q@K^T", lambda: attn.compute_attn_scores(q, k), num_flops=qk_flops)
    results.append(r)
    scores = attn.compute_attn_scores(q, k)

    # 8. Causal mask
    mask_bytes = B * nh * S * S * x.element_size()   # read+write scores
    r = profiler.benchmark_memory_bound("08 Causal mask", lambda: attn.apply_causal_mask(scores), bytes_accessed=mask_bytes)
    results.append(r)
    scores = attn.apply_causal_mask(scores)

    # 9. Softmax
    sm_bytes = 2 * B * nh * S * S * x.element_size()
    r = profiler.benchmark_memory_bound("09 Softmax", lambda: attn.attn_softmax(scores), bytes_accessed=sm_bytes)
    results.append(r)
    weights = attn.attn_softmax(scores)

    # 10. Attn @ V  —  batched GEMM: [B*nh, S, S] @ [B*nh, S, hd] -> [B*nh, S, hd]
    av_flops = 2 * B * nh * S * S * hd
    r = profiler.benchmark_compute_bound("10 Attn@V", lambda: attn.attn_v_multiply(weights, v), num_flops=av_flops)
    results.append(r)
    attn_out = attn.attn_v_multiply(weights, v)

    # 11. Output projection  —  GEMM: [B*S, nh*hd] @ [nh*hd, d]
    o_flops = 2 * B * S * (nh * hd) * d
    r = profiler.benchmark_compute_bound("11 Out proj", lambda: attn.proj_output(attn_out), num_flops=o_flops)
    results.append(r)

    print_comparison_table(results, "GQA Operation Breakdown", show_bandwidth=True, show_flops=True)

    return results


def profile_end_to_end(attn: GQAAttention, x: torch.Tensor, profiler: GPUProfiler):
    """Profile the full GQA forward pass as a single unit."""
    B, S, _ = x.shape
    d = attn.d_model
    nh = attn.n_heads
    nkv = attn.n_kv_heads
    hd = attn.head_dim

    # Total FLOPs: Q/K/V projections + QK^T + AttnV + output projection
    proj_flops = 2 * B * S * d * (nh * hd + 2 * nkv * hd + d)   # Q + K + V + O
    attn_flops = 2 * 2 * B * nh * S * S * hd                     # QK^T + AttnV
    total_flops = proj_flops + attn_flops

    r = profiler.benchmark_compute_bound(
        f"GQA full (S={S})",
        lambda: attn(x),
        num_flops=total_flops,
    )
    return r


def compare_gqa_vs_mha(configs, seq_len: int, batch: int, profiler: GPUProfiler):
    """Compare GQA (8 KV heads) vs MHA (all KV heads) at the same model size."""
    print(f"\n{'='*70}")
    print(f" GQA vs MHA Comparison  (B={batch}, S={seq_len})")
    print(f"{'='*70}")

    results = []
    for label, d_model, n_heads, n_kv_heads in configs:
        attn = GQAAttention(d_model, n_heads, n_kv_heads, max_seq_len=seq_len).cuda().to(torch.float16)
        x = torch.randn(batch, seq_len, d_model, device='cuda', dtype=torch.float16)

        nh = n_heads
        nkv = n_kv_heads
        hd = d_model // n_heads
        proj_flops = 2 * batch * seq_len * d_model * (nh * hd + 2 * nkv * hd + d_model)
        attn_flops = 2 * 2 * batch * nh * seq_len * seq_len * hd
        total_flops = proj_flops + attn_flops

        # KV cache size per token (for context)
        kv_cache_per_token = 2 * nkv * hd * 2   # 2 for K+V, *2 bytes for FP16
        r_label = f"{label}  (KV$/tok={kv_cache_per_token}B)"
        r = profiler.benchmark_compute_bound(r_label, lambda a=attn, inp=x: a(inp), num_flops=total_flops)
        results.append(r)

        del attn, x
        torch.cuda.empty_cache()

    print_comparison_table(results, "GQA vs MHA", show_bandwidth=False, show_flops=True)


def sweep_seq_lengths(attn: GQAAttention, d_model: int, batch: int, profiler: GPUProfiler):
    """Show how attention cost grows with sequence length (quadratic in QK^T / AttnV)."""
    print(f"\n{'='*70}")
    print(f" Sequence Length Sweep  (B={batch}, d={d_model})")
    print(f"{'='*70}")

    seq_lengths = [128, 256, 512, 1024, 2048]
    results = []
    nh = attn.n_heads
    nkv = attn.n_kv_heads
    hd = attn.head_dim

    for S in seq_lengths:
        x = torch.randn(batch, S, d_model, device='cuda', dtype=torch.float16)
        proj_flops = 2 * batch * S * d_model * (nh * hd + 2 * nkv * hd + d_model)
        attn_flops = 2 * 2 * batch * nh * S * S * hd
        total_flops = proj_flops + attn_flops

        r = profiler.benchmark_compute_bound(
            f"S={S}",
            lambda inp=x: attn(inp),
            num_flops=total_flops,
        )
        results.append(r)
        del x

    print_comparison_table(results, "GQA vs Sequence Length", show_bandwidth=False, show_flops=True)

    # Show quadratic scaling
    if len(results) >= 2:
        print("\n  Scaling relative to first seq_len:")
        base = results[0].time_ms
        for S, r in zip(seq_lengths, results):
            ratio = r.time_ms / base
            ideal_linear = S / seq_lengths[0]
            ideal_quad = (S / seq_lengths[0]) ** 2
            print(f"    S={S:5d}:  {r.time_ms:.3f} ms  ({ratio:.1f}x)  "
                  f"[linear would be {ideal_linear:.1f}x, quadratic {ideal_quad:.1f}x]")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print(" EXPERIMENT 7: Grouped Query Attention (GQA)")
    print("=" * 70)

    get_gpu_info()

    # --- Llama 3 8B config ---
    D_MODEL     = 4096
    N_HEADS     = 32
    N_KV_HEADS  = 8
    SEQ_LEN     = 512
    BATCH       = 2

    print(f"\n  Config (Llama 3 8B attention):")
    print(f"    d_model      = {D_MODEL}")
    print(f"    n_heads (Q)  = {N_HEADS}")
    print(f"    n_kv_heads   = {N_KV_HEADS}")
    print(f"    head_dim     = {D_MODEL // N_HEADS}")
    print(f"    Q per KV grp = {N_HEADS // N_KV_HEADS}")
    print(f"    seq_len      = {SEQ_LEN}")
    print(f"    batch        = {BATCH}")
    print(f"    dtype        = float16")

    # Build module
    attn = GQAAttention(D_MODEL, N_HEADS, N_KV_HEADS, max_seq_len=2048).cuda().half()
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device='cuda', dtype=torch.float16)

    profiler = GPUProfiler(warmup_iters=5, measure_iters=20)

    # ---- Phase 1: individual operation profiling ----
    print("\n" + "=" * 70)
    print(" Phase 1: Individual Operation Profiling")
    print("=" * 70)

    with torch.no_grad():
        profile_individual_ops(attn, x, profiler)

    # ---- Phase 2: end-to-end forward ----
    print("\n" + "=" * 70)
    print(" Phase 2: End-to-End Forward")
    print("=" * 70)

    with torch.no_grad():
        r = profile_end_to_end(attn, x, profiler)
        print(f"\n  Full GQA forward: {r.time_ms:.3f} ms  ({r.flops_tflops:.2f} TFLOPS)")

    # ---- Phase 3: GQA vs MHA comparison ----
    print("\n" + "=" * 70)
    print(" Phase 3: GQA vs MHA Comparison")
    print("=" * 70)

    configs = [
        ("GQA  (nkv=8)",  D_MODEL, N_HEADS, 8),
        ("MQA  (nkv=1)",  D_MODEL, N_HEADS, 1),
        ("MHA  (nkv=32)", D_MODEL, N_HEADS, 32),
    ]
    with torch.no_grad():
        compare_gqa_vs_mha(configs, SEQ_LEN, BATCH, profiler)

    # ---- Phase 4: sequence length sweep ----
    print("\n" + "=" * 70)
    print(" Phase 4: Sequence Length Scaling")
    print("=" * 70)

    with torch.no_grad():
        sweep_seq_lengths(attn, D_MODEL, BATCH, profiler)

    # ---- Memory summary ----
    print("\n" + "=" * 70)
    print(" Memory Summary")
    print("=" * 70)
    print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Peak VRAM:      {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ---- Analysis ----
    print("\n" + "=" * 70)
    print(" ANALYSIS")
    print("=" * 70)
    print("""
    Key observations:

    1. PROJECTION COST BREAKDOWN:
       - Q projection is the most expensive linear op (d -> n_heads * head_dim)
       - K and V projections are cheap: only 1/4 the size of Q (8 vs 32 heads)
       - Output projection cost matches Q (same matrix dimensions)
       - All four are standard GEMMs — compute-bound at large d_model

    2. ATTENTION SCORE COMPUTATION (Q @ K^T):
       - Batched GEMM: [B*n_heads, S, head_dim] @ [B*n_heads, head_dim, S]
       - FLOPs scale as O(S^2 * head_dim * n_heads) — quadratic in seq_len
       - At S=512, head_dim=128: each head does a 512x128 @ 128x512 matmul
       - Dominates cost at long sequences

    3. GQA vs MHA vs MQA:
       - MHA (32 KV heads): full KV cache, most KV projection compute
       - GQA (8 KV heads):  4x KV cache reduction, minimal accuracy loss
       - MQA (1 KV head):   32x KV cache reduction, some quality degradation
       - KV expansion (repeat) is nearly free (just memory aliasing)
       - Main GQA savings are in KV cache memory, not in compute

    4. SEQUENCE LENGTH SCALING:
       - Projections scale linearly with S (GEMM over tokens)
       - Q@K^T and Attn@V scale quadratically with S
       - At short S: projections dominate (compute-bound)
       - At long S: attention scores dominate (memory-bound for decode)

    5. RoPE OVERHEAD:
       - Cheap element-wise ops (sin/cos multiply)
       - Memory-bound: just reads and writes Q and K tensors
       - Negligible compared to projections and attention matmuls

    6. PROFILING WITH NSIGHT:

       nsys profile --trace=cuda,nvtx -o /output/gqa_nsys \\
         python experiments/07_gqa_attention.py

       Look for NVTX ranges: 01_Q_Projection through 11_Output_Projection
       Each operation maps to specific CUDA kernels you can inspect.
    """)


if __name__ == "__main__":
    main()
