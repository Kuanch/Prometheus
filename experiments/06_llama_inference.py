#!/usr/bin/env python3
"""
Experiment 6: Llama 3.1 8B Inference with Nsight Profiling

This experiment demonstrates:
1. How a real transformer model executes on GPU
2. Where time is spent during inference (attention, MLP, LayerNorm)
3. The difference between prefill (prompt processing) and decode (token generation)
4. How to use NVTX annotations for fine-grained Nsight profiling

Prerequisites:
    Place Llama 3.1 8B model files at /workspace/llama/:
    - model.safetensors (or model-00001-of-*.safetensors for sharded)
    - tokenizer.model
    - config.json
    - tokenizer_config.json

Usage:
    # Direct run
    python experiments/06_llama_inference.py

    # With Nsight Systems profiling (recommended first)
    nsys profile --trace=cuda,nvtx -o /output/llama_nsys python experiments/06_llama_inference.py

    # With Nsight Compute (deep kernel analysis, slow)
    ncu --set full --launch-count 50 --export /output/llama_ncu python experiments/06_llama_inference.py
"""

import torch
import time
import sys
import os
from contextlib import contextmanager

sys.path.insert(0, '/workspace')
from utils.profiler import get_gpu_info

# ============================================================================
# NVTX Annotation Helpers
# ============================================================================

def nvtx_range_push(name: str):
    """Push an NVTX range onto the stack (visible in Nsight timeline)"""
    torch.cuda.nvtx.range_push(name)


def nvtx_range_pop():
    """Pop the current NVTX range"""
    torch.cuda.nvtx.range_pop()


@contextmanager
def nvtx_range(name: str):
    """Context manager for NVTX range annotations"""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


# ============================================================================
# Layer-level Profiling via Hooks
# ============================================================================

class LayerProfiler:
    """
    Hooks into transformer layers to add NVTX annotations.
    In Nsight timeline, you'll see nested ranges for each layer and sublayer.
    """

    def __init__(self, model):
        self.handles = []
        self._install_hooks(model)

    def _install_hooks(self, model):
        # Hook into each transformer decoder layer
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            print("  Warning: Could not find transformer layers for profiling hooks")
            return

        for i, layer in enumerate(layers):
            # Pre-hook: push NVTX range before layer forward
            handle = layer.register_forward_pre_hook(
                self._make_pre_hook(f"Layer_{i}")
            )
            self.handles.append(handle)

            # Post-hook: pop NVTX range after layer forward
            handle = layer.register_forward_hook(
                self._make_post_hook(f"Layer_{i}")
            )
            self.handles.append(handle)

            # Hook into sublayers if accessible
            self._hook_sublayers(layer, i)

    def _hook_sublayers(self, layer, layer_idx):
        """Add NVTX hooks to attention, MLP, and norm sublayers"""
        sublayer_names = {
            'self_attn': 'Attention',
            'mlp': 'MLP',
            'input_layernorm': 'InputNorm',
            'post_attention_layernorm': 'PostAttnNorm',
        }

        for attr_name, display_name in sublayer_names.items():
            sublayer = getattr(layer, attr_name, None)
            if sublayer is not None:
                handle = sublayer.register_forward_pre_hook(
                    self._make_pre_hook(f"L{layer_idx}_{display_name}")
                )
                self.handles.append(handle)

                handle = sublayer.register_forward_hook(
                    self._make_post_hook(f"L{layer_idx}_{display_name}")
                )
                self.handles.append(handle)

    @staticmethod
    def _make_pre_hook(name):
        def hook(module, input):
            torch.cuda.nvtx.range_push(name)
        return hook

    @staticmethod
    def _make_post_hook(name):
        def hook(module, input, output):
            torch.cuda.nvtx.range_pop()
        return hook

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path: str):
    """
    Load Llama 3.1 8B with 4-bit quantization.

    4-bit NF4 quantization reduces VRAM from ~16GB to ~4-5GB,
    fitting within RTX 4060's 8GB.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\nLoading model from: {model_path}")

    with nvtx_range("Model_Loading"):
        # 4-bit quantization config
        with nvtx_range("Quantization_Config"):
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",           # NormalFloat4
                bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16
                bnb_4bit_use_double_quant=True,        # Double quantization
            )

        # Load tokenizer
        with nvtx_range("Load_Tokenizer"):
            start = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer_time = time.perf_counter() - start
            print(f"  Tokenizer loaded: {tokenizer_time:.2f}s")

        # Load model with quantization
        with nvtx_range("Load_Model_Weights"):
            start = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            model_time = time.perf_counter() - start
            print(f"  Model loaded: {model_time:.2f}s")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params / 1e9:.2f}B")
        print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  VRAM reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    return model, tokenizer


# ============================================================================
# Inference with Profiling
# ============================================================================

def run_prefill(model, tokenizer, prompt: str):
    """
    Prefill phase: process entire prompt in parallel.

    This is compute-bound (matrix multiplications over all tokens).
    In Nsight, you'll see large cuBLAS kernels.
    """
    print(f"\n--- Prefill Phase ---")
    print(f"  Prompt: '{prompt}'")

    with nvtx_range("Tokenization"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        print(f"  Tokens: {seq_len}")

    # Warmup
    with torch.no_grad():
        _ = model(input_ids)
    torch.cuda.synchronize()

    # Timed prefill
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        with nvtx_range("Prefill"):
            start.record()
            outputs = model(input_ids)
            end.record()

    torch.cuda.synchronize()
    prefill_time_ms = start.elapsed_time(end)

    print(f"  Prefill time: {prefill_time_ms:.2f} ms")
    print(f"  Tokens/sec: {seq_len / (prefill_time_ms / 1000):.0f}")

    return outputs, inputs


def run_decode(model, tokenizer, prompt: str, max_new_tokens: int = 50):
    """
    Decode phase: generate tokens one by one (autoregressive).

    This is memory-bound (each step reads all KV-cache + model weights
    but only computes for 1 token). In Nsight, you'll see repeated
    small kernels with memory transfers between them.
    """
    print(f"\n--- Decode Phase (Generation) ---")
    print(f"  Max new tokens: {max_new_tokens}")

    with nvtx_range("Tokenization"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

    # Warmup
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )
    torch.cuda.synchronize()

    # Timed generation
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        with nvtx_range("Full_Generation"):
            start.record()

            # Generate with NVTX annotations per step
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,   # KV-cache for efficient decode
            )

            end.record()

    torch.cuda.synchronize()
    gen_time_ms = start.elapsed_time(end)

    # Decode output
    new_tokens = generated_ids.shape[1] - prompt_len
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(f"  Generated {new_tokens} tokens in {gen_time_ms:.2f} ms")
    print(f"  Decode speed: {new_tokens / (gen_time_ms / 1000):.1f} tokens/sec")
    print(f"  Output: '{output_text}'")

    return generated_ids, gen_time_ms, new_tokens


def run_step_by_step_decode(model, tokenizer, prompt: str, max_new_tokens: int = 20):
    """
    Manual token-by-token generation with per-step NVTX annotations.

    This gives the most detailed view in Nsight: each decode step
    is a separate NVTX range, so you can see exactly how long each
    token takes and what kernels fire.
    """
    print(f"\n--- Step-by-Step Decode (Detailed Profiling) ---")

    with nvtx_range("Tokenization"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

    step_times = []
    past_key_values = None

    with torch.no_grad():
        # Prefill
        with nvtx_range("StepDecode_Prefill"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            outputs = model(input_ids, past_key_values=None, use_cache=True)
            end.record()
            torch.cuda.synchronize()

            prefill_ms = start.elapsed_time(end)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        print(f"  Prefill: {prefill_ms:.2f} ms ({input_ids.shape[1]} tokens)")

        # Decode tokens one by one
        generated_tokens = [next_token.item()]

        for step in range(max_new_tokens - 1):
            with nvtx_range(f"Decode_Step_{step}"):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                outputs = model(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                end.record()
                torch.cuda.synchronize()

                step_ms = start.elapsed_time(end)
                step_times.append(step_ms)

                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_tokens.append(next_token.item())

                # Stop on EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

    # Summary
    if step_times:
        avg_ms = sum(step_times) / len(step_times)
        min_ms = min(step_times)
        max_ms = max(step_times)
        print(f"  Decode steps: {len(step_times)}")
        print(f"  Avg step time: {avg_ms:.2f} ms ({1000/avg_ms:.1f} tokens/sec)")
        print(f"  Min/Max step: {min_ms:.2f} / {max_ms:.2f} ms")

    # Decode output
    all_ids = torch.cat([input_ids, torch.tensor([generated_tokens], device=input_ids.device)], dim=1)
    output_text = tokenizer.decode(all_ids[0], skip_special_tokens=True)
    print(f"  Output: '{output_text}'")

    return step_times


# ============================================================================
# Main
# ============================================================================

def main():
    MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "/workspace/llama")

    print("\n" + "="*70)
    print(" EXPERIMENT 6: Llama 3.1 8B Inference Profiling")
    print("="*70)

    get_gpu_info()

    # Check model files
    if not os.path.isdir(MODEL_PATH):
        print(f"\n  ERROR: Model directory not found at {MODEL_PATH}")
        print(f"  Place Llama 3.1 8B safetensors and tokenizer files there.")
        print(f"  Or set LLAMA_MODEL_PATH environment variable.")
        sys.exit(1)

    # Load model
    model, tokenizer = load_model(MODEL_PATH)

    # Install layer-level profiling hooks
    layer_profiler = LayerProfiler(model)

    # Test prompts
    prompts = [
        "The capital of France is",
        "Explain how a GPU processes matrix multiplication in three steps:",
        "Write a Python function that computes the fibonacci sequence:",
    ]

    # Run experiments
    print("\n" + "="*70)
    print(" Phase 1: Prefill Benchmark")
    print("="*70)

    for prompt in prompts:
        run_prefill(model, tokenizer, prompt)

    print("\n" + "="*70)
    print(" Phase 2: Full Generation")
    print("="*70)

    run_decode(model, tokenizer, prompts[0], max_new_tokens=50)

    print("\n" + "="*70)
    print(" Phase 3: Step-by-Step Decode (Detailed)")
    print("="*70)

    step_times = run_step_by_step_decode(model, tokenizer, prompts[0], max_new_tokens=20)

    # Clean up hooks
    layer_profiler.remove_hooks()

    # Memory summary
    print("\n" + "="*70)
    print(" Memory Summary")
    print("="*70)
    print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  VRAM reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"  Peak VRAM:      {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Analysis
    print("\n" + "="*70)
    print(" ANALYSIS")
    print("="*70)
    print("""
    Key observations:

    1. PREFILL vs DECODE:
       - Prefill: processes all prompt tokens in parallel (compute-bound)
       - Decode: generates one token at a time (memory-bound)
       - Prefill uses large batched matmuls → high GPU utilization
       - Decode reads entire KV-cache per step → bandwidth limited

    2. WHERE TIME IS SPENT (use Nsight to verify):
       - Attention (Q@K^T, softmax, @V): ~40% of each layer
       - MLP (2x linear + activation): ~50% of each layer
       - LayerNorm: ~5% (cheap but many launches)
       - Remaining: embedding lookup, output projection

    3. 4-BIT QUANTIZATION EFFECTS:
       - Model weights stored as INT4 → dequantized to FP16 on-the-fly
       - Saves VRAM (4GB vs 16GB) but adds dequantization overhead
       - cuBLAS still handles the core matmuls

    4. PROFILING WITH NSIGHT:

       Nsight Systems (timeline):
         nsys profile --trace=cuda,nvtx -o /output/llama \\
           python experiments/06_llama_inference.py

         → Look for: NVTX ranges (Layer_0..31, Attention, MLP)
         → Look for: gaps between kernels (CPU overhead)
         → Look for: memory transfer bars (KV-cache)

       Nsight Compute (kernel detail):
         ncu --set full --launch-count 50 --export /output/llama_kernels \\
           python experiments/06_llama_inference.py

         → Look for: occupancy, memory throughput
         → Look for: which kernels are memory-bound vs compute-bound
    """)


if __name__ == "__main__":
    main()
