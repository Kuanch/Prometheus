# Llama 3 Architecture Notes

Reference: *The Llama 3 Herd of Models*, Meta AI (July 2024)

---

## Overview

Llama 3 is a **dense, decoder-only Transformer** for autoregressive next-token prediction.
It inherits the core design from Llama 1/2 (pre-RMSNorm, SwiGLU FFN, RoPE, no bias, no dropout)
with key upgrades: **Grouped Query Attention (GQA)**, a **128K token vocabulary**, and
**RoPE base frequency of 500,000** for long-context support (up to 128K tokens).

---

## Model Configurations (Table 3 from paper)

| Hyperparameter         |     8B |      70B |       405B |
|------------------------|-------:|---------:|-----------:|
| Layers                 |     32 |       80 |        126 |
| Model Dimension (d)    |  4,096 |    8,192 |     16,384 |
| FFN Dimension (d_ff)   | 14,336 |   28,672 |     53,248 |
| Attention Heads (Q)    |     32 |       64 |        128 |
| Key/Value Heads (KV)   |      8 |        8 |          8 |
| Head Dimension         |    128 |      128 |        128 |
| Q Heads per KV Group   |      4 |        8 |         16 |
| Vocab Size             |128,000 |  128,000 |    128,000 |
| Context Length          |  128K  |    128K  |      128K  |
| Positional Encoding    | RoPE (theta=500K) | RoPE (theta=500K) | RoPE (theta=500K) |
| Activation             | SwiGLU | SwiGLU   | SwiGLU     |
| Peak Learning Rate     | 3e-4   | 1.5e-4   | 8e-5       |
| Training Precision     | BF16   | BF16     | BF16       |

- **Head Dimension** = Model Dimension / Attention Heads = 128 for all sizes
- **Q Heads per KV Group** = Attention Heads / KV Heads (GQA grouping ratio)

---

## Architecture Diagram (ASCII)

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                     INPUT TOKEN IDs                             │
 │                  [batch, seq_len]                                │
 └────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                  TOKEN EMBEDDING                                │
 │         Embedding(128,000 vocab, d_model)                       │
 │         Output: [batch, seq_len, d_model]                       │
 │         (no positional embedding added here; RoPE applied       │
 │          inside each attention layer)                            │
 └────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │              TRANSFORMER DECODER BLOCK  (× N_layers)            │
 │   ┌─────────────────────────────────────────────────────────┐   │
 │   │                                                         │   │
 │   │  ┌───────────────────────────────────────────────────┐   │   │
 │   │  │             RMSNorm (pre-norm)                    │   │   │
 │   │  │           normalize over d_model                  │   │   │
 │   │  └───────────────────────┬───────────────────────────┘   │   │
 │   │                          │                               │   │
 │   │                          ▼                               │   │
 │   │  ┌───────────────────────────────────────────────────┐   │   │
 │   │  │      GROUPED QUERY ATTENTION (GQA)               │   │   │
 │   │  │                                                   │   │   │
 │   │  │  Q = x @ W_q   →  [batch, seq, n_heads, d_head]  │   │   │
 │   │  │  K = x @ W_k   →  [batch, seq, n_kv,    d_head]  │   │   │
 │   │  │  V = x @ W_v   →  [batch, seq, n_kv,    d_head]  │   │   │
 │   │  │                                                   │   │   │
 │   │  │  Apply RoPE to Q and K                            │   │   │
 │   │  │  Expand K,V:  each KV head shared by              │   │   │
 │   │  │    (n_heads / n_kv) query heads                   │   │   │
 │   │  │                                                   │   │   │
 │   │  │  Attention = softmax(Q @ K^T / sqrt(d_head)) @ V  │   │   │
 │   │  │  (causal mask + document mask applied)            │   │   │
 │   │  │                                                   │   │   │
 │   │  │  Output = concat(heads) @ W_o                     │   │   │
 │   │  │         → [batch, seq, d_model]                   │   │   │
 │   │  └───────────────────────┬───────────────────────────┘   │   │
 │   │                          │                               │   │
 │   │                   ┌──────┤  (residual connection)        │   │
 │   │                   │ x + attn_out                         │   │
 │   │                   ▼                                      │   │
 │   │  ┌───────────────────────────────────────────────────┐   │   │
 │   │  │             RMSNorm (pre-norm)                    │   │   │
 │   │  │           normalize over d_model                  │   │   │
 │   │  └───────────────────────┬───────────────────────────┘   │   │
 │   │                          │                               │   │
 │   │                          ▼                               │   │
 │   │  ┌───────────────────────────────────────────────────┐   │   │
 │   │  │             SwiGLU FFN                            │   │   │
 │   │  │                                                   │   │   │
 │   │  │  gate = x @ W_gate  → [batch, seq, d_ff]         │   │   │
 │   │  │  up   = x @ W_up    → [batch, seq, d_ff]         │   │   │
 │   │  │                                                   │   │   │
 │   │  │  hidden = SiLU(gate) ⊙ up                        │   │   │
 │   │  │                                                   │   │   │
 │   │  │  out  = hidden @ W_down → [batch, seq, d_model]  │   │   │
 │   │  └───────────────────────┬───────────────────────────┘   │   │
 │   │                          │                               │   │
 │   │                   ┌──────┤  (residual connection)        │   │
 │   │                   │ x + ffn_out                          │   │
 │   │                   ▼                                      │   │
 │   │            output of this block                          │   │
 │   │         → input to next block                            │   │
 │   └─────────────────────────────────────────────────────────┘   │
 └────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                     FINAL RMSNorm                               │
 │                normalize over d_model                            │
 └────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                   OUTPUT PROJECTION (lm_head)                   │
 │            Linear(d_model → 128,000 vocab, no bias)             │
 │            Output: [batch, seq_len, 128,000]                    │
 └────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
                      logits → softmax → next-token probabilities
```

---

## Component Details

### 1. Token Embedding

- Lookup table: `128,000 × d_model`
- Vocabulary: 100K tiktoken tokens + 28K non-English tokens
- Compression rate: ~3.94 characters/token (English)
- No positional embedding added at this stage (RoPE is applied inside attention)

### 2. RMSNorm (Pre-Normalization)

Applied **before** each sub-layer (attention and FFN), not after. This is the "pre-norm"
variant used since Llama 1.

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```

- `gamma`: learnable per-element scale, shape `[d_model]`
- `eps`: small constant for numerical stability
- No bias, no mean subtraction (unlike LayerNorm)

### 3. Grouped Query Attention (GQA)

GQA reduces KV cache size while retaining most of multi-head attention quality.

**Projections (all without bias):**

| Projection | Input → Output                  | Parameters          |
|------------|---------------------------------|---------------------|
| W_q        | d_model → n_heads × d_head      | d × (n_heads × 128) |
| W_k        | d_model → n_kv × d_head         | d × (n_kv × 128)   |
| W_v        | d_model → n_kv × d_head         | d × (n_kv × 128)   |
| W_o        | n_heads × d_head → d_model      | (n_heads × 128) × d |

**Concrete dimensions per model:**

| Model | W_q               | W_k            | W_v            | W_o               |
|-------|--------------------|-----------------|-----------------|--------------------|
| 8B    | 4096 → 4096       | 4096 → 1024    | 4096 → 1024    | 4096 → 4096       |
| 70B   | 8192 → 8192       | 8192 → 1024    | 8192 → 1024    | 8192 → 8192       |
| 405B  | 16384 → 16384     | 16384 → 1024   | 16384 → 1024   | 16384 → 16384     |

KV cache reduction: instead of storing K,V for all Q heads, only 8 KV heads are stored.
Each KV head is shared across `n_heads / 8` query heads.

**KV Cache size per token per layer:**
- 8B:   2 × 8 × 128 = 2,048 elements
- 70B:  2 × 8 × 128 = 2,048 elements
- 405B: 2 × 8 × 128 = 2,048 elements

(Same KV cache footprint per layer across all sizes due to constant 8 KV heads.)

### 4. Rotary Position Embeddings (RoPE)

Applied to Q and K tensors **after** linear projection, **before** attention score computation.

- Base frequency θ = 500,000 (increased from Llama 2's 10,000)
- Enables extrapolation to longer contexts
- Operates on pairs of dimensions in the head dimension (d_head = 128, so 64 rotation pairs)
- Context length extended from 8K → 128K during continued pre-training

### 5. Causal + Document Attention Mask

Two masking mechanisms combined:
1. **Causal mask**: standard autoregressive mask — token at position i can only attend to positions ≤ i
2. **Document mask**: prevents attention between different documents packed in the same training sequence (important for long-context training)

### 6. SwiGLU Feed-Forward Network

A gated FFN variant with three weight matrices (all without bias):

```
FFN(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
```

where SiLU(z) = z × sigmoid(z) (also known as Swish).

**Dimensions per model:**

| Model | W_gate           | W_up             | W_down           | Total FFN Params     |
|-------|------------------|------------------|------------------|----------------------|
| 8B    | 4096 → 14,336    | 4096 → 14,336    | 14,336 → 4096    | 3 × 4096 × 14336 = 176M |
| 70B   | 8192 → 28,672    | 8192 → 28,672    | 28,672 → 8192    | 3 × 8192 × 28672 = 704M |
| 405B  | 16384 → 53,248   | 16384 → 53,248   | 53,248 → 16384   | 3 × 16384 × 53248 = 2,617M |

Note: The FFN dimension is ~3.5× the model dimension (the standard SwiGLU ratio of 8/3 × d
rounded to a convenient number).

### 7. Output Head

- **Final RMSNorm**: applied to the output of the last transformer block
- **Linear projection**: `d_model → 128,000` (no bias) to produce logits over the vocabulary

---

## Data Flow Summary (Single Forward Pass)

```
tokens [B, S]
   │
   ▼  Embed(128K, d)
hidden [B, S, d]
   │
   ▼  ×N_layers { RMSNorm → GQA(+RoPE) → Residual → RMSNorm → SwiGLU → Residual }
hidden [B, S, d]
   │
   ▼  RMSNorm
hidden [B, S, d]
   │
   ▼  Linear(d, 128K)
logits [B, S, 128K]
```

---

## Parameter Count Breakdown (405B as example)

| Component               | Per-Layer Params | Count   | Total           |
|-------------------------|------------------|---------|-----------------|
| W_q (16384×16384)       | 268M             | ×126    | 33.8B           |
| W_k (16384×1024)        | 16.8M            | ×126    | 2.1B            |
| W_v (16384×1024)        | 16.8M            | ×126    | 2.1B            |
| W_o (16384×16384)       | 268M             | ×126    | 33.8B           |
| Attention RMSNorm       | 16.4K            | ×126    | 2.1M            |
| W_gate (16384×53248)    | 873M             | ×126    | 110.0B          |
| W_up (16384×53248)      | 873M             | ×126    | 110.0B          |
| W_down (53248×16384)    | 873M             | ×126    | 110.0B          |
| FFN RMSNorm             | 16.4K            | ×126    | 2.1M            |
| Token Embedding         |                  | ×1      | 2.1B (128K×16384) |
| Final RMSNorm           |                  | ×1      | 16.4K           |
| Output Projection       |                  | ×1      | 2.1B (16384×128K) |
| **TOTAL**               |                  |         | **~405B**       |

---

## Key Design Decisions (from the paper)

1. **Dense architecture over MoE** — chose simplicity and training stability over potential
   efficiency gains from Mixture of Experts.

2. **GQA with 8 KV heads** — dramatically reduces KV cache memory (16× reduction for 405B
   vs full MHA), enabling efficient long-context inference.

3. **128K vocabulary** — larger vocab improves compression ratio (fewer tokens for same text),
   effectively giving the model more "reading capacity" per training FLOP.

4. **RoPE θ=500,000** — higher base frequency supports context extension to 128K tokens.
   Context length is extended in stages during continued pre-training.

5. **Document attention mask** — prevents cross-contamination between documents packed in
   the same sequence, critical for long-context training quality.

6. **No bias terms, no dropout** — simplifies the model and improves training stability at scale.

7. **Pre-normalization (RMSNorm before sub-layers)** — improves training stability compared
   to post-norm (original Transformer).

---

## Inference Optimizations (from paper)

- **Pipeline parallelism**: first stage holds only embedding, last stage holds only output
  projection + loss — remaining layers distributed evenly
- **FP8 quantization**: applied to FFN layers (W_gate, W_up, W_down) which account for ~50%
  of inference compute; attention layers kept in BF16 for quality
- **KV cache efficiency**: GQA with 8 KV heads means KV cache is the same size regardless
  of model scale (2 × 8 × 128 = 2048 elements per token per layer)

---

## Quick Reference: Tensor Shapes Through the Network (405B)

```
Input:         [B, S]                     (token IDs)
After Embed:   [B, S, 16384]              (hidden states)

In each of 126 transformer layers:
  RMSNorm:     [B, S, 16384]
  Q proj:      [B, S, 128, 128]           (128 query heads × 128 dim)
  K proj:      [B, S, 8, 128]             (8 KV heads × 128 dim)
  V proj:      [B, S, 8, 128]             (8 KV heads × 128 dim)
  +RoPE on Q,K
  Attn out:    [B, S, 128, 128] → concat → [B, S, 16384]
  W_o proj:    [B, S, 16384]
  +Residual:   [B, S, 16384]

  RMSNorm:     [B, S, 16384]
  Gate proj:   [B, S, 53248]
  Up proj:     [B, S, 53248]
  SiLU(gate)⊙up: [B, S, 53248]
  Down proj:   [B, S, 16384]
  +Residual:   [B, S, 16384]

Final RMSNorm: [B, S, 16384]
Output proj:   [B, S, 128000]             (logits)
```
