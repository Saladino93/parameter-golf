# # Deep Dive: `train_gpt_mlx.py`
# ## A PhD-level guide to the Parameter Golf baseline
#
# **Goal**: Understand every component of the training script deeply enough to modify it for the challenge.
#
# **Your background**: PhD in cosmology → strong in statistical mechanics, optimization, numerical methods, Fourier analysis, power spectra. We'll use these analogies throughout.
#
# ---
#
# ### Roadmap (the script has ~1100 lines, 7 logical sections)
#
# | Section | Lines | What it does | Physics analogy |
# |---------|-------|--------------|-----------------|
# | **1. Hyperparameters** | 43–130 | All tuneable knobs | Initial conditions of your simulation |
# | **2. Data Pipeline** | 195–273 | Token streaming & batching | Observation pipeline — raw photons → calibrated images |
# | **3. Model Architecture** | 280–453 | The neural network itself | The physical model (e.g., ΛCDM with parameters) |
# | **4. Optimizers** | 457–539 | Muon + Adam split | The MCMC sampler / gradient descent in parameter space |
# | **5. Quantization** | 543–666 | int8 compression for 16MB | Lossy data compression (like FITS → JPEG for thumbnails) |
# | **6. Evaluation** | 761–814 | val_loss and val_bpb | The χ² / likelihood evaluation |
# | **7. Training Loop** | 836–1104 | Orchestrates everything | The simulation main loop |
#
# We'll go **bottom-up**: understand the atoms (linear layers, norms) before the molecules (attention, MLP) before the organism (GPT model) before the ecosystem (training loop).

# Setup: run this cell first
import math
import os
import sys

import numpy as np

# Add the repo root so we can import pieces
sys.path.insert(0, os.path.abspath(".."))

import mlx.core as mx
import mlx.nn as nn

# We'll use small dimensions for interactive exploration
COMPUTE_DTYPE = mx.bfloat16  # brain float 16, designed by Google Brain
# 8-bit exponent as fp32. But 7-bit mantissa instead of 23 -> less precision. But 2x faster on Apple Silicon
# mantissa is the number after the . for the logarithm, in some base, of a number, e.g. \log_{10}(147) = 2.1673173, we have mantissa = 1673173
# fp32:    1 sign | 8 exp | 23 mantissa  (32 bits)
# bfloat16: 1 sign | 8 exp |  7 mantissa  (16 bits)
# float16:  1 sign | 5 exp | 10 mantissa  (16 bits)  ← smaller range, riskier

# COMPUTE_DTYPE = mx.bfloat16 means all forward-pass ops (matmuls, attention, activations) run in bf16.
# Weights are typically kept in fp32 for stable gradient
# accumulation, then cast down on each forward pass.
#
# On Apple Silicon: the ANE and GPU both have native
# bf16 units. Using mx.bfloat16 is essentially free
# precision for a 2× memory and compute win.

print(f"MLX version: {mx.__version__}")
print(f"Compute dtype: {COMPUTE_DTYPE}")

# ---
# # Part 1: The Atom — `CastedLinear` (line 280)
#
# The most fundamental building block. Every weight matrix in the model is a `CastedLinear`.
#
# ### What it does
# A linear layer computes $y = xW^T$ (matrix multiplication, no bias). But with a twist:
# - **Weights are stored in fp32** (32-bit float, full precision), for stable gradient accumulation
# - **Computation happens in bf16** (bfloat16, half precision)
#
# ### Why?
# This is like storing your CMB map at full resolution but doing the power spectrum estimation in reduced precision. The key insight:
#
# - **Gradients need full precision** to accumulate tiny updates over thousands of steps (like accumulating weak lensing signal)
# - **Forward pass can tolerate half precision** because the activations are inherently noisy (they'll be normalized anyway)
# - **bf16 is 2x faster** than fp32 on Apple Silicon's matrix units
#
# ### The cosmology analogy
# Think of it as: your simulation grid is stored at double precision, but the FFTs for the power spectrum use single precision. The small errors in the FFT wash out in the ensemble average, but you'd lose information if you stored the density field at reduced precision.

# Let's build and inspect a CastedLinear


class CastedLinear(nn.Module):
    """Stores weights in fp32, computes in the input's dtype (bf16)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Initialize via nn.Linear, then steal the weight and cast to fp32
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        # Cast weight to input dtype (bf16) on the fly, then matmul
        return x @ self.weight.astype(x.dtype).T


# Create a small one: 8-dimensional input, 4-dimensional output
layer = CastedLinear(8, 4)

print(
    f"Weight shape: {layer.weight.shape}"
)  # (4, 8) — note: (out, in), transposed in forward
print(f"Weight dtype: {layer.weight.dtype}")  # float32 — stored precisely
print(f"Weight values (first row):\n{layer.weight[0]}")

# Forward pass in bf16
x = mx.ones((2, 8), dtype=mx.bfloat16)  # batch=2, dim=8
y = layer(x)
print(f"\nInput dtype:  {x.dtype}")  # bfloat16
print(f"Output dtype: {y.dtype}")  # bfloat16 — computation happened in bf16
print(f"Output shape: {y.shape}")  # (2, 4)

# ---
# # Part 2: RMSNorm — The Normalizer (line 172)
#
# ### What it does
# RMSNorm normalizes a vector by its root-mean-square magnitude:
#
# $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}$$
# \epsilon = 1e-6, by default, prevents division by zero when a vector is all-zeros.
# It makes every vector have roughly unit RMS magnitude, regardless of what the raw values are.
#
# ### Why is this needed?
# Without normalization, activations can grow or shrink exponentially as they pass through layers. This is the **exploding/vanishing gradient problem** — the deep learning equivalent of numerical instability in long N-body integrations.
#
# RMSNorm is like **renormalizing your field** at each step:
# - In cosmology: you might normalize $\delta(\mathbf{x}) = (\rho - \bar{\rho})/\bar{\rho}$ to keep the overdensity field well-behaved
# - Here: you normalize the hidden state so each layer sees inputs of predictable magnitude
#
# ### Why RMSNorm and not LayerNorm?
# LayerNorm = subtract mean, then divide by std. RMSNorm = just divide by RMS (no mean subtraction).
# RMSNorm is ~30% faster and works just as well. The mean subtraction in LayerNorm is mostly redundant because the subsequent linear layer can learn to shift.
#
# ### The "NoWeight" variant
# Standard RMSNorm has a learnable scale parameter $\gamma$. This script uses `RMSNormNoWeight` — no learnable $\gamma$. The subsequent linear layers absorb that scaling, so the extra parameter is redundant.

# RMSNorm in action
#
# In cosmology, you know that a Gaussian random field has random phases — all structure (filaments, voids,the cosmic web) is encoded in phase correlations, not the power spectrum. Same logic applies: the
# "intelligence" of a token embedding is in its direction, not its norm. RMSNorm is acknowledging this.
# This is why the NoWeight variant (no learnable γ) works — γ would only re-introduce amplitude modulation, which the downstream linear layers already handle.


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """Normalize x so its RMS ≈ 1."""
    # Axis=-1 means it normalizes along the embedding dimension (each token independently), not across the
    # sequence.Tokens don't share normalization statistics — they're causally isolated.
    # this is a window function that whitens the amplitude. The gradient flow sees a stationary process
    # even if the raw activations are highly non-stationary
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


# Example: a vector with large magnitude
x = mx.array([[100.0, 200.0, 300.0, 400.0]], dtype=mx.float32)
x_normed = rms_norm(x)

print(f"Before RMSNorm: {x}")
print(f"  RMS magnitude: {float(mx.sqrt(mx.mean(x * x))):.2f}")
print(f"\nAfter RMSNorm:  {x_normed}")
print(f"  RMS magnitude: {float(mx.sqrt(mx.mean(x_normed * x_normed))):.4f}")
print(f"\nNote: direction is preserved, magnitude is ~1")
print(f"Ratio x[0]/x[1] before: {float(x[0, 0] / x[0, 1]):.2f}")
print(f"Ratio x[0]/x[1] after:  {float(x_normed[0, 0] / x_normed[0, 1]):.2f}  ← same!")

# ---
# # Part 3: Rotary Position Embeddings (RoPE) — line 324
#
# ### The problem
# Attention (which we'll see next) treats tokens as a **set** — it has no notion of order. "The cat sat on the mat" and "mat the on sat cat the" would produce identical attention patterns without positional information.
#
# ### The solution: RoPE
# RoPE encodes position by **rotating** the query and key vectors in 2D subspaces. For position $p$ and dimension pair $(2i, 2i+1)$:
#
# $$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(p \cdot \theta_i) & -\sin(p \cdot \theta_i) \\ \sin(p \cdot \theta_i) & \cos(p \cdot \theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$
#
# where $\theta_i = \text{base}^{-2i/d}$ (here base = 10000).
#
# ### Why this is beautiful (the cosmology connection)
#
# **RoPE is literally a Fourier basis.** Each dimension pair $(2i, 2i+1)$ oscillates at a different frequency $\theta_i$, just like harmonics in a spherical harmonic expansion:
# - Low $i$ → **high** frequency → captures **short-range** position differences (like high-$\ell$ CMB modes)
# - High $i$ → **low** frequency → captures **long-range** position differences (like low-$\ell$ CMB modes)
#
# The key property: $\langle q_p, k_{p'} \rangle$ depends only on the **relative distance** $|p - p'|$, not the absolute positions. This is **translation invariance** — exactly like how the power spectrum $C_\ell$ depends on angular separation, not absolute sky position.
# This cancellation is why RoPE generalizes to longer contexts than the training length — the model only ever sees relative distances, so an unseen absolute position $p=10000$ still produces familiar relative distances $(p'-p)$.
# The frequencies $\theta_i = 10000^{-2i/d}$ are geometric (log-spaced), exactly like $\ell$ modes on the CMB sky covering decades of angular scale. Low-$i$
# = **short** wavelength = fine-grained **local** correlations; high-$i$ = **long** wavelength = **large-scale** structure.
# The cancellation $R(-p)R(p') = R(p'-p)$ is the deep reason: rotating both vectors by the same angle doesn't change their dot product. It's the invariance under simultaneous rotation — the gauge freedom of absolute position.
# ### The `rope_base` parameter
# - `rope_base = 10000` (default) → standard frequency spacing
# - Larger base → lower frequencies → better at long-range dependencies
# - This is analogous to choosing the maximum $\ell_{\max}$ in your harmonic expansion

# Visualize RoPE frequencies — it's literally a Fourier decomposition

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for notebooks
import matplotlib.pyplot as plt

head_dim = 64  # the baseline uses dim=512, 8 heads → 64 per head
rope_base = 10000.0

# Compute the frequencies for each dimension pair
dim_indices = np.arange(0, head_dim, 2)  # pairs: (0,1), (2,3), ..., (62,63)
thetas = rope_base ** (-dim_indices / head_dim)

# For each position p, the rotation angle is p * theta_i
positions = np.arange(1024)  # sequence length

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: frequency spectrum (like plotting ℓ(ℓ+1)C_ℓ)
axes[0].semilogy(dim_indices // 2, thetas, "b.-")
axes[0].set_xlabel("Dimension pair index i")
axes[0].set_ylabel("Frequency θ_i (radians/position)")
axes[0].set_title("RoPE frequency spectrum\n(cf. harmonic ℓ-modes)")
axes[0].grid(True, alpha=0.3)

# Right: rotation angles for a few dimension pairs
for i, label in [
    (0, "i=0 (highest freq)"),
    (8, "i=8"),
    (16, "i=16"),
    (31, "i=31 (lowest freq)"),
]:
    angles = positions * thetas[i]
    axes[1].plot(positions[:100], np.cos(angles[:100]), label=label, alpha=0.7)
axes[1].set_xlabel("Token position p")
axes[1].set_ylabel("cos(p · θ_i)")
axes[1].set_title("RoPE rotation by position\n(like harmonics at different ℓ)")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rope_frequencies.png", dpi=100, bbox_inches="tight")
plt.show()
print("✓ Higher frequencies (small i) vary rapidly → capture short-range/local structure")
print("✓ Lower frequencies (large i) vary slowly → capture long-range structure")

# ---
# # Part 4: Causal Self-Attention — The Core Mechanism (line 295)
#
# This is the heart of the transformer. Every other component exists to support attention.
#
# ### The big picture
#
# Given a sequence of tokens, attention lets each token **look at all previous tokens** and decide which ones are relevant for predicting the next token. It's a learned, content-dependent weighting scheme.
#
# ### Step by step
#
# For each token at position $p$, we compute three vectors from its hidden state $x_p$:
# - **Query** $q_p = W_Q x_p$ — "What am I looking for?"
# - **Key** $k_p = W_K x_p$ — "What do I contain?"
# - **Value** $v_p = W_V x_p$ — "What information do I provide?"
#
# Then for each query position $p$:
# 1. Compute attention scores: $s_{p,j} = q_p \cdot k_j / \sqrt{d}$ for all $j \leq p$ (causal: can't look at future)
# 2. Apply softmax: $\alpha_{p,j} = \text{softmax}(s_{p,:})_j$ — these are the attention weights, they sum to 1
# 3. Weighted sum: $\text{out}_p = \sum_j \alpha_{p,j} \cdot v_j$
#
# ### The cosmology analogy
#
# Attention is like **optimal filtering / matched filtering**:
# - The query is your template (what pattern you're searching for)
# - The keys are the data at each position (what's there)
# - The dot product $q \cdot k$ is the correlation (how well the template matches)
# - The softmax converts correlations to weights (like a likelihood → posterior)
# - The weighted sum of values is your filtered estimate
#
# Or think of it as an **N-body interaction**: each particle (token) feels a force (attention weight) from every previous particle, weighted by how "relevant" they are (the $q \cdot k$ score). The causal mask makes it a retarded Green's function — information only flows forward in the sequence.
#
# ### Multi-head attention
#
# Instead of one set of Q, K, V, we have **8 heads** (in the baseline). Each head has its own Q, K, V projections operating on a slice of the 512-dim hidden state (64 dims per head). This lets different heads specialize:
# - Some heads might track syntactic structure (subject-verb agreement)
# - Others might track semantic meaning (topic coherence)
# - Like having multiple independent estimators that get combined
#
# ### Grouped Query Attention (GQA)
#
# The baseline uses 8 query heads but only **4 KV heads**. Each pair of query heads shares one KV head. This saves parameters (and memory for KV caching) with minimal quality loss. It's like using a lower-resolution grid for the gravitational potential while keeping the particle positions at full resolution.
#
# ### The learnable `q_gain`
#
# After RMSNorm on Q, the queries are multiplied by a learnable per-head gain factor (initialized to 1.5). This controls the **temperature** of the attention distribution — higher gain → sharper attention (more peaked softmax) → the model focuses on fewer tokens. It's like adjusting the width of your matched filter kernel.

# Let's build attention from scratch and trace the shapes


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim

        self.c_q = CastedLinear(dim, dim)  # Q: full rank
        self.c_k = CastedLinear(dim, kv_dim)  # K: reduced rank (GQA)
        self.c_v = CastedLinear(dim, kv_dim)  # V: reduced rank (GQA)
        self.proj = CastedLinear(dim, dim)  # output projection
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim**-0.5  # 1/sqrt(d_head)

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape

        # Project to Q, K, V
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # Rearrange to (batch, heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Normalize Q and K, then apply RoPE
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))

        # Apply per-head gain to queries
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]

        # Scaled dot-product attention with causal mask
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask="causal"
        )

        # Reshape back and project
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


# Create and trace shapes
dim, num_heads, num_kv_heads = 32, 4, 2  # small for demo
attn = CausalSelfAttention(
    dim, num_heads, num_kv_heads, rope_base=10000.0, qk_gain_init=1.5
)

x = mx.ones((1, 8, dim), dtype=mx.bfloat16)  # batch=1, seq=8, dim=32
out = attn(x)

print("=== Attention Shape Trace ===")
print(f"Input:          (batch=1, seq=8, dim={dim})")
print(
    f"Q projection:   (1, 8, {dim}) → (1, {num_heads}, 8, {dim // num_heads})  [{num_heads} heads × {dim // num_heads} dims]"
)
print(
    f"K projection:   (1, 8, {dim}) → (1, {num_kv_heads}, 8, {dim // num_heads})  [{num_kv_heads} KV heads × {dim // num_heads} dims]"
)
print(
    f"V projection:   (1, 8, {dim}) → (1, {num_kv_heads}, 8, {dim // num_heads})  [same as K]"
)
print(f"Attention out:  (1, {num_heads}, 8, {dim // num_heads})")
print(f"After reshape:  (1, 8, {dim})")
print(f"After proj:     {out.shape}")
print(f"\nParameter count:")
q_params = dim * dim  # c_q
k_params = dim * (num_kv_heads * (dim // num_heads))  # c_k
v_params = k_params  # c_v
proj_params = dim * dim  # proj
print(f"  Q: {q_params}, K: {k_params}, V: {v_params}, proj: {proj_params}")
print(f"  Total: {q_params + k_params + v_params + proj_params + num_heads}")
print(
    f"  GQA savings: {1 - (k_params + v_params) / (2 * q_params):.0%} fewer KV params"
)

# ---
# # Part 5: The MLP — The Nonlinear Processor (line 341)
#
# ### What it does
# Each transformer block has two sub-layers: attention (global mixing) and MLP (local processing). The MLP processes each token independently through a nonlinear transform:
#
# $$\text{MLP}(x) = W_{\text{proj}} \cdot (\text{ReLU}(W_{\text{fc}} \cdot x))^2$$
#
# ### The architecture
# 1. **Expand**: $W_{\text{fc}}$ projects from dim (512) to hidden (1024 = dim × `mlp_mult`)
# 2. **Activate**: Apply ReLU then square it ($\text{relu}^2$)
# 3. **Contract**: $W_{\text{proj}}$ projects back from 1024 to 512
#
# ### Why relu²?
# - Standard ReLU: $f(x) = \max(0, x)$ — cheap but has a hard discontinuity at 0
# - relu²: $f(x) = (\max(0, x))^2$ — smooth at 0, and the squaring provides stronger nonlinearity
# - It's cheaper to compute than GELU or SiLU (which need exp/erf), and works well for small models
#
# ### The role of MLP vs Attention
# Think of the transformer as alternating between two operations:
# - **Attention**: "What information from other positions is relevant?" (inter-token communication)
# - **MLP**: "Given the gathered information, what should I compute?" (per-token processing)
#
# The analogy: attention is like a telescope pointing at different parts of the sky (choosing where to look), and the MLP is like the CCD + readout electronics (processing what was observed).
#
# ### Parameter budget
# The MLP is **expensive**. With `mlp_mult=2`:
# - $W_{\text{fc}}$: 512 × 1024 = 524,288 parameters
# - $W_{\text{proj}}$: 1024 × 512 = 524,288 parameters
# - Total: 1,048,576 per block, vs ~786,432 for attention
#
# Leaderboard entries use `mlp_mult=3` — even more parameters in the MLP, funded by aggressive quantization. This is a key competition lever.

# Compare activation functions

x_range = np.linspace(-3, 3, 200)

relu = np.maximum(0, x_range)
relu_sq = relu**2
gelu = x_range * 0.5 * (1 + np.vectorize(math.erf)(x_range / math.sqrt(2)))

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(x_range, relu, label="ReLU: max(0, x)", alpha=0.7, linewidth=2)
ax.plot(x_range, relu_sq, label="relu²: max(0, x)²", alpha=0.7, linewidth=2)
ax.plot(
    x_range, gelu, label="GELU (used in GPT-2)", alpha=0.7, linewidth=2, linestyle="--"
)
ax.set_xlabel("Input x")
ax.set_ylabel("Activation f(x)")
ax.set_title("Activation functions: relu² is cheap and smooth")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.5, 5)
plt.tight_layout()
plt.savefig("activations.png", dpi=100, bbox_inches="tight")
plt.show()

print("relu² key properties:")
print("  - Zero for x < 0 (sparse activation, saves compute)")
print("  - Smooth at x=0 (no discontinuity in gradient)")
print("  - Quadratic growth (strong nonlinearity)")

# ---
# # Part 6: The Transformer Block (line 354)
#
# ### Residual connections: the most important idea in deep learning
#
# A Block wraps attention + MLP with **residual connections**:
#
# ```
# x_mixed = resid_mix[0] * x + resid_mix[1] * x0     # blend with original embedding
# x = x + attn_scale * attention(norm(x_mixed))        # residual add from attention
# x = x + mlp_scale * mlp(norm(x))                     # residual add from MLP
# ```
#
# The `+` is the residual connection. Without it, information has to pass *through* every layer — deep networks become impossible to train (vanishing gradients). With it, gradients can flow directly backward through the `+` addition, like a highway bypass.
#
# ### The cosmology analogy
# Residual connections are like **perturbation theory**:
# - $x$ is your background (zeroth order)
# - `attention(norm(x))` is the first-order perturbation
# - The `attn_scale` learned parameter controls the coupling strength
#
# Each layer adds a small perturbation to the residual stream, just like computing $\delta \rho / \rho$ to first order, then second order, etc.
#
# ### Three learnable control parameters per block
#
# | Parameter | Shape | Init | Purpose |
# |-----------|-------|------|---------|
# | `attn_scale` | (512,) | ones | Per-dimension scaling of attention output |
# | `mlp_scale` | (512,) | ones | Per-dimension scaling of MLP output |
# | `resid_mix` | (2, 512) | [ones, zeros] | Blend current residual with original embedding x0 |
#
# `resid_mix` is particularly clever: it lets each layer decide how much to "reset" toward the original token embedding vs. keeping the processed representation. Initialized to [1, 0] = "keep current x, ignore x0".

# ---
# # Part 7: The GPT Model — U-Net Skip Connections (line 382)
#
# ### Overall architecture
#
# ```
# Input tokens → Embedding → RMSNorm → [Encoder blocks] → [Decoder blocks] → Final Norm → Logits
#                                           ↓                    ↑
#                                         skip₁ ─────────────→ skip₁ (reversed)
#                                         skip₂ ─────────→ skip₂
#                                         skip₃ ────→ skip₃
# ```
#
# The 9 blocks are split into:
# - **Encoder** (first 4 blocks): processes tokens and saves intermediate states as "skips"
# - **Decoder** (last 5 blocks): processes tokens and adds back the saved skips (in reverse order)
#
# ### Why U-Net skips?
#
# This is borrowed from image segmentation (U-Net, 2015). The insight: early layers capture local/surface features, later layers capture abstract/global features. By connecting early to late via skip connections, the decoder has access to both.
#
# ### The cosmology analogy
#
# Think of a wavelet decomposition of a CMB map:
# - **Encoder**: decomposes the signal into scales (large → small)
# - **Skip connections**: preserve the detail coefficients at each scale
# - **Decoder**: reconstructs using coarse + detail, from large scale back to small
#
# Without skips, the decoder would only see the coarsest representation. With skips, it can combine coarse global understanding with fine local detail.
#
# ### Learnable `skip_weights`
# Each skip connection has a per-dimension weight (shape 512). This lets the model learn how much of the encoder's output to mix into the decoder. Initialized to ones = "pass everything through".
#
# ### Tied embeddings and the logit softcap
#
# The output logits are computed as:
# $$\text{logits} = c \cdot \tanh\left(\frac{x \cdot W_{\text{emb}}^T}{c}\right)$$
#
# where $c = 30$ is the softcap. Two important design choices:
#
# 1. **Tied embeddings**: The same weight matrix $W_{\text{emb}}$ is used for both input embedding and output projection. This halves the embedding parameter count — critical for a 16MB budget.
#
# 2. **Logit softcap**: The $c \cdot \tanh(\cdot/c)$ bounds logits to $[-c, c]$. Without this, logits can grow unboundedly, causing numerical instability. It's like a regularizer on the confidence — the model can never be "infinitely sure" about any token.

# Build the full GPT and count parameters by component
from mlx.utils import tree_flatten


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc(x))
        return self.proj(x * x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(
            np.stack([np.ones(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32)])
        )

    def __call__(self, x, x0):
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * self.attn(rms_norm(x))
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(rms_norm(x))
        return x


# Use the ACTUAL baseline dimensions
dim = 512
num_layers = 9
vocab_size = 1024
num_heads = 8
num_kv_heads = 4
mlp_mult = 2

# Count parameters
embed_params = vocab_size * dim  # tied embedding
block_attn = (
    dim * dim + dim * (num_kv_heads * (dim // num_heads)) * 2 + dim * dim
)  # Q + K + V + proj
block_attn += num_heads  # q_gain
block_mlp = dim * (dim * mlp_mult) + (dim * mlp_mult) * dim  # fc + proj
block_control = dim + dim + 2 * dim  # attn_scale + mlp_scale + resid_mix
block_total = block_attn + block_mlp + block_control
skip_params = (num_layers // 2) * dim

total = embed_params + num_layers * block_total + skip_params

print("═══ Parameter Budget Breakdown ═══")
print(
    f"\nEmbedding (tied):     {embed_params:>10,} params  ({embed_params / total:.1%})"
)
print(f"\nPer block:")
print(f"  Attention (Q+K+V+proj+gain): {block_attn:>10,} params")
print(f"  MLP (fc + proj):             {block_mlp:>10,} params")
print(f"  Control (scales + mix):      {block_control:>10,} params")
print(f"  Block total:                 {block_total:>10,} params")
print(
    f"\n{num_layers} blocks × {block_total:,}: {num_layers * block_total:>10,} params  ({num_layers * block_total / total:.1%})"
)
print(f"Skip weights:         {skip_params:>10,} params  ({skip_params / total:.1%})")
print(f"\n{'─' * 45}")
print(f"TOTAL:                {total:>10,} params")
print(f"\nAt fp32 (4 bytes each): {total * 4 / 1e6:.1f} MB")
print(f"At int8 (1 byte each):  {total * 1 / 1e6:.1f} MB  + scales overhead")
print(f"After zlib compression: ~{total * 1 * 0.85 / 1e6:.1f} MB  (target: < 16 MB)")

# ---
# # Part 8: The Muon Optimizer — Newton-Schulz Orthogonalization (line 457)
#
# This is the most mathematically interesting part of the script.
#
# ### The optimization landscape
#
# The model has ~15M parameters. We want to minimize the loss $L(\theta)$ by updating $\theta$ iteratively. Standard SGD: $\theta \leftarrow \theta - \eta \nabla L$. But this is slow because:
# 1. The loss landscape is highly anisotropic (some directions are steep, others flat)
# 2. Different parameter groups have wildly different scales
#
# ### What Muon does differently
#
# Muon (MomentUm Orthogonalization) is a specialized optimizer for **matrix-shaped parameters** (the weight matrices in attention and MLP). The key idea:
#
# Instead of applying the raw gradient $G$ to update a weight matrix $W$, Muon first **orthogonalizes** $G$ using a Newton-Schulz iteration to find the nearest orthogonal matrix:
#
# $$G_{\text{ortho}} = \text{NewtonSchulz}(G + \mu \cdot \text{buf})$$
#
# Then the update is: $W \leftarrow W - \eta \cdot G_{\text{ortho}}$
#
# ### Why orthogonalize gradients?
#
# For a matrix $W$ of shape $(m, n)$, the gradient $G$ lives in the same space. But not all directions are equally useful:
# - Raw gradients often have a few very large singular values and many near-zero ones
# - Orthogonalizing maps $G$ to a matrix with **all singular values = 1**
#
# This is **preconditioning** — you normalize the update so it moves equally in all directions of the weight space.
#
# ### The Newton-Schulz iteration (line 176)
#
# To find the nearest orthogonal matrix, 5 steps of:
#
# $$X_{k+1} = aX_k + b(X_kX_k^T)X_k + c(X_kX_k^T)^2 X_k$$
#
# with coefficients $a = 3.4445, b = -4.7750, c = 2.0315$.
#
# This converges to $G(G^TG)^{-1/2}$ — the polar factor. It's computing $G / \sqrt{G^TG}$ for matrices.
#
# ### The physics analogy
#
# **This is the matrix version of Fisher matrix preconditioning.** When you estimate cosmological parameters, you compute $F^{-1/2}$ to decorrelate the parameter space. Muon does the same thing to the gradient, making the optimization landscape isotropic.
#
# ### Momentum (Nesterov-like)
#
# ```python
# buf = momentum * buf + G          # accumulate momentum
# G_eff = G + momentum * buf        # Nesterov lookahead
# G_ortho = NewtonSchulz(G_eff)     # orthogonalize
# W = W - lr * G_ortho * scale      # update
# ```
#
# Momentum starts at 0.85 → warms up to 0.95 over 500 steps.

# Visualize what Newton-Schulz orthogonalization does to a gradient matrix


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


# Create a random "gradient" with skewed singular values (typical in practice)
np.random.seed(42)
G_np = np.random.randn(8, 8).astype(np.float32)
# Make it anisotropic: one direction has 10x the magnitude
G_np[0, :] *= 10.0
G = mx.array(G_np)

# Orthogonalize
G_ortho = zeropower_newtonschulz5(G, steps=5)

# Compare singular values before and after
U_before, S_before, Vt_before = np.linalg.svd(np.array(G))
U_after, S_after, Vt_after = np.linalg.svd(np.array(G_ortho))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(range(len(S_before)), S_before, color="steelblue", alpha=0.8)
axes[0].set_xlabel("Singular value index")
axes[0].set_ylabel("σ_i")
axes[0].set_title("BEFORE orthogonalization\n(gradient is anisotropic — σ₁ dominates)")
axes[0].grid(True, alpha=0.3)

axes[1].bar(range(len(S_after)), S_after, color="coral", alpha=0.8)
axes[1].set_xlabel("Singular value index")
axes[1].set_ylabel("σ_i")
axes[1].set_title(
    "AFTER orthogonalization\n(all singular values ≈ 1 — isotropic update)"
)
axes[1].set_ylim(0, max(S_before) * 1.1)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("newton_schulz.png", dpi=100, bbox_inches="tight")
plt.show()

print(f"Before: σ_max/σ_min = {S_before[0] / S_before[-1]:.1f}x  (condition number)")
print(f"After:  σ_max/σ_min = {S_after[0] / S_after[-1]:.2f}x   (nearly isotropic)")
print(f"\nThis is like decorrelating your parameter space with F^(-1/2)")

# ---
# # Part 9: The Split Optimizer — Muon + Adam (line 485)
#
# ### Why two optimizers?
#
# Not all parameters are matrices. The model has three kinds:
#
# | Parameter type | Shape | Optimizer | Why |
# |---------------|-------|-----------|-----|
# | **Matrix weights** (Q, K, V, proj, fc, mlp.proj) | 2D | **Muon** | Orthogonalization works on matrices |
# | **Embedding** (tok_emb.weight) | 2D | **Adam** | Tied embedding needs gentler updates; shared between input and output |
# | **Scalars/vectors** (attn_scale, mlp_scale, resid_mix, skip_weights, q_gain) | 1D/2D-small | **Adam** | Too small for Muon; these are "control knobs" not learned representations |
#
# ### Adam: the workhorse
#
# Adam maintains per-parameter running averages of:
# - **First moment** $m = \beta_1 m + (1-\beta_1) g$ (smoothed gradient, like a low-pass filter)
# - **Second moment** $v = \beta_2 v + (1-\beta_2) g^2$ (smoothed squared gradient)
# - **Update**: $\Delta\theta = -\eta \cdot m / (\sqrt{v} + \epsilon)$
#
# The division by $\sqrt{v}$ is **adaptive learning rate** — parameters with large gradients get smaller steps, and vice versa. It's like individually adjusting the step size for each cosmological parameter based on its Fisher information.
#
# ### Learning rate schedule: warmdown
#
# The learning rate follows a simple schedule:
# 1. Full LR for most of training
# 2. **Linear decay to 0** over the last `warmdown_iters` steps (1200 by default)
#
# The warmdown is **wallclock-adaptive**: it estimates how many steps remain based on elapsed time and the 10-minute cap. This means you don't waste steps at full LR when you're about to run out of time.
#
# $$\text{lr\_mul} = \frac{\text{remaining\_time}}{\text{warmdown\_time}} \quad \text{if remaining < warmdown, else 1.0}$$

# Visualize the learning rate warmdown schedule

total_steps = 13000  # typical for 10-min on 8xH100
warmdown_iters = 1200
warmdown_start = total_steps - warmdown_iters

steps = np.arange(total_steps)
lr_mul = np.ones_like(steps, dtype=float)
lr_mul[warmdown_start:] = np.linspace(1.0, 0.0, warmdown_iters)

# Also show the Muon momentum warmup
muon_warmup_steps = 500
momentum_start, momentum_end = 0.85, 0.95
momentum = np.minimum(np.arange(total_steps) / muon_warmup_steps, 1.0)
momentum = (1.0 - momentum) * momentum_start + momentum * momentum_end

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(steps, lr_mul * 0.04, "b-", linewidth=2)
axes[0].axvline(
    warmdown_start,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"warmdown starts (step {warmdown_start})",
)
axes[0].set_xlabel("Training step")
axes[0].set_ylabel("Learning rate (matrix_lr)")
axes[0].set_title("Learning rate schedule\n(flat → linear warmdown to 0)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(steps[:2000], momentum[:2000], "g-", linewidth=2)
axes[1].axvline(
    muon_warmup_steps,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"warmup ends (step {muon_warmup_steps})",
)
axes[1].set_xlabel("Training step")
axes[1].set_ylabel("Muon momentum")
axes[1].set_title("Muon momentum warmup\n(0.85 → 0.95 over 500 steps)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lr_schedule.png", dpi=100, bbox_inches="tight")
plt.show()
print("The warmdown is critical: it lets the model 'settle' into a good minimum")
print("Like cooling in simulated annealing — reduce the temperature at the end")

# ---
# # Part 10: Quantization — Fitting in 16MB (line 543)
#
# ### The compression pipeline
#
# After training, the model lives in fp32 (~60MB). To fit in 16MB:
#
# ```
# fp32 weights (60 MB)
#     → int8 quantization (~15 MB)
#         → zlib compression (~13-15 MB)
#             → ✓ under 16 MB!
# ```
#
# ### How int8 quantization works
#
# For each weight matrix $W$ (shape $m \times n$), quantize **per row**:
#
# 1. Find the clipping threshold: $c_i = \text{quantile}_{99.99984\%}(|W_{i,:}|)$ for each row $i$
# 2. Clip to $[-c_i, c_i]$
# 3. Compute scale: $s_i = c_i / 127$
# 4. Quantize: $W^q_{i,j} = \text{round}(W_{i,j} / s_i)$, clamped to $[-127, 127]$
#
# To dequantize: $\hat{W}_{i,j} = W^q_{i,j} \times s_i$
#
# ### Per-row vs per-tensor
#
# **Per-row** scaling is crucial. Different rows (output neurons) can have very different weight magnitudes. Per-row scaling gives each row its own dynamic range, reducing quantization error.
#
# This is analogous to **per-channel calibration** in CMB detector arrays — each bolometer has its own gain, and calibrating them independently gives better maps than a single global calibration.
#
# ### The 99.99984% clipping percentile
#
# Why not clip at the absolute max? Because outlier weights would set a very large scale, wasting most of the int8 range on values near zero. By clipping at 99.99984%, we sacrifice a tiny fraction of extreme values for much better resolution everywhere else. This is the same tradeoff as **sigma clipping** in astronomical image stacking.
#
# ### Small tensors get special treatment
#
# - **Control tensors** (attn_scale, mlp_scale, etc.): kept in fp32 — they're tiny and precision-sensitive
# - **Other small tensors** (< 65,536 elements): stored in fp16 — cheap enough to keep at higher precision
# - **Large matrices**: int8 with per-row scaling
#
# ### The budget accounting
#
# ```
# 16,000,000 bytes total budget
#  - code bytes (train_gpt.py)     ~25,000 bytes
#  - compressed model              ~15,800,000 bytes
#    = int8 quantized weights      ~15,000,000 bytes (raw)
#    + fp16 scales                 ~   100,000 bytes
#    + fp32 control tensors        ~    50,000 bytes
#    → zlib compressed             ~15,500,000 bytes
# ```
#
# ### Challenge lever: this is where the leaderboard wins come from!
#
# Top submissions use **int6** (6 bits) or even **int5** (5 bits) instead of int8, with QAT (quantization-aware training) to maintain quality. This frees 25-37% more bytes → bigger model → lower bpb.

# Demonstrate quantization error at different bit widths

np.random.seed(42)
weights = np.random.randn(256, 512).astype(np.float32) * 0.02  # typical weight scale


def quantize_at_bits(w, bits):
    """Quantize per-row at a given bit width."""
    max_val = 2 ** (bits - 1) - 1  # e.g., int8 → 127, int6 → 31, int5 → 15
    row_max = np.quantile(np.abs(w), 0.9999984, axis=1)
    scale = np.maximum(row_max / max_val, 1e-10)
    q = np.clip(np.round(w / scale[:, None]), -max_val, max_val)
    w_hat = q * scale[:, None]
    return w_hat, q


bit_widths = [8, 6, 5, 4, 3]
errors = []
compressed_sizes = []

import zlib

for bits in bit_widths:
    w_hat, q = quantize_at_bits(weights, bits)
    # Quantization error (like noise in your measurement)
    mse = np.mean((weights - w_hat) ** 2)
    rel_error = np.sqrt(mse) / np.std(weights)
    errors.append(rel_error)

    # Compressed size (how many bytes in the 16MB budget)
    raw_bytes = q.astype(np.int8).tobytes()  # int8 storage even for lower bits
    compressed = len(zlib.compress(raw_bytes, 9))
    compressed_sizes.append(compressed)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(bit_widths, [e * 100 for e in errors], "ro-", linewidth=2, markersize=8)
axes[0].set_xlabel("Quantization bits")
axes[0].set_ylabel("Relative error (%)")
axes[0].set_title("Quantization error vs bit width\n(like S/N degradation)")
axes[0].invert_xaxis()
axes[0].grid(True, alpha=0.3)

sizes_mb = [s / 1e6 for s in compressed_sizes]
axes[1].bar(
    range(len(bit_widths)),
    sizes_mb,
    color=["green" if s < 16 else "red" for s in sizes_mb],
)
axes[1].set_xticks(range(len(bit_widths)))
axes[1].set_xticklabels([f"int{b}" for b in bit_widths])
axes[1].set_ylabel("Compressed size (MB) — for ONE matrix")
axes[1].set_title("Size savings at lower bit widths\n(bytes freed → bigger model)")
axes[1].axhline(
    y=sizes_mb[0],
    color="gray",
    linestyle="--",
    alpha=0.5,
    label=f"int8 baseline: {sizes_mb[0]:.3f} MB",
)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("quantization.png", dpi=100, bbox_inches="tight")
plt.show()

print(f"{'Bits':>4} | {'Rel Error':>10} | {'Compressed':>12} | {'vs int8':>8}")
print(f"{'─' * 4:>4}-+-{'─' * 10:>10}-+-{'─' * 12:>12}-+-{'─' * 8:>8}")
for b, e, s in zip(bit_widths, errors, compressed_sizes):
    print(f"{b:>4} | {e:>9.4%} | {s / 1e6:>10.3f} MB | {s / compressed_sizes[0]:>7.1%}")
print(f"\nint6 saves ~25% → room for ~4M more parameters!")
print(f"int5 saves ~37% → room for ~6M more parameters!")

# ---
# # Part 11: Evaluation — val_loss and val_bpb (line 761)
#
# ### Two metrics
#
# 1. **val_loss**: Cross-entropy in nats (natural log units). This is what the model directly optimizes.
#
# $$\text{val\_loss} = -\frac{1}{T}\sum_{t=1}^{T} \ln P_\theta(x_t | x_{<t})$$
#
# 2. **val_bpb**: Bits per byte — the challenge metric. Converts the token-level loss to a byte-level compression rate.
#
# $$\text{val\_bpb} = \text{val\_loss} \times \frac{\text{total\_tokens}}{\text{total\_bytes}} \times \log_2(e)$$
#
# ### Why bpb instead of val_loss?
#
# val_loss depends on the **tokenizer**: a 1024-token vocabulary produces different losses than a 50,000-token vocabulary, even for the same model quality. bpb normalizes by raw bytes, making it **tokenizer-agnostic**.
#
# The challenge's scoring is bpb precisely because they want to allow creative tokenizer choices without gaming the metric.
#
# ### The byte counting (subtle!)
#
# Not every token maps to the same number of bytes. The script builds lookup tables:
# - `base_bytes_lut[token_id]` → how many UTF-8 bytes this token represents
# - `has_leading_space_lut[token_id]` → whether the SentencePiece "▁" prefix represents a space byte
# - `is_boundary_token_lut[token_id]` → whether this is a special control/unknown token
#
# The leading space "▁" is tricky: SentencePiece uses it to mark word boundaries. It counts as 1 byte ONLY if the previous token isn't a boundary token (i.e., the space is real, not a sentence-initial artifact).
#
# ### The physics interpretation
#
# bpb is an **information-theoretic compression rate**. Shannon's source coding theorem says the best possible compression rate equals the entropy of the source. So:
#
# $$\text{bpb}_{\text{model}} \geq H_{\text{true}}(\text{English text})$$
#
# The true entropy of English is estimated at ~1.0-1.2 bits/byte. The baseline achieves 1.22, the #1 entry achieves 1.14. Getting closer to the true entropy means your model is a better statistical model of natural language.

# ---
# # Part 12: The Training Loop (line 836)
#
# ### The full pipeline in pseudocode
#
# ```python
# # 1. Setup
# model = GPT(...)                    # initialize model
# optimizer = SplitOptimizers(model)  # Muon for matrices, Adam for rest
#
# # 2. Warmup (20 steps) — prime MLX compilation, then reset data loader
# for step in range(warmup_steps):
#     loss, grads = forward_backward(batch)
#     mx.eval(loss, grads)            # force compilation, DON'T update weights
#
# # 3. Main training loop
# for step in range(iterations):
#     # Check wallclock: stop if 10 minutes elapsed
#     if elapsed > max_wallclock_seconds: break
#
#     # Compute learning rate (1.0 during training, linear decay during warmdown)
#     lr_mul = compute_lr_multiplier(step, elapsed)
#
#     # Gradient accumulation: simulate large batch with small microbatches
#     total_grads = 0
#     for _ in range(grad_accum_steps):          # 8 accumulation steps
#         for chunk in sub_batches:               # further chunking for memory
#             loss, grads = forward_backward(chunk)
#             total_grads += grads * (chunk_size / total_size)
#
#     # Clip gradients (if enabled)
#     total_grads = clip_norm(total_grads)
#
#     # Update weights
#     optimizer.step(model, total_grads, step, lr_mul)
#
# # 4. Final evaluation
# val_loss, val_bpb = evaluate(model, val_data)
#
# # 5. Quantize and save
# quantized = int8_quantize(model)
# compressed = zlib.compress(pickle.dumps(quantized))
# # Verify: reload quantized model and re-evaluate
# ```
#
# ### Gradient accumulation: why?
#
# The effective batch size is 524,288 tokens per step. On a Mac, we can't fit this in memory at once. So we split it:
# - 8 `grad_accum_steps` (microbatches of 65,536 tokens)
# - Each microbatch further chunked into sub-batches of 8,192 tokens (`mlx_max_microbatch_tokens`)
#
# Gradients are accumulated (summed with proper scaling) across all chunks. Mathematically identical to processing the full batch at once.
#
# This is like **stacking exposures** in astronomy: each short exposure is noisy, but averaging 8 of them gives the same S/N as one long exposure (minus read noise, which here is negligible).
#
# ### The wallclock cap
#
# The training loop monitors elapsed time. When `max_wallclock_seconds` (600s = 10 min) is approached:
# 1. The warmdown kicks in (linear LR decay)
# 2. When time runs out, the loop sets `stop_after_step` and exits cleanly
#
# This is adaptive — a faster machine gets more steps in 10 minutes.
#
# ### The warmup phase
#
# The first 20 steps are "warmup" — they run real forward/backward passes but **don't update weights**. This primes MLX's JIT compiler (traces the computation graph, allocates memory). After warmup, the data loader resets so training starts from the true beginning.
#
# This is like a **burn-in period** in MCMC — you run the sampler for a while to reach equilibrium before collecting samples.

# ---
# # Part 13: Challenge Levers — What to Change to Win
#
# Now that you understand every component, here's a map of **what to modify** and **what it affects**:
#
# ### Architecture levers
#
# | Lever | Current | Change | Effect | Risk |
# |-------|---------|--------|--------|------|
# | `num_layers` | 9 | 10-11 | More depth = more abstraction | Need better compression to fit |
# | `mlp_mult` | 2 | 3 | Wider MLP = more capacity | Same — need more compression |
# | `num_kv_heads` | 4 | 2 | Fewer KV params | Slight quality loss |
# | Add SmearGate | none | bigram context | ~0.008 bpb | 512 extra params |
# | Add BigramHash | none | hash embedding | ~0.004 bpb | Hash table memory |
# | Weight tying across layers | none | share weights | More virtual depth for free | May hurt quality |
#
# ### Compression levers
#
# | Lever | Current | Change | Effect |
# |-------|---------|--------|--------|
# | Quantization | int8 post-hoc | int6 QAT | 25% smaller, ~same quality |
# | Compression | zlib-9 | zstd-22 | ~10% better ratio |
# | SWA | none | average checkpoints | Smoother weights → better quantization |
# | Pruning | none | 3% magnitude | Small size savings |
#
# ### Evaluation levers
#
# | Lever | Current | Change | Effect |
# |-------|---------|--------|--------|
# | Eval method | non-overlapping chunks | Sliding window stride=64 | **-0.032 bpb** (biggest single win!) |
# | TTT | none | LoRA per-document | -0.003 bpb (unoptimized) |
#
# ### Optimizer levers
#
# | Lever | Current | Change | Effect |
# |-------|---------|--------|--------|
# | Weight decay | 0 | 0.01-0.04 | Helps quantization (smaller weights) |
# | Init | default | orthogonal | ~0.003 bpb |
#
# ### Priority order for your first experiments
#
# 1. **Sliding window eval** (eval-only, no retraining needed, biggest win)
# 2. **Int6 QAT + zstd** (compression, enables bigger model)
# 3. **3x MLP + more layers** (architecture, uses freed bytes)
# 4. **SmearGate + BigramHash** (architecture modules)
# 5. **SWA + weight decay** (training refinements)

# Summary: the complete data flow through the model

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMPLETE DATA FLOW: train_gpt_mlx.py                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INPUT: token_ids  [batch, seq_len]  (integers 0..1023)                     ║
║    │                                                                         ║
║    ▼                                                                         ║
║  EMBEDDING: tok_emb(token_ids)  →  [batch, seq, 512]  (bf16)               ║
║    │                                                                         ║
║    ▼                                                                         ║
║  RMS_NORM  →  x0 (saved for resid_mix in every block)                       ║
║    │                                                                         ║
║    ├─── ENCODER (blocks 0-3) ──────────────────────────────────────────────  ║
║    │    For each block:                                                      ║
║    │      x = resid_mix[0]*x + resid_mix[1]*x0    (blend with original)     ║
║    │      x = x + attn_scale * Attention(RMSNorm(x))   (+ residual)        ║
║    │      x = x + mlp_scale  * MLP(RMSNorm(x))         (+ residual)        ║
║    │      save skip = x                                                      ║
║    │                                                                         ║
║    ├─── DECODER (blocks 4-8) ──────────────────────────────────────────────  ║
║    │    For each block:                                                      ║
║    │      x = x + skip_weight * skip.pop()    (add reversed encoder skip)   ║
║    │      x = resid_mix[0]*x + resid_mix[1]*x0                              ║
║    │      x = x + attn_scale * Attention(RMSNorm(x))                        ║
║    │      x = x + mlp_scale  * MLP(RMSNorm(x))                              ║
║    │                                                                         ║
║    ▼                                                                         ║
║  FINAL_NORM(x)  →  [batch, seq, 512]                                        ║
║    │                                                                         ║
║    ▼                                                                         ║
║  LOGITS: x @ tok_emb.weight.T  →  [batch, seq, 1024]  (tied embedding!)    ║
║    │                                                                         ║
║    ▼                                                                         ║
║  SOFTCAP: 30 * tanh(logits / 30)  →  bounded to [-30, 30]                   ║
║    │                                                                         ║
║    ▼                                                                         ║
║  CROSS_ENTROPY(logits, target_ids)  →  scalar loss (nats)                   ║
║    │                                                                         ║
║    ▼                                                                         ║
║  BACKPROP  →  gradients for all parameters                                   ║
║    │                                                                         ║
║    ├── Matrix weights  →  MUON (orthogonalize + momentum + update)          ║
║    ├── Embedding       →  ADAM (adaptive learning rate)                      ║
║    └── Scalars/vectors →  ADAM (separate LR)                                 ║
║                                                                              ║
║  After training:                                                             ║
║    weights → int8 quantize → zlib compress → save (must be < 16 MB)         ║
║    reload → dequantize → re-evaluate → final val_bpb (the score!)           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# ---
# # Exercises
#
# Now that you understand the full pipeline, try these (each builds on the previous):
#
# ### Exercise 1: Run the baseline
# ```bash
# cd parameter-golf
# RUN_ID=baseline_test ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
# ```
# Record the final `val_bpb`. This is your reference point.
#
# ### Exercise 2: Change one hyperparameter
# Try `NUM_LAYERS=12` or `MLP_MULT=3`. Does bpb improve? Does the model still fit in 16MB?
#
# ### Exercise 3: Implement sliding window eval
# Modify `eval_val()` to use overlapping windows with stride=64 instead of non-overlapping chunks. This should give ~0.03 bpb improvement for free.
#
# ### Exercise 4: Add weight decay to Muon
# In the `Muon.step()` method, add `p = p * (1 - wd)` before the gradient update. Try `wd=0.01` and `wd=0.04`.
#
# ### Exercise 5: Implement int6 QAT
# In the `GPT.loss()` method, add fake quantization before the forward pass:
# ```python
# # Fake-quantize: simulate int6 during training
# w_q = (w / scale).round().clamp(-31, 31) * scale
# # Use w_q for forward, but w for gradients (straight-through estimator)
# ```
#
# ### Exercise 6: Your own idea
# What would you try that nobody else has? Depth recurrence? MoE? A better activation function? The challenge is open until April 30th.
#
# ---
#
# **Good luck! The leaderboard awaits.**
