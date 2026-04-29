# Experiment 05 — Mamba Hybrid (Educational + Non-Record)

## What it is
Replaces select transformer self-attention layers with **Mamba selective state-space
model (SSM)** blocks. The rest of the bigbag stack (RMSNorm, MLP 4x, residuals,
layer-loop, parallel residuals on non-Mamba layers, MuonEq-R, GPTQ, TTT) is
unchanged.

## What you're learning here

### Self-attention vs Mamba in one paragraph
Self-attention computes a similarity matrix `softmax(QK^T / sqrt(d)) @ V`. Cost:
**O(N^2) compute, O(N^2) memory** in seq length N. Every token can directly
attend to any prior token, but you pay for it.

Mamba is a **selective SSM**: each token has a fixed-size hidden state
`h_t in R^d_state` (we use d_state=16). The state update is:

```
h_t = A_t * h_{t-1} + B_t * x_t        (linear recurrence)
y_t =       C_t * h_t                   (state read-out)
```

**The "selective" part**: A, B, C are *functions of x_t* (input-dependent, hence
"selective" — the network decides at each step what to remember, what to ignore,
what to emit). This was Mamba's key innovation over earlier SSMs (S4, S5) which
had time-invariant A, B, C.

Cost: **O(N) compute, O(1) memory** in seq length. The recurrence runs as a
**parallel scan** on GPU — the trick that makes it fast enough to be competitive.

### Tradeoffs at small param scale
- Mamba block (d_model=512, expand=2, d_state=16) ~= 1.6M params
- Attention block (d_model=512, 8 heads, 4 KV) ~= 0.6M params
- Mamba is **~2.5x the params** of an attention block; but the **MLP after** is
  the dominant param cost in either case.
- For long sequences (N >> d_state), Mamba is dramatically cheaper at inference.
- For short sequences (N=2048 here), the asymptotic win is marginal.

### Why hybrid, not full Mamba?
- Pure-Mamba models tend to slightly underperform pure-attention at the same
  param count on most language benchmarks at small scale (Mamba-2 paper).
- Hybrid (Mamba + attention) often beats either alone — attention is good at
  precise long-range copy/retrieval, Mamba is good at smooth context aggregation.
- Stripped Hyena, Jamba, and Zamba all use this hybrid pattern.
- Practical: at parameter-golf's 16MB cap and 600s training, fewer ablation runs
  are wasted by being conservative. Replace 1-2 layers, see what happens.

## Implementation
- New `MambaInner` class wraps `mamba_ssm.Mamba` — drop-in replacement for
  `CausalSelfAttention` with the same `forward(x) -> (B, S, D)` signature.
- New env var `MAMBA_LAYER_INDICES` (comma-separated, default `""` = no Mamba).
- After all attn-specific setup (rope, xsa, parallel residuals), swap the
  selected indices' `block.attn` to `MambaInner`. Mamba blocks get
  `block.parallel = False` since parallel residual logic is attention-specific.
- MLP, RMSNorm, residual scales, layer-loop, EMA, all unchanged.

## Caveats
- **Quantization**: GPTQ scans `CastedLinear` layers. Mamba's internal `Linear`
  layers are not `CastedLinear`, so they pass through as fp16. This inflates
  the artifact size (~+1 MB per Mamba layer). For the non-record track that's
  fine; for an actual submission you'd extend `gptq_quantize_weight` to handle
  Mamba's projection matrices.
- **torch.compile**: mamba-ssm's CUDA kernels may break `fullgraph=True`. If
  the run errors at compile time, drop to `dynamic=True, fullgraph=False` for
  the model compile. We'll iterate at runtime.
- **No depth recurrence on Mamba layers**: putting Mamba inside `loop_start..loop_end`
  means the same Mamba block runs multiple times — fine algorithmically, but
  whether it helps or hurts is empirical.

## Suggested starting config
- `MAMBA_LAYER_INDICES=6` — single Mamba block at layer 6, between the loop
  block (3-5) and the parallel-residual block (7-10). Conservative.
- Then try `MAMBA_LAYER_INDICES=2,8` — one early, one late, neither in the loop
  or parallel band.
- For pure learning: `MAMBA_LAYER_INDICES=0,1,2,3,4,5,6,7,8,9,10` to make a
  pure-Mamba run, see how much it underperforms.

## Run
```bash
source experiments/05_mamba_hybrid/config.env
RUN_ID=05_mamba_seed42 SEED=42 \
  torchrun --standalone --nproc_per_node=3 experiments/05_mamba_hybrid/tw_eval.py
```

## Further reading
- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
  — the original Mamba paper. The selective-scan derivation (eq. 3-4) is the core idea.
- Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms
  Through Structured State Space Duality" (2024) — the "Mamba-2" paper. Shows
  attention is a special case of structured SSM, unifies the two views.
- Lieber et al., "Jamba" (2024) — the canonical hybrid Mamba+Attention paper.
- modded-nanogpt has experimented with Mamba-2 as a drop-in attn replacement.
