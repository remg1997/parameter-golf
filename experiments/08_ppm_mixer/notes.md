# Experiment 08 — Byte-PPM Mixer (Path 1)

## What it is
Adopts the byte-level PPM-D order-4 mixer from PR #1795 (OE-GOD), applied at
eval time only, with a strict-legal causal "outcome-independent adaptive-lambda
gate." Identical Python source as PR #1795's final corrected version.

## Why it should win
PR #1933 stacked this same mixer on yahya010's neural base and reported
**val_bpb = 0.99145** (3-seed mean), beating the merged SOTA (1.0611) by
**0.0697 BPB**. The mixer alone contributes ~ -0.075 BPB regardless of
neural base.

Applied to **our 1.07987 baseline**, expected post-mixer BPB ~ **1.005-1.010**
— enough for top-5.

## How it works (the mixer in one paragraph)
At eval time, we already compute per-token NLL from the neural model. We
spread each token's log-probability uniformly across its bytes (token's
log_p / piece_len). In parallel, an **online PPM-D byte model** processes the
same byte stream causally: at each byte, score it under the current PPM
context (max order 4) using PPM-D's escape mechanism, then update the
counters. The mixer combines neural and PPM byte log-probs with an adaptive
weight `lambda`:

```
lambda = L_=0.05 if cf > T=0.9 else H=0.9
p_mix  = lambda * exp(neural_log_p_byte) + (1-lambda) * exp(ppm_log_p_byte)
bpb    = -mean(log2(max(p_mix, 1e-300)))
```

The gate `cf` is `max_count / total` at the **deepest context with data**.
Critical compliance bit: `cf` is computed from PPM tables BEFORE looking up
the observed byte's count, so the gate decision depends only on the prefix
(causal). When the deepest context has seen lots of repeats (`cf` high), the
neural model is trusted (lambda small, all-neural). When the prefix is
non-distinctive (`cf` low), PPM dominates (lambda ~ 0.9 toward PPM).

## Compliance review
- **Causal**: `cf` and `pf` are computed from PPM tables built only from
  bytes seen before position i. Each byte is **scored before update**.
- **Single pass**: each byte is scored exactly once.
- **Normalized softmax over vocab**: yes — the neural model is unchanged.
  The PPM mixer is a separate byte-level distribution combined post-hoc.
- **Non-conditional gate**: `cf` is computed from `(max_count, total)` at the
  deepest context that has any data, BEFORE consulting `d.get(x)` at any
  shallower context. This was the explicit fix in PR #1795 review (an earlier
  illegal "outcome-conditioned" version was retracted).
- **No SLOT, no n-gram-as-cache, no multi-pass**.
- **Same artifact as before**: the mixer is pure Python at eval time, ships
  ~60 lines of code (~2 KB minified). Negligible artifact-size impact.

## Implementation details
- New `_ppm_mixture_bpb(tgt_np, lp_np, sp, O=4, H=0.9, L_=0.05, T=0.9)` function
  inserted before `eval_val_sliding`.
- `eval_val_sliding` now captures per-token `(target_id, log_p)` tuples in
  scoring order. After local accumulation, ranks all-gather sizes, then
  `dist.gather` to rank 0, which concatenates and runs the PPM mixer.
- Order preservation: rank-0 windows score the earliest val tokens, rank-1
  the next chunk, etc. Concatenating in rank order restores the global
  validation token stream order, which is what the PPM model needs.
- Non-distributed (single-rank) path also supported.
- Eval time impact: ~30-60 s extra on rank 0 only (sequential pure-Python loop
  over ~8M bytes). Doesn't block other ranks.

## Hyperparameters (env vars)
- `PPM_MIXER_ENABLED=1` (default on)
- `PPM_ORDER=4` (PPM-D order)
- `PPM_H=0.9` — gate weight when context is non-distinctive (lean on PPM)
- `PPM_L=0.05` — gate weight when context is highly distinctive (lean on neural)
- `PPM_T=0.9` — threshold to switch between H and L

## Expected output
The eval log will print:
```
quantized_sliding_window val_loss:... val_bpb:1.0815 eval_time:...ms
ppm_mixer val_bpb:1.0080 eval_time:30000ms order=4 H=0.9 L=0.05 T=0.9 N_bytes=...
quantized_ttt val_loss:... val_bpb:1.0798 eval_time:...ms
```

The `ppm_mixer val_bpb` is the new headline number.

## Run
```bash
source experiments/08_ppm_mixer/config.env
RUN_ID=08_ppm_seed42 SEED=42 \
  torchrun --standalone --nproc_per_node=3 experiments/08_ppm_mixer/tw_eval.py
```

## Sweep candidates after baseline
1. `PPM_ORDER=5` — deeper PPM context (slower but finer)
2. `PPM_T=0.8` — earlier switch to PPM-dominant mixing
3. `PPM_H=0.95, PPM_L=0.03` — more aggressive separation
4. Combine with `04_doc_boundary_ttt` — the doc-boundary chunking might pair
   well with the PPM model since both reason about local statistics.

## Lineage
- PR #1795 (OE-GOD) — original mixer + the legal-gate fix
- PR #1933 (deborahnelson8788726) — applied the mixer on top of yahya010's
  base, claimed 0.99145 BPB
- Our experiment 08 — same mixer on top of bigbag's stack
