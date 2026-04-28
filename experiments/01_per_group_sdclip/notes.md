# Experiment 01 — Per-Group SDClip Allocator

## What changed
Replaces the SOTA's two-bucket SDClip (matrix `k=12.85` / embed `k=20.0`) with a five-group dispatch: `embed`, `early` (layers `< loop_start`), `loop` (the recurrent layers), `mid`, and `late` (last 3 layers). Each group has its own `k` (and optional bit-width) via env vars `GROUP_CLIP_SIGMAS_*` and `GROUP_BITS_*`.

## Why
Robby Sneiderman's PR #1412 found that per-group Hessian-trace ratios are highly stable across seeds (r=0.997) — early blocks have ~30x the trace of late blocks — but per-row importance is noise (r=0.12). Robby modulated SDClip per-row at lambda=0.175 and saw a small win. Per-group should be a strictly more reliable signal since the noise floor is much lower.

## Defaults
`early=10.5, loop=12.0, mid=12.85, late=14.5, embed=20.0` — biases bits-per-importance toward early blocks, which carry the most loss-curvature mass.

## Expected gain
0.002–0.005 BPB. Robby's per-row variant got ~0.001 BPB at lambda=0.175; per-group should match or exceed that with lower variance.

## Ablation
- `PER_GROUP_QUANT_ENABLED=0` (default) -> byte-identical to SOTA.
- `PER_GROUP_QUANT_ENABLED=1` with all `GROUP_CLIP_SIGMAS_*=12.85, EMBED=20.0` -> also byte-identical (sanity check).

## Touched
- `Hyperparameters` (11 new fields)
- New helper `_assign_quant_group`
- `gptq_mixed_quantize` (dispatch + per-group raw-byte logging)

## Sweep candidates after baseline
1. Pull more from `late`: `late=16.0, 18.0` (likely safe, late layers have lowest trace).
2. Push `early` further: `early=9.5, 9.0` (paid for by `late=16.0`).
3. int7 for `early` only: `GROUP_BITS_EARLY=7, GROUP_CLIP_SIGMAS_EARLY=21` (same compressed size by entropy bound, more precision where it matters).

## Run
```bash
source experiments/01_per_group_sdclip/config.env
RUN_ID=01_pgsdclip_seed42 SEED=42 \
  torchrun --standalone --nproc_per_node=3 experiments/01_per_group_sdclip/train_gpt.py
```
