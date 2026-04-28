# 03_qat_sdclip — Quantization-Aware Training on SDClip

## STE construction
`_SDClipFakeQuantizeSTE` is a custom `torch.autograd.Function`:

- **Forward**: `s = (k_sigmas * W.std(dim=1, keepdim=True)) / clip_range`,
  `q = clamp(round(W / s), -clip_range, clip_range)`, return `q * s`.
  This mirrors the per-row scale that `gptq_quantize_weight` derives from
  `W_orig.std(dim=1)` at calibration time.
- **Backward**: identity gradient on `W`; `None` for the python-scalar
  `k_sigmas` and `clip_range`. Pure straight-through estimator — round/clip
  are non-differentiable, so we pretend they had unit Jacobians.

## Per-row sigma is recomputed each step
Snapshotting the calibration-time `std` would let the model drift away from
its quantization grid as Muon updates the rows. Recomputing keeps the fake
quant grid co-moving with the actual `std(W)` GPTQ will see at the end of
training, so the gradient signal teaches the model to operate within the grid
density it will actually be evaluated against.

## Alpha-blend choice
`W_used = (1 - alpha) * W + alpha * fake_quant(W)`. `alpha` lives in a 0-d
buffer that's *shared by reference* across every `CastedLinear`. The graph is
always-on; only the buffer value changes per step. This avoids `torch.compile`
recompilation that toggling a python flag mid-training would cause, and DDP
sees a stable forward graph. `alpha` ramps linearly from 0 over
`[QAT_START_FRAC, QAT_START_FRAC + QAT_RAMP_FRAC]` (default 0.85 → 0.90).
EMA still tracks the underlying un-quantized `W` (Muon writes to the param,
QAT never mutates it), so GPTQ at the end gets the right calibration target.

## Expected gain & risks
~0.005 BPB if QAT closes ~half of the ~0.012 pre/post-quant gap. Risks:
torch.compile may recompile when `_qat_apply` flips per-module on first run
(stable thereafter); too sharp an alpha ramp can spike loss and trigger the
0.3 grad-clip; per-row `std` over fp32 weights adds a small per-step cost
on every matrix linear (~3-5% step time).
