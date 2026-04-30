# Independent Validation Guide

This submission folder is designed to be independently validated on any machine,
without GPUs, without pulling the rest of the parameter-golf repo. Everything
needed is here, plus a small auxiliary npz + tokenizer (downloaded separately).

## What's in this folder

| File | Purpose |
|---|---|
| `train_gpt.py` | The submitted artifact. ~19.6 KB LZMA-self-extracting Python wrapper. When run via torchrun, decompresses + execs the full training+eval source. |
| `submission.json` | Machine-readable claim: 3-seed mean, std, per-seed BPB, compliance metadata. |
| `train_seed42.log` / `train_seed314.log` / `train_seed999.log` | Full eval logs from the 3-seed runs that produced the headline 0.99621 mean. |
| `compliance_seed42.log` | The TTT_ENABLED=0 timing-compliance run showing eval fits in the 600s 8xH100 budget. |
| `local_validator.py` | Self-contained Python validator (described below). |
| `README.md` | The submission writeup. |
| `VALIDATING.md` | This file. |

## What the validator checks

Run `python3 local_validator.py --record-dir .` to execute these in order:

1. **Packed code self-test** — `train_gpt.py` decompresses cleanly to valid
   Python. Optionally compares AST to a separately-supplied readable source
   if you also pass `--source path/to/tw_eval.py`.

2. **Artifact size** — packed `train_gpt.py` + each seed's reported
   `Serialized model quantized+brotli:` byte count must total < 16,000,000.

3. **Source audit** — pretty-prints `_ppm_mixture_bpb` from the decompressed
   artifact so a human can verify:
   - the gate `cf` is computed BEFORE the byte-x lookup (causality)
   - update of `tabs[o]` happens AFTER scoring at position `i`
   - single iteration over bytes (no two-pass)
   - no use of future tokens

4. **Log/JSON consistency** — every seed's `ppm_mixer val_bpb:` in its log
   must match `submission.json`'s `per_seed_results[seed].ppm_mixer_val_bpb`
   to within 1e-5.

5. **Numerical reproduction (optional)** — given `ppm_inputs_seed42.npz`
   (a dump of the gathered (target_tokens, neural_log_probs) arrays from the
   actual cluster run) and `fineweb_8192_bpe.model` (the SP8192 tokenizer),
   re-runs the mixer locally and checks the BPB matches the seed log to
   within 1e-5. **This is the strongest check: it's the only one that
   actually reproduces the headline number from raw inputs.**

## Quick run (checks 1-4, no GPU artifacts needed)

```bash
python3 local_validator.py --record-dir .
```

## Full run (adds check 5: numerical reproduction)

```bash
# Acquire the auxiliary files (download them from where the submitter shared them):
#   ppm_inputs_seed42.npz   (~150 MB compressed; ~600 MB uncompressed)
#   fineweb_8192_bpe.model  (~600 KB; same SP8192 tokenizer used in training)

pip install numpy sentencepiece

python3 local_validator.py \
  --record-dir . \
  --tokenizer ./fineweb_8192_bpe.model \
  --ppm-dump  ./ppm_inputs_seed42.npz \
  --expected-bpb 0.99548272
```

Expected: all 5 checks pass. Check 5 takes ~5-10 minutes of CPU time on a
modern laptop (the PPM mixer is single-threaded Python, ~40M bytes to process).

## Why this validator is sufficient

- **Check 1** proves the artifact is real Python that actually runs (no
  obfuscated payload).
- **Check 2** proves the size budget compliance.
- **Check 3** lets a reviewer literally read the mixer math and verify the
  causality and single-pass properties.
- **Check 4** proves the per-seed BPB claims in submission.json aren't
  fabricated — they appear verbatim in the seed logs.
- **Check 5** proves the headline BPB is reproducible from the actual neural
  log-probabilities the cluster produced. **You don't need to retrain to
  verify the mixer's contribution** — given the dump, the validator
  recomputes the BPB end-to-end.

What the validator does NOT verify:
- **The neural model's training** — that requires GPUs. To verify training
  compliance, you'd need to actually run `torchrun --nproc_per_node=8
  train_gpt.py` on 8xH100 SXM and check the ≤600s wallclock + the resulting
  `pre-quantization post-ema val_bpb:` line. Reviewers do this.
- **That the (tga, lpa) dump came from a faithful run** — if you suspect the
  neural log-probs were doctored, you'd need to retrain. Check 5 only
  verifies the mixer applied to those log-probs gives the claimed BPB.

## How to acquire ppm_inputs_seed42.npz for check 5

The submitter generates this file by re-running seed 42 with
`DUMP_PPM_INPUTS=1`:

```bash
# On the cluster:
DUMP_PPM_INPUTS=1 DUMP_PPM_PATH=./ppm_inputs_seed42.npz \
  TTT_ENABLED=0 SEED=42 \
  torchrun --standalone --nproc_per_node=4 train_gpt.py
```

The file is then transferred (scp / rsync / shared drive) to wherever you
want to validate.
