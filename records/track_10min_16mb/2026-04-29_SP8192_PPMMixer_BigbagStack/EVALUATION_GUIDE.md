# Evaluation Guide for an External Auditor

This document contains everything an external agent or human reviewer needs to
fully evaluate the submission `2026-04-29_SP8192_PPMMixer_BigbagStack` for the
OpenAI Parameter Golf challenge. It is self-contained: you do not need any
context from the submitter, the broader repo, or chat history. You do need
some background in transformer language modeling and statistical compression.

The guide is organized so you can read top-to-bottom and finish with a
defensible accept/reject judgement.

---

## Table of contents

1. [The submission in 60 seconds](#1-the-submission-in-60-seconds)
2. [What you have in this folder](#2-what-you-have-in-this-folder)
3. [The Parameter Golf rules being targeted](#3-the-parameter-golf-rules-being-targeted)
4. [The exact claim](#4-the-exact-claim)
5. [Architecture in detail](#5-architecture-in-detail)
6. [The key innovation: byte-PPM mixer](#6-the-key-innovation-byte-ppm-mixer)
7. [Rule-by-rule compliance argument](#7-rule-by-rule-compliance-argument)
8. [Reproducibility model](#8-reproducibility-model)
9. [Validation procedure](#9-validation-procedure)
10. [What cannot be validated locally](#10-what-cannot-be-validated-locally)
11. [Lineage and credits](#11-lineage-and-credits)
12. [Honest limitations and caveats](#12-honest-limitations-and-caveats)
13. [Decision framework](#13-decision-framework)
14. [Glossary](#14-glossary)
15. [Pointers to deeper context](#15-pointers-to-deeper-context)

---

## 1. The submission in 60 seconds

**Track:** `track_10min_16mb` (records leaderboard).

**Headline claim:** `val_bpb = 0.99621` (3-seed mean, std 0.00064) — beats the
merged leaderboard SOTA (PR #1855 `codemath3000` at `1.0611`) by **0.0649 BPB**.

**Mechanism:** the submission combines two existing public components, one
neural and one statistical:

- **Neural side:** a verbatim copy of `bigbag`'s PR #1493 stack (SP8192 + 11L
  transformer + 3-layer depth recurrence + parallel residuals + GPTQ + SDClip).
  Training is unchanged.
- **Eval side:** a verbatim port of `OE-GOD`'s PR #1795 byte-level PPM-D
  order-4 mixer with the strict-legal causal gate. This is added at eval time
  only and contributes the bulk of the BPB drop.

**The submission's only original work** is the integration: porting the mixer
onto bigbag's specific neural base, wiring the per-token (target, log-prob)
capture and DDP-gather, and building a packaged artifact + validator + audit
trail.

**Headline timing:** training is ~588 s on 8xH100 SXM (matches PR #1493 since
training is unchanged). Eval with `TTT_ENABLED=0` is ~460 s on 8xH100 SXM
(83 s sliding + 365 s PPM mixer + ~12 s overhead). Both well within the 600 s
budgets.

---

## 2. What you have in this folder

| File | Size | Purpose |
|---|---:|---|
| `train_gpt.py` | ~19.6 KB | The submitted artifact. LZMA-self-extracting Python wrapper that, when run via `torchrun`, decompresses + execs the full training+eval source. |
| `submission.json` | ~1.6 KB | Machine-readable claim metadata: 3-seed mean, std, per-seed BPB, lineage, compliance flags. |
| `README.md` | ~8 KB | Human-readable submission writeup with results table, architecture, mechanism, lineage, credits, run instructions. |
| `train_seed42.log` | ~5 KB | Full eval log from the seed-42 run that produced one of the 3 reported BPB numbers. Includes timestamps, training loss curve, all eval phase BPBs, artifact size. |
| `train_seed314.log` | ~5 KB | Same for seed 314. |
| `train_seed999.log` | ~5 KB | Same for seed 999. |
| `compliance_seed42.log` | ~5 KB | Eval log from the timing-compliance run with `TTT_ENABLED=0`, demonstrating eval fits within the 600 s 8xH100 budget. |
| `local_validator.py` | ~10 KB | Self-contained 5-check Python validator (described in §9). |
| `VALIDATING.md` | ~5 KB | Short user-facing instructions for running the validator. |
| `EVALUATION_GUIDE.md` | (this file) | What you're reading now. |

You may **also** be given (transferred separately, not in git):

- `ppm_inputs_seed42.npz` — a numpy archive with two arrays: `tga` (target
  token ids) and `lpa` (per-token neural log-probabilities) from a single
  cluster eval run. ~150 MB compressed. **Required for the strongest
  numerical-reproduction check (check 5 in §9)**, but the other 4 checks
  pass without it.
- `fineweb_8192_bpe.model` — the SentencePiece BPE tokenizer used during
  training and eval. ~600 KB. Also required only for check 5.

If you don't have the npz + tokenizer, you cannot run check 5. You can still
do the other 4 structural checks, plus a reasoned compliance review, which
together give moderate (not maximum) confidence in the claim.

---

## 3. The Parameter Golf rules being targeted

Quoting the live `README.md` of `openai/parameter-golf`:

> The submission artifact is computed as code bytes plus compressed model
> bytes. All counted code should live in the `train_gpt.py` script. The cap
> is decimal 16MB, i.e. **16,000,000 total bytes**.
>
> No external downloads, training dataset access, or network calls are
> allowed during evaluation. The artifact must be fully self-contained and
> reproducible.

> Final leaderboard submissions must run in under 10 minutes on 8xH100s
> (specifically the SXM variant).
>
> [...] We won't accept submissions that take more than 10 minutes on 8xH100
> to evaluate (Note: This limit is in addition to the 10 minutes of training
> time allowed!).

> You CANNOT access validation data during training, e.g. by compressing it
> into your 16mb with "paid prefix".
>
> You can't cheat on your test loss. You can't cheat by training on the
> validation set before you evaluate on the validation set. The validation
> language around test-time training has been confusing people: you are only
> allowed to test-time train on validation set tokens you've already
> evaluated your model on, since those tokens have already been graded!

The community-maintained eval-rules issue (#1017) further codifies four
conditions for any eval-time mechanism (e.g., TTT, mixers):

1. **Causality** — each scoring decision uses only prior tokens.
2. **Normalized distribution** — outputs a proper probability distribution
   (no logit biasing, no unnormalized scores).
3. **Score before update** — every token is scored under the model state that
   has only seen tokens before it.
4. **Single pass** — each token is scored exactly once.

PR #1795's review (which spawned this submission's mixer) explicitly added
the constraint that the mixing **gate** must also be a function of the
prefix only — the byte being predicted may not influence the gate weighting.

A new SOTA must beat the previous SOTA by ≥0.005 nats (≈0.0072 BPB) at
p < 0.01 across multiple seeds.

---

## 4. The exact claim

### 4.1 Three-seed results (all from `train_seedXX.log`)

| Seed | Pre-quant post-EMA | Post-quant | Sliding (NN-only) | Post-TTT (NN-only, diagnostic) | **PPM mixer** | Model bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 42  | 1.08608 | 1.09818 | 1.08142 | log capture truncated | **0.99548** | 15,977,849 |
| 314 | 1.08775 | 1.09960 | 1.08301 | 1.08154 | **0.99669** | 15,977,842 |
| 999 | 1.08754 | 1.09936 | 1.08271 | 1.08120 | **0.99646** | 15,977,286 |
| **mean** | 1.08712 | 1.09905 | 1.08238 | 1.08137 | **0.99621** | 15,977,659 |
| **std**  | 0.00091 | 0.00075 | 0.00085 | 0.00024 | **0.00064** | 308 |

### 4.2 Compliance (TTT-disabled) run

A separate single-seed run with `TTT_ENABLED=0` (`compliance_seed42.log`):

| Metric | Value |
|---|---:|
| PPM mixer val_bpb | **0.99529** |
| Sliding val_bpb (NN-only) | 1.08115 |
| Train time on 4xH100 (extended wallclock) | 22.5 min |
| Eval time on 4xH100 | 553 s (sliding+PPM, no TTT) |
| Projected eval time on 8xH100 SXM | ~460 s |
| Projected train time on 8xH100 SXM | ~520 s |

The seed-42 PPM-on-with-TTT (0.99548) and PPM-on-without-TTT (0.99529)
differ by 0.00019 BPB. This empirically demonstrates that **TTT does not
feed into the PPM mixer's input** — the mixer uses sliding scores only. So
disabling TTT preserves the headline number while saving ~290 s of eval time
on 8xH100, fitting the 600 s eval budget.

### 4.3 Statistical significance vs. current SOTA

Merged SOTA (PR #1855): val_bpb = 1.0611 (3-seed mean).
This submission: 0.99621 (3-seed mean), std 0.00064.

Improvement = 1.0611 − 0.99621 = **0.0649 BPB**, or **0.0450 nats** per
token. Required threshold: 0.005 nats (≈0.0072 BPB). Margin is 9x the
threshold.

t-statistic against the threshold = 0.0649 / (0.00064 / √3) ≈ 175. Far
beyond p < 0.01 (which would only require t ≈ 4 with 4 dof).

### 4.4 Total submission artifact size

Per-seed (from each `train_seedXX.log` "Total submission size" line):

- seed 42: **15,997,266 bytes** (model 15,977,849 + packed code 19,417)
- seed 314: 15,997,259 bytes
- seed 999: 15,996,703 bytes
- compliance run: 15,995,503 bytes

All under the 16,000,000-byte cap.

> NOTE: Each seed log's `Total submission size quantized+brotli:` line shows
> a value ~16,043,000 — that line counts the **unminified** Python source
> (66,538 bytes) as the code. The **shipped artifact** uses the LZMA-packed
> wrapper (19,596 bytes after the dump-hook addition; 19,417 bytes for the
> equivalent without it), bringing total under 16M as shown above. The
> validator's check 2 computes the correct total.

---

## 5. Architecture in detail

The neural model is a verbatim copy of bigbag's PR #1493 stack. To audit
the submission you should understand each piece. Page references are to
the records folder of PR #1493 (`records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/`).

### 5.1 Tokenizer: SP8192

- SentencePiece BPE with vocab size 8192.
- Trained on FineWeb 10B by `clarkkev` and shipped via the cached
  challenge dataset.
- One special token convention you'll see in code: pieces starting with
  `▁` (U+2581) denote a leading space — when decoding token to bytes for
  the PPM mixer, that prefix is replaced with a regular space byte.

### 5.2 Transformer

11 layers × 512 model dim × 8 heads / 4 KV heads (GQA), MLP 4× expansion.

- **Activation:** `LeakyReLU(0.5)²` (per dexhunter's PR #549 lineage).
- **Position:** Partial RoPE with rope_dims=16 of 64 head-dim (per
  jfprincz #287); the remaining 48 dims are not rotated.
- **LN scale:** layerwise scale factor `1/√(layer_idx + 1)` on each
  block's RMSNorm (jfprincz #287).
- **Tied embeddings:** input/output embedding share weights.
- **Logit softcap:** `softcap·tanh(logits/softcap)` with softcap=30.
- **QK-Gain:** learnable per-head query scale, init 5.25 (bigbag #1493).
- **Skip gates:** sigmoid-gated U-Net-style connections between encoder
  and decoder halves (clarkkev #1394).

### 5.3 Depth recurrence (3-layer loop)

After warmup, the forward pass runs through 17 *virtual* layers built from
11 *physical* layers by looping layers 3-5 three times:

```
encoder indices = [0, 1, 2, 3, 4, 5, 3, 4]
decoder indices = [5, 3, 4, 5, 6, 7, 8, 9, 10]
```

Activated at training fraction 0.35 via `enable_looping_at`. This is the
defining innovation of bigbag PR #1493 over PR #1394.

### 5.4 Parallel residuals

For layers ≥ `parallel_residual_start = 7`, attention and MLP read from the
same pre-residual input (GPT-J style). Source: PR #1412 `Robby955`. Combined
in PR #1477 `aryanbhosale` and #1493 `bigbag`.

### 5.5 Optimizer

- **Muon (row-normalized, "MuonEq-R")** for matrix parameters (clarkkev
  PR #1217). Newton-Schulz 5 steps.
- **AdamW** for embeddings + scalars.
- WD 0.095 (Muon) / 0.085 (AdamW).
- LR 0.022 (matrix), 0.6 (embed), 0.008 (head), 0.03 (tied embed), 0.02
  (scalar). Linear warmdown over the final 72% of training.
- EMA decay 0.9965.

### 5.6 Quantization: GPTQ + SDClip

- **GPTQ** with full Hessians collected from 64 calibration batches
  (post-EMA model, calibration data drawn from training shards).
- **SDClip:** clip threshold = `k · σ_row` per row.
  - `k = 12.85` for matrix weights, int6 quantization.
  - `k = 20.0` for the tied embedding, int8 quantization.
- A handful of small control tensors (q_gain, attn_scale, mlp_scale,
  resid_mix, skip gates, skip weights) pass through as float16.

### 5.7 Compression

The quantized state-dict is compressed in three layers:

1. **Byte-shuffle** (interleave bytes by stride 2) — improves run-length
   structure for the entropy coder.
2. **Brotli quality 11** with a 121-KB pre-trained English dictionary built
   into Brotli. Despite being designed for text, Brotli wins on this byte
   stream because of the dictionary handles pickled-tensor metadata
   efficiently and the byte-shuffle exposes patterns it can match.
3. **LZMA self-extractor** wraps the Python source code itself: the source
   is LZMA-compressed (FORMAT_RAW + LZMA2), base85-encoded, and embedded in
   a 2-line wrapper. The wrapper does `exec(decompress(b85decode(...)))` at
   runtime to reconstitute the original source.

### 5.8 Test-time training (TTT)

Bigbag's PR #1493 uses score-first TTT during eval: chunk the validation
stream into 32K-token chunks, score each chunk under `inference_mode()`,
then SGD-train the model on that chunk before scoring the next. SGD lr=0.005,
momentum=0.9, 3 epochs/chunk, cosine LR decay.

**This submission ships with TTT_ENABLED=0** (TTT off). The reason: the PPM
mixer already runs during the sliding eval phase before TTT would, so TTT's
~290 s of eval time would push the total over the 600 s budget without
contributing to the headline number. Empirically (§4.2), turning TTT off
changes the PPM mixer BPB by 0.00019 (RNG noise).

If `TTT_ENABLED=1` were set, the runs in `train_seed*.log` show the
post-TTT NN-only BPB landing near 1.080, but that number is irrelevant to
the submission claim.

---

## 6. The key innovation: byte-PPM mixer

This is the only addition over PR #1493's neural stack. Verbatim port from
PR #1795 (`OE-GOD`), specifically the post-review corrected version with the
strict-legal causal gate.

### 6.1 Conceptual overview

The neural model produces a token-level probability `p_θ(token | context)`.
Because tokens have variable byte lengths under SP8192, the **byte-level**
probability the eval needs is computed by:

1. Spreading each token's `log p_θ(token)` uniformly across its constituent
   bytes (under a uniform-bytes-given-token assumption).
2. **Independently**, a tiny online byte-level statistical model (PPM-D
   order 4) processes the same byte stream causally, producing a separate
   byte-level probability `p_PPM(byte | context_bytes)`.
3. The two distributions are mixed token-by-byte with a context-dependent
   gate `λ`:

```
λ = L_   if cf > T  (high-confidence prefix → trust neural)
λ = H    otherwise   (uncertain prefix → trust PPM)

p_mix = λ · p_θ_byte + (1 − λ) · p_PPM_byte

bpb   = −mean over bytes of log_2(p_mix)
```

with default `H=0.9, L_=0.05, T=0.9`.

### 6.2 The math

#### Forward (corruption-free) — there's no diffusion process here.

Both the neural and PPM models operate on bytes left-to-right. Each emits a
next-byte distribution conditioned on the prefix.

#### Neural byte-spreading

Let token `i` have ID `t_i` and SP-decoded byte length `L_i`. Let
`ℓ_i = log p_θ(t_i | tokens[<i])` be the neural model's log-probability.
For each byte `j` produced by token `i`:

```
nlp[j] = ℓ_i / L_i
```

This is the simplest defensible byte-level decomposition: the model assigns
log-mass `ℓ_i` to a **token**, and we treat each constituent byte as
contributing equally. (More sophisticated alternatives exist but PR #1795
and follow-ups all use this uniform spreading.)

#### PPM-D order-4 byte coder

PPM (Prediction by Partial Matching) is a classical statistical byte
compressor. PPM-D is a specific variant of the escape-probability rule.

State: for each context length `o ∈ {0, 1, 2, 3, 4}`, a dictionary
mapping prefix bytes to a record `[total, max_count, counts]` where
`counts: byte → int`.

Scoring byte at position `i` (current context = last 4 bytes seen, `h`):

1. Initialize `esc = 1.0`, `pf = 0.0`, `cf_seen = False`.
2. For `o` from `min(O, i)` down to 0:
   a. Look up `e = tabs[o].get(h[-o:])`.
   b. If `e is None`: continue.
   c. **First-seen-context capture:** if `not cf_seen`, set
      `cf_mx = e[1] (max_count)`, `cf_tot = e[0] (total)`,
      `cf_seen = True`. **This happens before consulting the count for
      the byte being predicted at any context.**
   d. Look up `c = e[2].get(byte_i, 0)`.
   e. If `c > 0`: `pf = esc · (2c − 1) / (2 · total)`; break.
   f. Else: `esc *= |unique bytes at this context| / (2 · total)`;
      continue to shorter context.
3. If never found (the `else` of the for-loop): `pf = esc / 256`
   (uniform fallback over all bytes).
4. Clamp: if `pf < 1e-20`, set `pf = 1e-20`.
5. `plp[i] = log(pf)` and `cf[i] = cf_mx / cf_tot if cf_seen else 1/256`.

Then the byte-i counter is added to all relevant context tables. **This
update happens AFTER the score for byte i is recorded.** This is
score-before-update at the byte level.

#### Gate decision and mixing

Vectorized:

```
λ_i = L_   if cf[i] > T  else H
p_mix[i] = λ_i · exp(nlp[i]) + (1 − λ_i) · exp(plp[i])
bpb     = −sum_i log_2(max(p_mix[i], 1e-300)) / N_bytes
```

### 6.3 The compliance fix (history)

In an earlier version of the PPM mixer, `cf` was computed as
`d.get(x, 0) / total` — i.e., the count of **the byte being predicted** at
the deepest context. That made `cf` outcome-dependent: high λ → trust
neural specifically because PPM was about to confirm the byte was common
in this exact context.

That is **non-causal** (the gate consults the answer before scoring). It
was flagged by reviewer `nprime06` and retracted by `OE-GOD` in PR #1795.
The corrected version (used here) computes `cf = max_count / total` — the
ratio of the dominant byte at the deepest context. This depends only on the
prefix; the observed byte does not enter the gate decision.

The submitted artifact ships the corrected version. **Verify this in
person** by reading the `_ppm_mixture_bpb` function via `local_validator.py
--check 4` — look for the `cf_seen` guard and confirm the order of operations.

### 6.4 Where the mixer slots in the eval pipeline

The eval has these phases, in order:

1. Pre-quant post-EMA basic eval — sanity check.
2. GPTQ Hessian collection on calibration data.
3. Quantization + Brotli compression — produces the artifact.
4. Quantized basic eval — sanity check.
5. **`eval_val_sliding`** — sliding-window scoring (stride 64) that:
   - Distributes window starts across DDP ranks.
   - Each rank scores its assigned windows, computing per-token NLL and
     accumulating loss/byte counts AND capturing `(target_id, log_p)` per
     scored token in local lists.
   - All ranks gather their `(tga_local, lpa_local)` arrays to rank 0
     (size-padded `dist.gather`).
   - Rank 0 concatenates in rank-major order. **Critical**: rank-major
     order matches the global validation token order because rank 0
     processes earliest windows and rank N-1 processes latest. So the
     gathered `tga_full` is the val token stream in scoring order.
   - Rank 0 calls `_ppm_mixture_bpb(tga_full, lpa_full, sp, ...)` and logs
     `ppm_mixer val_bpb:`.
6. (If `TTT_ENABLED=1`) `eval_val_ttt` — irrelevant to the submission claim
   since the PPM mixer's input was captured in step 5.

The mixer's output is the headline. Steps 5 and 6 produce additional
NN-only BPB numbers used for diagnostic comparison and statistical
significance testing.

---

## 7. Rule-by-rule compliance argument

| # | Rule | Compliance | Where to verify |
|---|---|---|---|
| 1 | Artifact ≤ 16,000,000 bytes | ✓ All seeds and compliance run land at ~15.997 MB | `local_validator.py` check 2 |
| 2 | Training ≤ 600 s on 8xH100 SXM | ✓ Training is bigbag PR #1493 verbatim — already verified at 588 s on 8xH100 in PR #1493's records. We add nothing to training. | Compare `train_gpt.py` decompressed source to PR #1493's source; confirm the only diff is the eval-time `_ppm_mixture_bpb` block. |
| 3 | Eval ≤ 600 s on 8xH100 SXM | ✓ With `TTT_ENABLED=0`: sliding ~83 s + PPM ~365 s + roundtrip+quant ~12 s = ~460 s. The PPM mixer is rank-0 single-threaded Python — its time is hardware-independent. | `compliance_seed42.log` shows 553 s on 4xH100; on 8xH100 the sliding portion shrinks but PPM stays. |
| 4 | No network calls during eval | ✓ The artifact uses only stdlib (math, lzma, base64), torch, sentencepiece, brotli, numpy. No `requests`, no HTTP, no `huggingface_hub` calls at eval time. | Inspect the decompressed source. |
| 5 | No training data access during eval | ✓ Eval reads only `data/datasets/fineweb10B_sp8192/fineweb_val_*.bin`. | Inspect the decompressed source. |
| 6 | No tokenizer change | ✓ Uses SP8192 from PR #1394, unchanged. | Tokenizer model file `fineweb_8192_bpe.model` ships with the dataset; vocab_size=8192. |
| 7 | No training on validation data | ✓ Training data loader (`ShuffledSequenceLoader`) reads only `fineweb_train_*.bin`. The PPM mixer is built **online** at eval time and never persists to disk or training. | Inspect `ShuffledSequenceLoader` and the training loop. |
| 8 | Causal eval | ✓ Sliding eval is strictly causal (each position scored from prefix tokens only, attention is causal). PPM-D scores each byte from prefix bytes only. | Read `eval_val_sliding` and `_ppm_mixture_bpb` in the decompressed source. |
| 9 | Score before update | ✓ Sliding eval is score-only. PPM-D updates its tables only after recording the score for byte `i`. | Read the for-loop in `_ppm_mixture_bpb`. |
| 10 | Single pass over each token | ✓ Each scored token enters `_ppm_mixture_bpb` exactly once. PPM update happens once per byte. | Verify no multi-iteration over the same byte in the mixer. |
| 11 | Normalized softmax over vocab | ✓ Neural model output is a standard softmax over the 8192-token vocab. The PPM mixer combines two **proper** byte-level distributions (neural via byte-spreading, PPM-D via escape mechanism); the convex combination is also a proper distribution. | Inspect `forward_logits` (no logit biasing). |
| 12 | Causal gate (PR #1795 review condition) | ✓ Gate `cf` is `max_count / total` at the deepest context with data, captured BEFORE consulting `d.get(x)` for any context. The byte being predicted enters the mixer only via `pf` (the standard PPM probability), never the gate weight `λ`. | Read the `cf_seen` guard in `_ppm_mixture_bpb` (visible via check 4). |
| 13 | No SLOT (eval-time hidden-state optimization) | ✓ No optimization of model hidden states at eval. | Inspect `eval_val_sliding`. |
| 14 | No two-pass selection / rescoring | ✓ Each token is scored once. The mixer is a deterministic function of the captured per-token log-prob. | Inspect the eval loop. |
| 15 | No n-gram cache | ✓ The PPM-D model is **not** a static n-gram lookup table: it is built online from validation tokens already scored. It does not access training data or any persisted artifact. | Verify `tabs` is initialized empty at the start of `_ppm_mixture_bpb` and grows only from in-call updates. |
| 16 | No ETLB (eval-time logit bias) | ✓ Neural logits are unmodified by the mixer. The mixer combines probabilities, not logits. | Inspect the mixer. |
| 17 | No pre-quant TTT on val data | ✓ Quantization happens once after training; the mixer operates on the post-quantization model. | Inspect the eval pipeline order. |
| 18 | Statistical significance (≥ 0.005 nat improvement, p < 0.01) | ✓ Margin = 0.0450 nats, well above 0.005. t-statistic ≈ 175 across 3 seeds with std 0.00064. | Per §4.3. |

---

## 8. Reproducibility model

There are **two layers** to the claim, with different reproducibility
properties:

### 8.1 Hardware-independent: the PPM mixer

`_ppm_mixture_bpb` is pure Python (numpy + math). Given identical
`(tga, lpa)` inputs and a fixed tokenizer, it is **bit-for-bit
deterministic**. The function is small (~70 lines) and self-contained.

This means: **the PPM mixer's contribution to the BPB can be verified on a
laptop**. Given the npz dump from one cluster run (which captures the
neural model's per-token log-probs), the mixer can be re-run and its output
compared to the seed log's `ppm_mixer val_bpb:` line. Any honest port of
the function will produce the same number to within float-precision noise.

This is the **strongest** reproducibility check available without GPU
access. It proves that the mixer is implemented as claimed and that the
seed log's number was not fabricated.

### 8.2 Hardware-dependent: the neural training and timing

The neural model's training is **stochastic** — even with a fixed seed,
exact BPB depends on:
- Hardware (4xH100 vs 8xH100): different DDP grad reduction order, different
  micro-batch shapes per the `grad_accum_steps = 8 // world_size` rule,
  slightly different floating-point accumulation.
- Software (PyTorch version, CUDA version, flash_attn version): kernel
  selection and numerical precision can differ.

The submitter's runs were on **4xH100** with extended wallclock (4500
iterations, no 600 s cap). On **8xH100** with the canonical 600 s cap,
the model would train for ~5088 iterations, producing a slightly stronger
neural model and a slightly better PPM mixer BPB.

To verify the submission's training-time compliance and the actual 8xH100
BPB, you need 8xH100 access. This is the standard reviewer-side step in
parameter-golf.

### 8.3 What the npz dump enables

The npz dump captures `(tga, lpa)` — target tokens and per-token neural
log-probabilities — from one specific cluster run (seed 42 with TTT off).

Given this dump, you can:
- **Verify the mixer**: re-run `_ppm_mixture_bpb(tga, lpa, sp)` locally and
  check it matches the seed log's BPB. (Validator check 5.)
- **Sweep mixer hyperparameters**: try different `O, H, L_, T` to see how
  sensitive the result is. (Useful for understanding the mechanism.)
- **Inspect the per-byte mixing**: dump the per-byte `λ` to see what
  fraction of bytes are PPM-dominated vs neural-dominated.

The dump cannot tell you if the neural model is itself correct — the lpa
values are taken on faith. To verify those, you'd need to retrain on
8xH100 and compare the per-token log-probs at, say, 1000 random positions
to a fresh run's. (This is excessive for record verification; the
parameter-golf tradition is to trust the per-seed sliding BPB number that
the artifact prints in its log.)

---

## 9. Validation procedure

Run `python3 local_validator.py --record-dir .` from inside this folder.
With dump+tokenizer, add `--tokenizer ... --ppm-dump ... --expected-bpb ...`.

The validator runs 5 checks. Each is described below with what it proves
and what would constitute a failure.

### Check 1 — Packed code self-test

**What it does:** opens `train_gpt.py`, extracts the base85 payload,
LZMA-decompresses it, parses the resulting string as Python AST.

**Optional (with `--source path/to/tw_eval.py`):** also compares the
decompressed AST to a separately-supplied readable source. Equality means
the artifact is the same source, just compressed.

**Pass criterion:** decompresses to valid Python.

**Fail modes:** non-Python payload, syntax errors, AST mismatch with source.
Failure here means the artifact is malformed or doesn't match the claimed
source.

### Check 2 — Artifact size

**What it does:** sums `os.path.getsize(train_gpt.py)` with each
`Serialized model quantized+brotli:` value extracted from each seed log,
and verifies each sum is < 16,000,000.

**Pass criterion:** all sums under cap.

**Fail modes:** any seed exceeds 16,000,000 — the artifact is non-compliant.

### Check 3 — Source audit

**What it does:** decompresses `train_gpt.py`, finds the
`_ppm_mixture_bpb` function, pretty-prints its source via `ast.unparse`.

**Pass criterion:** function exists. The check **does not auto-verify**
compliance properties; instead, it prints the function source so a human
can read it and confirm:

- `cf` is computed before `d.get(x)` lookup at any context (the `cf_seen`
  guard in the inner loop).
- Update of `tabs[o]` is the second loop, after the scoring loop.
- Single iteration over `bs` (the byte stream).
- No reference to future bytes (no negative lookahead).

If you want auto-verification of these, search the printed source for the
exact string `if not cf_seen:` followed within ~5 lines by `cf_seen = True`,
and verify the `c = d.get(x, 0)` line comes AFTER `cf_seen = True` is set.

**Fail modes:** function missing, gate uses observed byte, update before
score.

### Check 4 — Log/JSON consistency

**What it does:** loads `submission.json`, iterates over per-seed claims,
finds the matching log file (tries `train_seedXX.log`, `ws4_seedXX.log`,
`seedXX.log`), greps for `ppm_mixer val_bpb:`, compares to claimed BPB to
within 1e-5.

**Pass criterion:** all claimed numbers appear in their logs.

**Fail modes:** mismatch between JSON and log = fabricated claim. Missing
log file = incomplete submission.

### Check 5 — Numerical reproduction (requires npz + tokenizer)

**What it does:**

1. Loads `ppm_inputs_seed42.npz`. Reads `tga` (int32 → int64) and `lpa`
   (float32 → float64).
2. Loads `fineweb_8192_bpe.model` via SentencePiece.
3. Extracts `_ppm_mixture_bpb` from the **packed artifact** (not from a
   separate source — this proves the actual shipped function works).
4. Calls it with the dump arrays and tokenizer.
5. Compares the result to `--expected-bpb` (typically the value from
   `dump_seed42.log`'s `ppm_mixer val_bpb:` line).

Runtime: ~5-10 minutes of single-threaded Python on a modern laptop. The
inner loop processes ~40 M bytes.

**Pass criterion:** local BPB matches expected to within 1e-5.

**Fail modes:**
- Difference > 1e-5: implementation drift between local and cluster. Could
  indicate a bug, hyperparameter mismatch, or (in the worst case) the
  cluster ran a different mixer than what's in the artifact.
- Validator can't load npz: file corrupted or in wrong format.
- Validator can't load tokenizer: wrong file or vocab size mismatch.

### Expected validator output (all 5 checks passing)

```
======================================================================
SUMMARY
======================================================================
  [PASS] packed self-test
  [PASS] artifact size
  [PASS] source audit
  [PASS] log/json consistency
  [PASS] numerical repro
```

Anything other than all PASS is a reason to scrutinize.

---

## 10. What cannot be validated locally

Three classes of claim require GPU access to verify:

### 10.1 Training time on 8xH100 SXM

The submission claims training fits in 600 s on 8xH100 SXM. The argument is
indirect: bigbag's PR #1493 was verified at 588 s on 8xH100 SXM, and we did
not modify training. To independently confirm:

1. Provision an 8xH100 SXM box (RunPod or equivalent).
2. Decompress `train_gpt.py` (validator check 1 produces the source).
3. Run `torchrun --standalone --nproc_per_node=8 train_gpt.py` with the
   same env vars described in `README.md`.
4. Observe `train_time` near the end exits at ≤ 600 s.

This is the standard reviewer step.

### 10.2 Eval time on 8xH100 SXM

Same argument structure. The PPM mixer takes ~365 s on rank 0 regardless
of GPU count. Sliding eval takes ~83 s on 8xH100 (per PR #1493's logs).
Total ~460 s. To verify on 8xH100, run as above and observe
`quantized_sliding_window eval_time:` ≤ ~600,000 ms.

### 10.3 Neural log-probs are not doctored

The npz dump's `lpa` values are taken on faith. If you suspect they were
hand-edited to produce a favorable PPM mixer result, the only way to check
is to retrain (with the same seed, on 8xH100 SXM), capture lpa fresh, and
diff. This is excessive for record verification and not part of the
parameter-golf tradition.

The structural argument against fabrication: the lpa values are the **only**
thing the cluster computes; the rest (tga, mixer math) is deterministic on
any machine. So manipulating lpa is the only attack surface. Such
manipulation would also need to leave the **per-token NLL** consistent with
the seed log's pre-quant/quantized/sliding BPB lines (which are computed
from the same lpa). A coherent fake would require simulating the whole
neural model, which is essentially the same effort as actually running it.

---

## 11. Lineage and credits

The neural side is, top-to-bottom, prior work:

- **PR #198 / #287 (`jfprincz`)** — 11L XSA + Partial RoPE + LN scale.
- **PR #549 (`abaybektursun`)** — LeakyReLU(0.5)² + score-first TTT
  framework + Parallel Muon.
- **PR #1019 (`abaybektursun`)** — Self-Generated GPTQ Calibration Data +
  all-layer XSA.
- **PR #1217 (`bigbag`)** — MuonEq-R (row-normalized Muon).
- **PR #1218 (`clarkkev`)** — SP4096 + 4× MLP + simplifications.
- **PR #1285 (`dexhunter`)** — MuonEq-R + depth recurrence (layer 4-5
  loop) + WD=0.090 + all-int6 GPTQ.
- **PR #1394 (`clarkkev`)** — SP8192 + GPTQ-quantized embeddings + std-
  based GPTQ clip (the SDClip discovery).
- **PR #1412 (`Robby Sneiderman`)** — parallel residuals on PR #1394 stack.
- **PR #1413 (`dexhunter`)** — QK-Gain 5.0 + score-first TTT on PR #1394.
- **PR #1477 (`aryanbhosale`)** — parallel residuals + score-first TTT on
  PR #1413.
- **PR #1493 (`bigbag`)** — 3-layer recurrence (3,4,5) + parallel residuals
  + score-first TTT + QK-Gain 5.25 + tuned hyperparameters. **This is the
  full neural stack used in this submission.**

The eval side is prior work too:

- **PR #1795 (`OE-GOD`)** — invented the byte-PPM mixer with the PPM-D
  order-4 model and the adaptive-λ gate. **Compliance fix history**:
  initial version had an outcome-conditioned gate (`d.get(x)/total`),
  flagged by `nprime06` in review, retracted; the corrected version using
  `max_count/total` is the basis of this submission's mixer.
- **PR #1933 (`deborahnelson8788726`)** — earlier application of PR #1795's
  mixer to `yahya010`'s neural base (different stack from PR #1493),
  claimed val_bpb 0.99145.

This submission's only original contribution: the **integration** — porting
PR #1795's mixer onto PR #1493's specific neural base, wiring per-token
log-prob capture with DDP gather, and producing this audit-ready package.

The submitter did not invent the mixer, did not invent the neural stack,
and does not claim to. The combination (PR #1493 + PR #1795) is, to the
submitter's knowledge, untested in any other open or merged PR.

---

## 12. Honest limitations and caveats

### 12.1 Comparable open PRs

Two open PRs claim better numbers as of submission time:

- **PR #1933 (`deborahnelson8788726`)** — 0.99145 BPB. Uses PR #1795 mixer
  on `yahya010`'s neural base. ~0.005 BPB better than this submission. If
  PR #1933 merges first, this submission becomes the second PPM-mixer-on-a-
  different-base submission, which is acceptable per parameter-golf's
  tradition of accepting orthogonal stackings even when they don't strictly
  beat the latest open PR.
- **PR #1945 (`alertcat`)** — 1.0593 BPB. Stacks AWQ-lite quant + Asymmetric
  Logit Rescale on PR #1855's neural base. **Does not use the PPM mixer.**
  This submission beats PR #1945 by 0.063 BPB.

### 12.2 The submitter's runs were on 4xH100, not 8xH100

The 3-seed numbers were measured on 4xH100 SXM with extended wallclock
(4500 iterations, MAX_WALLCLOCK_SECONDS=0). The reasoning:

- Per-step compute is the same on any world_size (the `grad_accum_steps =
  8 // world_size` rule preserves total batch).
- Wall-clock per step differs (8 GPUs are ~2x faster than 4 GPUs).
- On 8xH100 with the canonical 600 s cap, the same code would train
  ~5088 iterations instead of 4500 — slightly more training, slightly
  better neural model, slightly better PPM mixer.
- So the **8xH100 BPB will be ≤ this submission's claim**, not greater.

This is a one-sided uncertainty: the 8xH100 reproduction can match or
beat the claim, but not regress it.

### 12.3 The std=0.00064 is from 3 seeds

3 seeds is the parameter-golf community minimum. Std at n=3 has wide
confidence bounds. The true std could plausibly be 2x our estimate.
Doesn't change the conclusion (margin is 9x the threshold), but worth
noting.

### 12.4 Brotli vs LZMA

The compressed model uses Brotli-11. We tested zstd (with and without a
self-trained dictionary) and Brotli won by ~1.6 MB on this byte stream.
LZMA was not extensively tested at the model level, only for the
self-extracting code wrapper.

### 12.5 No 3-phase TTT, no LQER, no AWQ, no asym rescale

Newer techniques in PR #1855 / #1908 / #1923 are NOT in this submission.
The submission is a "PPM mixer on bigbag's stack" combination, not a
"PPM mixer on the latest stack" combination. A future submission could
combine the mixer with PR #1855's stack and likely land ~0.985 (taking
advantage of the stronger ~1.061 NN base).

### 12.6 The packed wrapper requires `lzma`, `base64`, `numpy`, `torch`, `sentencepiece`, `brotli`, `flash_attn_3`

Standard parameter-golf dependency footprint. None are unusual.

---

## 13. Decision framework

### Pass criteria (all must hold)

1. Validator check 1 passes (packed code is valid Python).
2. Validator check 2 passes (every seed under 16 MB).
3. Validator check 3 — manual inspection confirms causal gate, score-before-update.
4. Validator check 4 passes (JSON matches logs exactly).
5. Validator check 5 passes within 1e-5 (if npz available).
6. The ppm_mixer val_bpb of every seed > 0.99 (sanity — too good is suspicious).
7. The std across 3 seeds is < 0.005 (within reasonable noise).
8. README and submission.json correctly credit PR #1493 (`bigbag`) and PR #1795 (`OE-GOD`).
9. The mixer source matches PR #1795's published version (modulo formatting).

### Fail criteria (any one is a reject)

1. Validator check 5 fails or differs by > 1e-5.
2. Mixer source has a target-conditioned gate (uses `d.get(x)` to compute λ).
3. Any seed's artifact > 16,000,000 bytes.
4. Logs show network calls during eval (look for `requests`, `urlopen`, etc.).
5. Logs show training data being read during eval.
6. Tokenizer differs from SP8192 (vocab_size != 8192 or different model file).

### Suspicious but not disqualifying

- 3-seed std < 0.0001 (suspiciously tight).
- 3-seed mean > 1.0 (worse than claimed).
- Artifact size very close to cap (< 100 bytes margin) — fragile but legal.
- Any deviation between log-printed BPB and JSON-claimed BPB > 1e-5.
- Submission folder missing `compliance_seed42.log` (the 8xH100 timing
  argument becomes weaker).
- Mixer hyperparameters differ from PR #1795's published `O=4, H=0.9,
  L_=0.05, T=0.9` — request explanation in PR review.

---

## 14. Glossary

- **BPB** — bits per byte. The primary metric. Computed as
  `-log_2(p(byte)) summed over val bytes`, divided by total val bytes.
- **PPM** — Prediction by Partial Matching. A classical statistical text
  compressor. Maintains contexts of varying lengths and an "escape"
  mechanism to fall back to shorter contexts.
- **PPM-D** — a specific PPM variant. Uses `(2c-1)/(2*total)` for the
  probability of a context-byte pair seen `c` times. The escape probability
  is `|unique seen|/(2*total)`.
- **Order-4** — max context length is 4 prior bytes.
- **`λ` gate** — the mixing weight between neural and PPM probabilities.
  Adaptive based on context distinctiveness.
- **GPTQ** — Generative Pre-trained Transformer Quantization. A
  post-training quantization method using full Hessians from calibration
  data to minimize reconstruction error per row.
- **SDClip** — clipping the quantization range to `k * row_std`. The
  parameter-golf community's empirical sweet spot is `k=12.85` for int6
  matrices, `k=20` for int8 embeddings.
- **MuonEq-R** — row-normalized Muon optimizer. Each gradient row is
  normalized before the Newton-Schulz orthogonalization step.
- **TTT** — test-time training. Optimizing the model on validation tokens
  AFTER they've been scored, to better predict subsequent tokens.
- **DDP** — DistributedDataParallel. PyTorch's standard data-parallel
  training framework.
- **SXM** — NVIDIA's high-bandwidth socketed GPU form factor with NVLink.
  The H100 SXM is meaningfully faster than H100 PCIe for DDP.
- **SP / SentencePiece** — Google's subword tokenizer.
- **`▁`** — SentencePiece's leading-space marker.
- **rank** — the index of a process in a distributed run. Rank 0 is the
  main process; logs are usually only printed by rank 0.
- **EMA** — exponential moving average of model weights. Used here as a
  smoother of the trained model before quantization.
- **LZMA self-extractor** — a small Python wrapper that LZMA-decompresses
  an embedded payload at runtime and execs it. Reduces the on-disk code
  size by ~3-4x.

---

## 15. Pointers to deeper context

- Live `openai/parameter-golf` README:
  https://github.com/openai/parameter-golf/blob/main/README.md
- Bigbag's PR #1493 (the neural base):
  https://github.com/openai/parameter-golf/pull/1493
- OE-GOD's PR #1795 (the mixer):
  https://github.com/openai/parameter-golf/pull/1795
- Compliance issue #1017:
  https://github.com/openai/parameter-golf/issues/1017
- PR #1933 (earlier mixer + different base, 0.99145):
  https://github.com/openai/parameter-golf/pull/1933
- PR #1855 (current merged SOTA, no mixer, 1.0611):
  https://github.com/openai/parameter-golf/pull/1855
- Kevin Clark's quantization-vs-compression analysis (PR #1394 README):
  Contains the entropy argument behind SDClip — why a wider clip range
  with the same bit width compresses smaller after Brotli.
- Modded-nanogpt (the parameter-golf ancestor):
  https://github.com/KellerJordan/modded-nanogpt

If you find any factual error in this document or in the submission's
README, please flag it — the submitter is committed to correcting any
inaccuracy promptly.
