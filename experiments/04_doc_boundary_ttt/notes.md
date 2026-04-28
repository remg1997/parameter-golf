# Experiment 04 — Doc-Boundary Chunking for TTT

## What changed
TTT chunks were arbitrary 32K-token slices that straddled FineWeb document boundaries. Now (when `TTT_DOC_BOUNDARY_ENABLED=1`) chunks align with doc boundaries: greedy walk of doc-separator positions, emit a chunk when accumulated >= `TTT_CHUNK_TOKENS`. Fall back to a fixed cut at `chunk_start + target` if no boundary is within `[min, 2*target]`. Last chunk extends to `total_tokens`.

## Why
TTT adapts to "context just seen", but the SOTA's fixed cuts mean SGD updates within a chunk see partial documents — adapting to one topic's tail and another's head simultaneously. Doc-aligned chunks let TTT specialize on coherent topical context per chunk.

## Doc-sep token
`val_data.sp.bos_id()` (= 1 for the SP8192 tokenizer). FineWeb's tokenized stream prepends BOS at every doc start; EOS is not appended. Override via `TTT_DOC_SEP_TOKEN=<id>` if needed.

## Compliance preserved
- Score-first ordering unchanged: every token still scored under `inference_mode()` BEFORE any SGD update.
- `is_last_chunk` guard preserved: short tail chunks are scored but not trained on.
- No two-pass scoring, no logit biasing, no n-gram cache, no SLOT.

## Expected gain
~0.001 BPB. Low-risk, near-free change (eval-only, no training cost). Combines orthogonally with `01_per_group_sdclip` and `03_qat_sdclip`.

## Ablation
`TTT_DOC_BOUNDARY_ENABLED=0` (default) -> byte-identical to SOTA.

## Touched
- `Hyperparameters` (3 new fields)
- New helpers `_find_doc_boundaries`, `_chunk_at_doc_boundaries`
- `eval_val_ttt` chunking + window->chunk mapping (uses precomputed `chunk_ranges` instead of fixed-stride math)

## Run
```bash
source experiments/04_doc_boundary_ttt/config.env
RUN_ID=04_docttt_seed42 SEED=42 \
  torchrun --standalone --nproc_per_node=3 experiments/04_doc_boundary_ttt/tw_eval.py
```
