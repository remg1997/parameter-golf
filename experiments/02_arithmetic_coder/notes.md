# Experiment 02 — Alternative Compressor

## What changed
Adds `zstd` and `zstd_dict` (zstd level 22 with a self-trained dictionary) as
alternatives to Brotli-11 in the artifact compression pipeline. Decompression
re-routes accordingly. Also gates `byte_shuffle` (default ON to match SOTA).

## Why not pure-Python range coding?
Original plan was a from-scratch range coder. Carry-less range coding has subtle
correctness issues (carry propagation, precision) that are easy to get wrong
under time pressure. zstd-22 with a self-trained dictionary is a proven, fast,
well-tested path that captures most of the entropy-bound gain. We can swap in
constriction or a hand-rolled coder later if zstd doesn't deliver.

## Mechanism
- `zstd_dict`: pre-pass scans `quant_raw` (the pickled state-dict bytes), trains
  an 8 KB Zstd dictionary on equal-sized samples, then compresses with that
  dictionary. The dictionary ships embedded in the artifact (4-byte LE length
  prefix + dictionary bytes + payload), so decode is self-contained.
- `zstd` (no dict): plain zstd-22 — useful as a baseline.
- A roundtrip test asserts `_decompress(_compress(x)) == x` before writing the
  artifact (gated by `COMPRESS_ROUNDTRIP_TEST=1`).
- `COMPRESS_COMPARE=1` also encodes the same `quant_raw` with `brotli`, `zstd`,
  and `zstd_dict` for size comparison logged at serialize time.

## Expected gain
0.3–1.0 MB shaved off the ~16 MB artifact. Each MB freed lets us raise `k`
(SDClip) or add ~1 bit somewhere, which translates monotonically to BPB. The
size win itself is BPB-neutral; the *reinvestment* is what produces the gain.

## Combinations
- 02 + 01: extra MB lets early-group `k` go even smaller (more precision).
- 02 + 03: similar; extra MB can absorb the slightly larger STE-trained weights.

## Cost in artifact code-bytes
The decoder is ~30 lines: `_zstd_decompress` + small header parse +
`zstandard.ZstdDecompressor(dict_data=...).decompress(...)`. After LZMA
self-extractor minification, ~200–400 bytes of code overhead.

## Dependency
Requires `zstandard` Python package at runtime. Add to Containerfile.

## Ablation
`COMPRESSOR=brotli` (default) -> byte-identical to SOTA.

## Touched
- `Hyperparameters` (5 new fields: byte_shuffle, zstd_level, zstd_dict_size, compress_compare, compress_roundtrip_test)
- New helpers `_zstd_compress`, `_zstd_decompress`
- `_compress` (signature: now takes byte_shuffle/zstd_level/zstd_dict_size kwargs; dispatches new compressors)
- `_decompress` (signature: now takes byte_shuffle kwarg)
- `serialize` (passes new kwargs, runs compare + roundtrip test)

## Run
```bash
source experiments/02_arithmetic_coder/config.env
RUN_ID=02_zstd_seed42 SEED=42 \
  torchrun --standalone --nproc_per_node=3 experiments/02_arithmetic_coder/train_gpt.py
```
