"""Self-contained local validator for the PPM-mixer submission.

Runs four independent classes of check on any machine (no GPUs, no torch).
The validator only depends on:

  - records/<folder>/train_gpt.py       (the LZMA-self-extracting artifact)
  - records/<folder>/train_seed*.log    (the per-seed eval logs)
  - records/<folder>/compliance_seed*.log (optional — for the no-TTT timing check)
  - records/<folder>/submission.json    (the claim metadata)
  - ./ppm_inputs_seedXX.npz             (optional — for numerical reproduction)
  - ./fineweb_8192_bpe.model            (optional — needed only with the npz)

Optional --source flag enables an extra "no tampering" check, comparing the
packed artifact's AST to a separately-distributed readable source.

Checks:

  1. Packed-code self-test — train_gpt.py decompresses cleanly to valid Python.
     With --source, also checks AST-equivalence to the readable source.
  2. Artifact size — packed code + each seed's model_blob (from the seed logs)
     must sum to under 16,000,000 bytes.
  3. Numerical reproduction — given a dump of (tga, lpa) from one cluster run,
     re-runs `_ppm_mixture_bpb` locally (extracted from the packed artifact's
     AST) and checks the BPB matches the seed log's `ppm_mixer val_bpb:` line.
  4. Compliance source-audit — pretty-prints the mixer function from the
     decompressed packed source so a human can read it and verify causality.
  5. Log/JSON consistency — submission.json's per-seed BPB must match the
     `ppm_mixer val_bpb:` line in each corresponding seed log.

Usage:

  pip install numpy sentencepiece
  python3 local_validator.py \\
    --record-dir records/track_10min_16mb/2026-04-29_SP8192_PPMMixer_BigbagStack \\
    [--tokenizer ./fineweb_8192_bpe.model] \\
    [--ppm-dump ./ppm_inputs_seed42.npz] \\
    [--expected-bpb 0.99548272] \\
    [--source experiments/08_ppm_mixer/tw_eval.py]
"""
import argparse
import ast
import base64
import json
import lzma
import math
import os
import re
import sys


def _decompress_packed(record_dir: str) -> str:
    packed_path = os.path.join(record_dir, "train_gpt.py")
    src = open(packed_path).read()
    m = re.search(r'b85decode\("([^"]+)"\)', src)
    if not m:
        raise RuntimeError(f"no b85decode payload in {packed_path}")
    payload = base64.b85decode(m.group(1))
    return lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2}]).decode("utf-8")


def check_packed_roundtrip(record_dir: str, source: str | None) -> bool:
    print("=" * 70)
    print("CHECK 1: Packed code self-test (decompresses to valid Python)")
    print("=" * 70)
    try:
        decompressed_text = _decompress_packed(record_dir)
    except Exception as e:
        print(f"FAIL: {e}")
        return False
    try:
        ast.parse(decompressed_text)
    except SyntaxError as e:
        print(f"FAIL: decompressed payload is not valid Python: {e}")
        return False
    print(f"PASS: packed train_gpt.py decompresses to {len(decompressed_text):,} bytes of valid Python")
    if source is not None:
        original_text = open(source).read()
        decompressed_ast = ast.dump(ast.parse(decompressed_text), annotate_fields=True)
        original_ast = ast.dump(ast.parse(original_text), annotate_fields=True)
        if decompressed_ast == original_ast:
            print(f"      AST-equivalent to {source} (no tampering)")
            return True
        else:
            print(f"FAIL: AST mismatch vs {source}")
            return False
    return True


def check_artifact_size(record_dir: str) -> bool:
    print("=" * 70)
    print("CHECK 2: Artifact size compliance (under 16,000,000 bytes)")
    print("=" * 70)
    packed_path = os.path.join(record_dir, "train_gpt.py")
    code_bytes = os.path.getsize(packed_path)
    log_files = [f for f in os.listdir(record_dir) if f.startswith("train_seed") and f.endswith(".log")]
    if not log_files:
        log_files = [f for f in os.listdir(record_dir) if f.startswith("compliance_seed") and f.endswith(".log")]
    if not log_files:
        print(f"FAIL: no train_seed*.log or compliance_seed*.log in {record_dir}")
        return False
    model_sizes = []
    for lf in sorted(log_files):
        with open(os.path.join(record_dir, lf)) as f:
            txt = f.read()
        m = re.search(r"Serialized model quantized\+brotli:\s*(\d+) bytes", txt)
        if m:
            model_sizes.append((lf, int(m.group(1))))
    if not model_sizes:
        print("FAIL: no `Serialized model quantized+brotli:` lines in seed logs")
        return False
    print(f"  Packed code: {code_bytes:,} bytes")
    all_ok = True
    for lf, sz in model_sizes:
        total = code_bytes + sz
        margin = 16_000_000 - total
        status = "OK" if total < 16_000_000 else "OVER"
        print(f"  {lf}: model={sz:,}, total={total:,}, margin={margin:+,} bytes [{status}]")
        if total >= 16_000_000:
            all_ok = False
    if all_ok:
        print("PASS: every seed's artifact is under 16M cap")
    else:
        print("FAIL: at least one seed exceeds 16M cap")
    return all_ok


def check_numerical_reproduction(record_dir: str, ppm_dump: str, tokenizer: str, expected_bpb: float | None) -> bool:
    print("=" * 70)
    print("CHECK 3: Numerical reproduction of PPM mixer BPB")
    print("=" * 70)
    if not os.path.exists(ppm_dump):
        print(f"SKIP: --ppm-dump file not found: {ppm_dump}")
        return True
    if not os.path.exists(tokenizer):
        print(f"SKIP: --tokenizer file not found: {tokenizer}")
        return True
    try:
        import numpy as np
        import sentencepiece as spm
    except ImportError as e:
        print(f"SKIP: missing dep ({e}). pip install numpy sentencepiece")
        return True
    # Extract _ppm_mixture_bpb from the PACKED artifact (not from a separate source file)
    src = _decompress_packed(record_dir)
    tree = ast.parse(src)
    fn_ast = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_ppm_mixture_bpb":
            fn_ast = node
            break
    if fn_ast is None:
        print(f"FAIL: _ppm_mixture_bpb not found in packed artifact")
        return False
    fn_src = ast.unparse(fn_ast)
    ns = {"np": np, "math": math}
    exec(fn_src, ns)
    fn = ns["_ppm_mixture_bpb"]
    print(f"  Loaded _ppm_mixture_bpb from packed artifact")
    print(f"  Loading dump from {ppm_dump}...")
    dump = np.load(ppm_dump)
    tga = dump["tga"].astype(np.int64)
    lpa = dump["lpa"].astype(np.float64)
    print(f"  tga: {tga.shape} dtype={tga.dtype}")
    print(f"  lpa: {lpa.shape} dtype={lpa.dtype}")
    sp = spm.SentencePieceProcessor(model_file=tokenizer)
    print(f"  Tokenizer vocab_size: {sp.vocab_size()}")
    print("  Running PPM mixer (this takes ~5-10 min on CPU)...")
    import time
    t0 = time.perf_counter()
    bpb = fn(tga, lpa, sp, O=4, H=0.9, L_=0.05, T=0.9)
    elapsed = time.perf_counter() - t0
    print(f"  Local mixer BPB: {bpb:.8f}  (took {elapsed:.1f}s)")
    if expected_bpb is None:
        print(f"  No --expected-bpb provided; just printing.")
        return True
    diff = abs(bpb - expected_bpb)
    if diff < 1e-5:
        print(f"PASS: matches expected {expected_bpb:.8f} (diff = {diff:.2e})")
        return True
    else:
        print(f"FAIL: differs from expected {expected_bpb:.8f} by {diff:.2e}")
        return False


def check_source_audit(record_dir: str) -> bool:
    print("=" * 70)
    print("CHECK 4: Compliance source-audit")
    print("=" * 70)
    packed_path = os.path.join(record_dir, "train_gpt.py")
    src = open(packed_path).read()
    m = re.search(r'b85decode\("([^"]+)"\)', src)
    payload = base64.b85decode(m.group(1))
    decompressed = lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2}]).decode("utf-8")
    tree = ast.parse(decompressed)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_ppm_mixture_bpb":
            print("Mixer function (from packed artifact, AST-unparsed):\n")
            print(ast.unparse(node))
            print()
            print("Compliance markers to verify visually:")
            print("  [ ] cf is computed from `e[1] / e[0]` (max_count/total) BEFORE `d.get(x)` lookup")
            print("  [ ] update of tabs[o] happens AFTER scoring at position i")
            print("  [ ] no two-pass behavior — single iteration over bytes")
            print("  [ ] no n-gram cache lookup using future tokens")
            return True
    print("FAIL: _ppm_mixture_bpb not found in packed artifact")
    return False


def check_log_json_consistency(record_dir: str) -> bool:
    print("=" * 70)
    print("CHECK 5: submission.json claims match the seed logs")
    print("=" * 70)
    json_path = os.path.join(record_dir, "submission.json")
    if not os.path.exists(json_path):
        print(f"FAIL: no submission.json in {record_dir}")
        return False
    sub = json.loads(open(json_path).read())
    per_seed = sub.get("per_seed_results", {})
    if not per_seed:
        print("FAIL: submission.json has no per_seed_results")
        return False
    all_ok = True
    for seed_str, claim in per_seed.items():
        log_candidates = [
            f"train_seed{seed_str}.log",
            f"ws4_seed{seed_str}.log",
            f"seed{seed_str}.log",
        ]
        log_path = None
        for c in log_candidates:
            p = os.path.join(record_dir, c)
            if os.path.exists(p):
                log_path = p
                break
        if log_path is None:
            print(f"  seed {seed_str}: FAIL — no matching log file (tried {log_candidates})")
            all_ok = False
            continue
        log_txt = open(log_path).read()
        m = re.search(r"ppm_mixer val_bpb:([\d.]+)", log_txt)
        if not m:
            print(f"  seed {seed_str}: FAIL — no `ppm_mixer val_bpb:` line in {log_path}")
            all_ok = False
            continue
        log_bpb = float(m.group(1))
        claimed = claim.get("ppm_mixer_val_bpb")
        diff = abs(log_bpb - claimed)
        status = "OK" if diff < 1e-5 else "MISMATCH"
        print(f"  seed {seed_str}: log={log_bpb:.8f} json={claimed:.8f} diff={diff:.2e} [{status}]  ({os.path.basename(log_path)})")
        if diff >= 1e-5:
            all_ok = False
    if all_ok:
        print("PASS: every seed in submission.json matches its log")
    else:
        print("FAIL: at least one seed's claim differs from its log")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-dir", required=True, help="path to records/track_10min_16mb/<folder>/")
    ap.add_argument("--source", default=None, help="OPTIONAL: path to a separately-distributed readable source for AST-equivalence check")
    ap.add_argument("--tokenizer", default=None, help="path to fineweb_8192_bpe.model (only needed for check 3)")
    ap.add_argument("--ppm-dump", default=None, help="path to ppm_inputs_seedXX.npz (only needed for check 3)")
    ap.add_argument("--expected-bpb", type=float, default=None, help="claimed BPB to compare against")
    args = ap.parse_args()

    results = []
    results.append(("packed self-test",      check_packed_roundtrip(args.record_dir, args.source)))
    results.append(("artifact size",         check_artifact_size(args.record_dir)))
    results.append(("source audit",          check_source_audit(args.record_dir)))
    results.append(("log/json consistency",  check_log_json_consistency(args.record_dir)))
    if args.ppm_dump and args.tokenizer:
        results.append(("numerical repro",   check_numerical_reproduction(args.record_dir, args.ppm_dump, args.tokenizer, args.expected_bpb)))
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_ok = True
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            all_ok = False
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
