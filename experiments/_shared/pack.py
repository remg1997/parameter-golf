"""Pack a tw_eval.py source into a bigbag-style LZMA self-extracting wrapper.

Usage:
    python3 experiments/_shared/pack.py SOURCE_PATH [DEST_PATH]

If python-minifier is installed (pip install python-minifier), the source is
minified first, which dramatically improves the compression ratio. Without it,
the raw source is compressed (still useful, just larger).

The output is one Python file with two lines: imports and an exec() of the
LZMA-decompressed, base85-decoded source. Same format as bigbag's PR #1493
artifact in records/.
"""
import argparse
import ast
import base64
import lzma
import os
import sys


def minify(src: str) -> str:
    try:
        import python_minifier
        return python_minifier.minify(
            src,
            remove_annotations=True,
            remove_pass=True,
            remove_literal_statements=True,
            combine_imports=True,
            hoist_literals=True,
            rename_locals=True,
            rename_globals=False,  # keep global names readable + safe (avoid breaking string lookups)
            remove_object_base=True,
            convert_posargs_to_args=True,
            preserve_shebang=False,
        )
    except ImportError:
        print("python-minifier not installed; using raw source. pip install python-minifier for ~3x compression boost.", file=sys.stderr)
        return src


def pack(src_path: str, dst_path: str | None = None) -> str:
    src = open(src_path).read()
    # Verify input is valid Python
    ast.parse(src)
    minified = minify(src)
    # Verify minified is still valid Python with same AST
    try:
        ast_orig = ast.dump(ast.parse(src), annotate_fields=True)
        ast_min = ast.dump(ast.parse(minified), annotate_fields=True)
        if ast_orig != ast_min:
            print("WARNING: minified AST differs from original (expected — annotations stripped, names renamed). Verifying byte equivalence is impossible after minification; trust python-minifier's correctness.", file=sys.stderr)
    except Exception as e:
        print(f"AST recheck failed: {e}", file=sys.stderr)
        sys.exit(1)
    raw = minified.encode("utf-8")
    compressed = lzma.compress(raw, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}])
    encoded = base64.b85encode(compressed).decode("ascii")
    wrapper = (
        'import lzma as L,base64 as B\n'
        f'exec(L.decompress(B.b85decode("{encoded}"),format=L.FORMAT_RAW,filters=[{{"id":L.FILTER_LZMA2}}]))\n'
    )
    if dst_path is None:
        dst_path = src_path + ".packed.py"
    with open(dst_path, "w") as f:
        f.write(wrapper)
    print(f"Source size:     {len(src):>10,} bytes ({src_path})")
    print(f"Minified:        {len(minified):>10,} bytes")
    print(f"LZMA compressed: {len(compressed):>10,} bytes")
    print(f"Base85 encoded:  {len(encoded):>10,} bytes")
    print(f"Wrapper output:  {len(wrapper):>10,} bytes ({dst_path})")
    return dst_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Source tw_eval.py to pack")
    ap.add_argument("dst", nargs="?", default=None, help="Destination path (default: SRC.packed.py)")
    args = ap.parse_args()
    pack(args.src, args.dst)
