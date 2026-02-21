#!/usr/bin/env python3
"""
Inspect structure of a single .pt embedding file (e.g. chemprop-embeddings.pt, chemeleon-embeddings.pt).
Writes human-readable structure to an output text file.
"""
import argparse
import sys
from pathlib import Path


def _describe(obj, depth=0, max_str=80):
    indent = "  " * depth
    if depth > 5:
        return indent + "...(max depth)\n"
    t = type(obj).__name__
    if hasattr(obj, "shape"):
        return f"{indent}{t} shape={obj.shape}\n"
    if isinstance(obj, dict):
        lines = [f"{indent}dict with {len(obj)} keys:\n"]
        for i, (k, v) in enumerate(obj.items()):
            if i >= 20:
                lines.append(f"{indent}  ... and {len(obj) - 20} more keys\n")
                break
            lines.append(f"{indent}  [{repr(k)}]: ")
            lines.append(_describe(v, depth + 2, max_str))
        return "".join(lines)
    if isinstance(obj, (list, tuple)):
        lines = [f"{indent}{t} len={len(obj)}\n"]
        for i, item in enumerate(obj):
            if i >= 10:
                lines.append(f"{indent}  ... and {len(obj) - 10} more\n")
                break
            lines.append(_describe(item, depth + 1, max_str))
        return "".join(lines)
    s = repr(obj)
    if len(s) > max_str:
        s = s[: max_str - 3] + "..."
    return f"{indent}{t} {s}\n"


def main():
    parser = argparse.ArgumentParser(description="Inspect .pt embedding file structure")
    parser.add_argument("pt_path", type=str, help="Path to .pt file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output text file (default: stdout)")
    args = parser.parse_args()
    pt_path = Path(args.pt_path)
    if not pt_path.exists():
        print(f"Error: file not found: {pt_path}", file=sys.stderr)
        sys.exit(1)
    try:
        import torch
    except ImportError:
        print("Error: PyTorch required", file=sys.stderr)
        sys.exit(1)
    try:
        data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading .pt: {e}", file=sys.stderr)
        sys.exit(1)
    out = _describe(data)
    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
        print(f"Wrote structure to {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
