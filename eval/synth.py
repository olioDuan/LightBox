"""Standalone entrypoint for synthetic demo & E1 dataset generation (SPEC §11.1)."""

from __future__ import annotations

import argparse
from pathlib import Path

from lightbox.services.synth import (
    SynthSampleSpec,
    build_demo_dataset,
    build_e1_dataset,
    restore_dataset_write_bits,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Lightbox synthetic demo or E1 dataset.")
    parser.add_argument("output_dir", type=Path, help="Destination directory for synthetic JPEGs")
    parser.add_argument("--kind", choices=("demo", "e1"), default="demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.kind == "demo":
        specs = build_demo_dataset(args.output_dir, seed=args.seed)
    else:
        specs = build_e1_dataset(args.output_dir, seed=args.seed)
    print(f"Generated {len(specs)} synthetic {args.kind} photos in {args.output_dir}")
    return 0


__all__ = [
    "SynthSampleSpec",
    "build_demo_dataset",
    "build_e1_dataset",
    "main",
    "restore_dataset_write_bits",
]

if __name__ == "__main__":
    raise SystemExit(main())
