"""Standalone E1 deterministic engine evaluation runner & CI gate (SPEC §11.2)."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from lightbox.services.synth import (
    E1_MAX_EXEMPT_MISLABEL,
    E1_MAX_RED_FPR,
    E1_MIN_REJECT_RECALL,
    E1EvalReport,
    run_e1_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Lightbox E1 synthetic detection CI gate.")
    parser.add_argument("--sensitivity", choices=("conservative", "standard"), default="conservative")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="lightbox_e1_") as tmp:
        report = run_e1_evaluation(
            Path(tmp),
            sensitivity=args.sensitivity,
            seed=args.seed,
            max_workers=args.workers,
        )

    payload = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")

    return 0 if report.passed else 1


__all__ = [
    "E1EvalReport",
    "E1_MAX_EXEMPT_MISLABEL",
    "E1_MAX_RED_FPR",
    "E1_MIN_REJECT_RECALL",
    "main",
    "run_e1_evaluation",
]

if __name__ == "__main__":
    raise SystemExit(main())
