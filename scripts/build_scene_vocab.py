"""One-shot generator for bundled SigLIP 2 photography scene/motion vocabulary embeddings (PLAN T11).

Run once whenever `SCENE_VOCAB` in `lightbox.models.embedder` changes:
    uv run python scripts/build_scene_vocab.py

Outputs `src/lightbox/models/data/scene_vocab.npy` (float32, L2-normalized, shape `(N, 768)`),
distributed inside the Python package so scan-time scene tagging never loads the text tower.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lightbox.models.embedder import (
    EMBEDDING_DIM,
    SCENE_VOCAB,
    SCENE_VOCAB_NPY_PATH,
    Siglip2Embedder,
)


def build_scene_vocab(output_path: Path = SCENE_VOCAB_NPY_PATH) -> Path:
    embedder = Siglip2Embedder()
    prompts = [entry.prompt for entry in SCENE_VOCAB]
    vectors = embedder.embed_texts(prompts)
    if vectors.shape != (len(SCENE_VOCAB), EMBEDDING_DIM):
        raise RuntimeError(
            f"Unexpected vocab embedding shape: {vectors.shape} "
            f"(expected {(len(SCENE_VOCAB), EMBEDDING_DIM)})."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, vectors.astype(np.float32, copy=False))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate bundled SigLIP 2 scene/motion vocabulary embeddings (.npy)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCENE_VOCAB_NPY_PATH,
        help=f"Output .npy file path (default: {SCENE_VOCAB_NPY_PATH}).",
    )
    args = parser.parse_args()
    out = build_scene_vocab(args.output)
    size_kb = out.stat().st_size / 1024.0
    print(f"Wrote {len(SCENE_VOCAB)} x {EMBEDDING_DIM} scene vocab embeddings -> {out} ({size_kb:.1f} KiB)")


if __name__ == "__main__":
    main()
