"""Isolated MiniLM inference process.

PyTorch and the saved XGBoost/LightGBM native libraries conflict on macOS when
they perform inference in one process. A small JSON-lines worker keeps the
transformer runtime isolated without changing any model output.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    model_path = Path(sys.argv[1])
    if not model_path.is_dir():
        return 2
    encoder = SentenceTransformer(
        str(model_path),
        trust_remote_code=False,
        device="cpu",
        local_files_only=True,
    )
    print(json.dumps({"status": "ready"}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            url = request["url"]
            if not isinstance(url, str) or len(url) > 2_048:
                raise ValueError("invalid URL")
            matrix = encoder.encode(
                [url],
                batch_size=1,
                device="cpu",
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                precision="float32",
            )
            print(json.dumps({"embedding": matrix[0].tolist()}), flush=True)
        except Exception:
            print(json.dumps({"error": "MiniLM inference failed."}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
