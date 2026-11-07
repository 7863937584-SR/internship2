from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_PATH
from src.data import ensure_sample_data, load_sample_tickets
from src.training import train_and_save


def main() -> None:
    ensure_sample_data()
    df = load_sample_tickets()
    artifact = train_and_save(df)
    print(f"Trained {artifact['trained_rows']} rows")
    print(f"Saved model to {MODEL_PATH}")
    for target, metrics in artifact.get("metrics", {}).items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            print(f"{target}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
