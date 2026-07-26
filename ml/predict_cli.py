"""CLI quick check: python ml/predict_cli.py Ahri ..."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROLES = ["top", "jng", "mid", "bot", "sup"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("blue", nargs=5, help="Blue top jng mid bot sup")
    parser.add_argument("red", nargs=5, help="Red top jng mid bot sup")
    args = parser.parse_args()

    bundle = joblib.load(ROOT / "models" / "draft_model.joblib")
    model = bundle["model"]
    cal = bundle["calibrator"]
    le = bundle["label_encoder"]
    champ_wr = bundle["champ_wr"]
    matchups = bundle["matchups"]
    synergies = bundle["synergies"]

    from train import heuristic_matrix, multi_hot_matrix
    import pandas as pd

    row = {f"blue_{r}": args.blue[i] for i, r in enumerate(ROLES)}
    row.update({f"red_{r}": args.red[i] for i, r in enumerate(ROLES)})
    df = pd.DataFrame([row])
    X = np.hstack(
        [
            multi_hot_matrix(df, le),
            heuristic_matrix(df, champ_wr, matchups, synergies),
        ]
    )
    raw = model.predict_proba(X)[:, 1]
    proba = cal.predict_proba(raw.reshape(-1, 1))[:, 1][0]
    print(json.dumps({"blue_win_prob": proba, "red_win_prob": 1 - proba}, indent=2))


if __name__ == "__main__":
    main()
