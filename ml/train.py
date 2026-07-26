"""Train draft-win models on all historical pro 5v5 drafts.

Uses a multi-hot composition model (strong baseline for draft-only) plus
recency weighting so recent metas matter more while still training on
the full 2014→present history.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
WEB_MODEL = ROOT / "web" / "public" / "model"

ROLES = ["top", "jng", "mid", "bot", "sup"]
PICK_COLS = [f"blue_{r}" for r in ROLES] + [f"red_{r}" for r in ROLES]


def recency_weights(years: np.ndarray) -> np.ndarray:
    years = np.asarray(years, dtype=float)
    years = np.where(np.isnan(years), np.nanmin(years), years)
    # Exponential emphasis on recent years (still keeps old data).
    age = years.max() - years
    return np.exp(-0.35 * age).astype(np.float32)


def multi_hot_matrix(df: pd.DataFrame, le: LabelEncoder) -> np.ndarray:
    """Blue multi-hot | Red multi-hot | Blue-Red diff | role indices."""
    n = len(df)
    k = len(le.classes_)
    X = np.zeros((n, k * 3 + 10), dtype=np.float32)

    rows = np.arange(n)
    for i, role in enumerate(ROLES):
        b = le.transform(df[f"blue_{role}"].astype(str))
        r = le.transform(df[f"red_{role}"].astype(str))
        X[rows, b] = 1.0
        X[rows, k + r] = 1.0
        X[rows, 2 * k + b] += 1.0
        X[rows, 2 * k + r] -= 1.0
        X[rows, 3 * k + i] = b
        X[rows, 3 * k + 5 + i] = r
    return X


def feature_names(le: LabelEncoder) -> list[str]:
    champs = [str(c) for c in le.classes_]
    names = [f"blue::{c}" for c in champs]
    names += [f"red::{c}" for c in champs]
    names += [f"diff::{c}" for c in champs]
    names += [f"idx_blue_{r}" for r in ROLES]
    names += [f"idx_red_{r}" for r in ROLES]
    return names


def champ_strength_table(train: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
    """Weighted Bayesian win rate per champion (side-agnostic)."""
    wins: dict[str, float] = {}
    ns: dict[str, float] = {}
    for i, row in enumerate(train.itertuples(index=False)):
        w = float(weights[i])
        y = int(row.blue_win)
        for role in ROLES:
            b = getattr(row, f"blue_{role}")
            r = getattr(row, f"red_{role}")
            wins[b] = wins.get(b, 0.0) + y * w
            ns[b] = ns.get(b, 0.0) + w
            wins[r] = wins.get(r, 0.0) + (1 - y) * w
            ns[r] = ns.get(r, 0.0) + w
    prior_n = 40.0
    return {
        c: (wins[c] + 0.5 * prior_n) / (ns[c] + prior_n)
        for c in ns
    }


def matchup_table(train: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
    wins: dict[str, float] = {}
    ns: dict[str, float] = {}
    for i, row in enumerate(train.itertuples(index=False)):
        w = float(weights[i])
        y = int(row.blue_win)
        for role in ROLES:
            key = f"{role}::{getattr(row, f'blue_{role}')}||{getattr(row, f'red_{role}')}"
            wins[key] = wins.get(key, 0.0) + y * w
            ns[key] = ns.get(key, 0.0) + w
    prior_n = 25.0
    return {
        k: (wins[k] + 0.5 * prior_n) / (ns[k] + prior_n)
        for k, n in ns.items()
        if n >= 8
    }


def synergy_table(train: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
    wins: dict[str, float] = {}
    ns: dict[str, float] = {}
    for i, row in enumerate(train.itertuples(index=False)):
        w = float(weights[i])
        y = int(row.blue_win)
        blue = [getattr(row, f"blue_{r}") for r in ROLES]
        red = [getattr(row, f"red_{r}") for r in ROLES]
        for a in range(5):
            for b in range(a + 1, 5):
                bk = "||".join(sorted([blue[a], blue[b]]))
                rk = "||".join(sorted([red[a], red[b]]))
                wins[bk] = wins.get(bk, 0.0) + y * w
                ns[bk] = ns.get(bk, 0.0) + w
                wins[rk] = wins.get(rk, 0.0) + (1 - y) * w
                ns[rk] = ns.get(rk, 0.0) + w
    prior_n = 25.0
    return {
        k: (wins[k] + 0.5 * prior_n) / (ns[k] + prior_n)
        for k, n in ns.items()
        if n >= 10
    }


def score_draft_heuristic(
    blue: list[str],
    red: list[str],
    champ_wr: dict[str, float],
    matchups: dict[str, float],
    synergies: dict[str, float],
) -> dict[str, float]:
    b_wr = np.mean([champ_wr.get(c, 0.5) for c in blue])
    r_wr = np.mean([champ_wr.get(c, 0.5) for c in red])
    mu = np.mean(
        [
            matchups.get(f"{role}::{blue[i]}||{red[i]}", 0.5)
            for i, role in enumerate(ROLES)
        ]
    )
    def syn(team: list[str]) -> float:
        vals = []
        for a in range(5):
            for b in range(a + 1, 5):
                vals.append(synergies.get("||".join(sorted([team[a], team[b]])), 0.5))
        return float(np.mean(vals)) if vals else 0.5

    b_syn, r_syn = syn(blue), syn(red)
    # Combine into a logit-ish score then sigmoid externally via calibrator inputs
    return {
        "wr_diff": float(b_wr - r_wr),
        "matchup": float(mu - 0.5),
        "synergy_diff": float(b_syn - r_syn),
        "blue_avg_wr": float(b_wr),
        "red_avg_wr": float(r_wr),
    }


def heuristic_matrix(
    df: pd.DataFrame,
    champ_wr: dict[str, float],
    matchups: dict[str, float],
    synergies: dict[str, float],
) -> np.ndarray:
    rows = []
    for row in df.itertuples(index=False):
        blue = [getattr(row, f"blue_{r}") for r in ROLES]
        red = [getattr(row, f"red_{r}") for r in ROLES]
        h = score_draft_heuristic(blue, red, champ_wr, matchups, synergies)
        rows.append(
            [
                h["wr_diff"],
                h["matchup"],
                h["synergy_diff"],
                h["blue_avg_wr"],
                h["red_avg_wr"],
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    drafts_path = PROCESSED / "drafts.parquet"
    if not drafts_path.exists():
        raise SystemExit("Run build_drafts.py first")

    print("Loading drafts...")
    df = pd.read_parquet(drafts_path)
    df = df.dropna(subset=PICK_COLS + ["blue_win"]).copy()
    df["blue_win"] = df["blue_win"].astype(int)
    for col in PICK_COLS:
        df[col] = df[col].astype(str)

    if "date" in df.columns and df["date"].notna().any():
        df["_date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("_date")
        cut = int(len(df) * 0.85)
        train_df = df.iloc[:cut].copy()
        test_df = df.iloc[cut:].copy()
    else:
        train_df, test_df = train_test_split(
            df, test_size=0.15, random_state=42, stratify=df["blue_win"]
        )

    train_df, cal_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df["blue_win"]
    )

    le = LabelEncoder()
    le.fit(sorted(pd.unique(df[PICK_COLS].values.ravel())))

    years_train = (
        pd.to_numeric(train_df["year"], errors="coerce").to_numpy()
        if "year" in train_df.columns
        else np.full(len(train_df), 2020.0)
    )
    w_train = recency_weights(years_train)

    print("Building tables...")
    champ_wr = champ_strength_table(train_df, w_train)
    matchups = matchup_table(train_df, w_train)
    synergies = synergy_table(train_df, w_train)

    print("Building feature matrices...")
    X_train = np.hstack(
        [
            multi_hot_matrix(train_df, le),
            heuristic_matrix(train_df, champ_wr, matchups, synergies),
        ]
    )
    X_cal = np.hstack(
        [
            multi_hot_matrix(cal_df, le),
            heuristic_matrix(cal_df, champ_wr, matchups, synergies),
        ]
    )
    X_test = np.hstack(
        [
            multi_hot_matrix(test_df, le),
            heuristic_matrix(test_df, champ_wr, matchups, synergies),
        ]
    )
    y_train = train_df["blue_win"].to_numpy()
    y_cal = cal_df["blue_win"].to_numpy()
    y_test = test_df["blue_win"].to_numpy()

    names = feature_names(le) + [
        "wr_diff",
        "matchup",
        "synergy_diff",
        "blue_avg_wr",
        "red_avg_wr",
    ]

    print("Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.5,
        min_child_samples=50,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_X=X_cal,
        eval_y=y_cal,
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(80, verbose=True)],
    )

    raw_cal = model.predict_proba(X_cal)[:, 1]
    raw_test = model.predict_proba(X_test)[:, 1]

    # Probability calibration
    calibrator = LogisticRegression(C=1e6, solver="lbfgs")
    calibrator.fit(raw_cal.reshape(-1, 1), y_cal)
    proba = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    pred = (proba >= 0.5).astype(int)

    conf = np.abs(proba - 0.5)
    high = conf >= 0.07
    baseline = float(max(y_test.mean(), 1 - y_test.mean()))

    metrics = {
        "n_train": int(len(train_df)),
        "n_cal": int(len(cal_df)),
        "n_test": int(len(test_df)),
        "n_games_total": int(len(df)),
        "n_champions": int(len(le.classes_)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "baseline_accuracy": baseline,
        "lift_vs_baseline": float(accuracy_score(y_test, pred) - baseline),
        "accuracy_confident": float(accuracy_score(y_test[high], pred[high]))
        if high.any()
        else None,
        "confident_coverage": float(high.mean()),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "blue_base_rate_test": float(y_test.mean()),
        "feature_names": names,
        "best_iteration": int(
            getattr(model, "best_iteration_", None) or model.n_estimators
        ),
    }

    MODELS.mkdir(parents=True, exist_ok=True)
    WEB_MODEL.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "calibrator": calibrator,
            "label_encoder": le,
            "champ_wr": champ_wr,
            "matchups": matchups,
            "synergies": synergies,
            "roles": ROLES,
        },
        MODELS / "draft_model.joblib",
    )

    # Keep only useful synergies/matchups for the browser bundle size
    web_bundle = {
        "version": 3,
        "mode": "lgbm_multihot",
        "metrics": {
            "accuracy": metrics["accuracy"],
            "baseline_accuracy": metrics["baseline_accuracy"],
            "accuracy_confident": metrics["accuracy_confident"],
            "confident_coverage": metrics["confident_coverage"],
            "roc_auc": metrics["roc_auc"],
            "n_games_total": metrics["n_games_total"],
            "n_test": metrics["n_test"],
            "n_champions": metrics["n_champions"],
            "best_iteration": metrics["best_iteration"],
        },
        "champions": [str(c) for c in le.classes_],
        "champion_index": {str(c): int(i) for i, c in enumerate(le.classes_)},
        "roles": ROLES,
        "feature_names": names,
        "stats": {
            "champ_wr": champ_wr,
            "matchup_wr": matchups,
            "synergy_wr": synergies,
        },
        "lgbm_model": model.booster_.dump_model(),
        "best_iteration": metrics["best_iteration"],
        "calibrator": {
            "coef": float(calibrator.coef_.ravel()[0]),
            "intercept": float(calibrator.intercept_.ravel()[0]),
        },
        "default_year": float(
            pd.to_numeric(train_df["year"], errors="coerce").dropna().max() or 2025
        ),
        "default_patch": str(train_df["patch"].dropna().astype(str).iloc[-1])
        if "patch" in train_df.columns and train_df["patch"].notna().any()
        else "15.1",
    }

    with open(MODELS / "draft_model.json", "w", encoding="utf-8") as f:
        json.dump(web_bundle, f)
    with open(WEB_MODEL / "draft_model.json", "w", encoding="utf-8") as f:
        json.dump(web_bundle, f)
    with open(MODELS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: v for k, v in metrics.items() if k != "feature_names"}, indent=2))
    size_mb = (WEB_MODEL / "draft_model.json").stat().st_size / 1e6
    print(f"Saved web bundle ({size_mb:.1f} MB) -> {WEB_MODEL / 'draft_model.json'}")


if __name__ == "__main__":
    main()
