"""Etape 2 du modele v4 : gradient boosting en walk-forward mensuel sur les variables causales.

    python ml/eval_v4.py
Compare : BT seul, stack v3 (bt+ts+draft), LR riche, LightGBM pre-draft, LightGBM post-draft.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as st
warnings.filterwarnings("ignore")
PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
ROLES = ["top", "jng", "mid", "bot", "sup"]

TEAM = ["bt", "elo", "d_f5", "d_f10", "d_fw", "d_streak", "d_rest", "d_load7", "b_rest", "r_rest", "b_ngames", "r_ngames", "h2h", "h2h_n", "playoffs", "intl"]
PLAYERS = ["ts", "ts_max", "ts_min", "ts_sig", "ts_gp_b", "ts_gp_r"] + [f"ts_{r}" for r in ROLES]
PRE = TEAM + PLAYERS
DRAFT = ["draft", "meta_wr", "meta_pres", "prof", "comfort", "comfort_min"] + [c for c in ["c_cc", "c_engage", "c_disengage", "c_mobility", "c_waveclear", "c_poke", "c_scaling", "c_tankiness", "c_sustain", "c_burst", "c_global", "c_anti_dive", "c_pick", "c_ad", "c_ap", "c_imm", "c_front", "c_ad_bal_b", "c_ad_bal_r"]]
POST = PRE + DRAFT

PARAMS = dict(objective="binary", num_leaves=8, learning_rate=0.03, n_estimators=350, min_child_samples=120, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=8.0, verbose=-1, n_jobs=4)


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def metrics(p, y):
    return f"ll={ll(p, y):.4f} acc={((p > .5) == y).mean():.3f}"


def weights(dates, asof, hl=500.0):
    age = (asof - dates).dt.total_seconds().to_numpy() / 86400
    return 0.5 ** (age / hl)


def main():
    F = pd.read_parquet(PROC / "v4_feats.parquet")
    F["date"] = pd.to_datetime(F.date)
    F = F[F.bt.notna() & F.league.isin(set(st.GROUP) | st.INTL)].sort_values("date").reset_index(drop=True)
    tier = F.home_b.isin(st.MAJOR) & F.home_r.isin(st.MAJOR)
    start = pd.Period("2025-09", "M")
    months = [m for m in pd.period_range(start, F.date.max(), freq="M")]
    out = {k: np.full(len(F), np.nan) for k in ["bt", "lr_pre", "gbm_pre", "gbm_post", "v3"]}
    imp = {}
    for m in months:
        a, b = m.start_time, (m + 1).start_time
        te = np.where((F.date >= a) & (F.date < b) & tier)[0]
        if not len(te):
            continue
        tr = F[(F.date < a) & (F.date >= "2021-06-01")]
        w = weights(tr.date, a); y = tr.y.to_numpy()
        out["bt"][te] = 1 / (1 + np.exp(-F.bt.iloc[te].to_numpy()))
        # v3 : bt + ts + draft (LR)
        v3 = LogisticRegression(C=1e3).fit(tr[["bt", "ts", "draft"]].fillna(0), y, sample_weight=w)
        out["v3"][te] = v3.predict_proba(F[["bt", "ts", "draft"]].iloc[te].fillna(0))[:, 1]
        lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.05, max_iter=500))
        lr.fit(tr[PRE], y, logisticregression__sample_weight=w)
        out["lr_pre"][te] = lr.predict_proba(F[PRE].iloc[te])[:, 1]
        g1 = lgb.LGBMClassifier(**PARAMS).fit(tr[PRE], y, sample_weight=w)
        out["gbm_pre"][te] = g1.predict_proba(F[PRE].iloc[te])[:, 1]
        g2 = lgb.LGBMClassifier(**PARAMS).fit(tr[POST], y, sample_weight=w)
        out["gbm_post"][te] = g2.predict_proba(F[POST].iloc[te])[:, 1]
        for f, v in zip(POST, g2.booster_.feature_importance("gain")): imp[f] = imp.get(f, 0) + v
        print(m, len(te), flush=True)
    ok = ~np.isnan(out["gbm_post"]); y = F.y.to_numpy()
    intl = F.intl.to_numpy() == 1
    print(f"\ntest tier-1 games: {int(ok.sum())} (international {int((ok & intl).sum())})")
    for k in ["bt", "v3", "lr_pre", "gbm_pre", "gbm_post"]:
        p = out[k]; print(f"{k:9s} {metrics(p[ok], y[ok])} | intl {metrics(p[ok & intl], y[ok & intl])}")
    # blends
    for name, p in [("avg(gbm_post,v3)", (out["gbm_post"] + out["v3"]) / 2), ("avg(gbm_pre,lr_pre)", (out["gbm_pre"] + out["lr_pre"]) / 2),
                    ("avg(gbm_post,lr_pre)", (out["gbm_post"] + out["lr_pre"]) / 2)]:
        print(f"{name:20s} {metrics(p[ok], y[ok])} | intl {metrics(p[ok & intl], y[ok & intl])}")
    tot = sum(imp.values())
    print("importance (gain %):", sorted([(k, round(100 * v / tot, 1)) for k, v in imp.items()], key=lambda kv: -kv[1])[:18])
    F["p_gbm_post"] = out["gbm_post"]; F["p_gbm_pre"] = out["gbm_pre"]; F["p_v3"] = out["v3"]; F["p_lr_pre"] = out["lr_pre"]
    F[["gameid", "date", "p_gbm_post", "p_gbm_pre", "p_v3", "p_lr_pre", "y"]].to_parquet(PROC / "v4_eval_preds.parquet", index=False)


if __name__ == "__main__":
    main()
