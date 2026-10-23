"""Walk-forward : apport du modele de draft v2 au-dessus de la force d'equipe (BT) et du TrueSkill roster.

    python ml/eval_draft.py
Sort : logloss/accuracy de chaque combinaison + poids du stack final.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
from strength import fit_bt, home_groups, INTL, MAJOR, GROUP        # noqa: E402
import draftmodel as dm                                              # noqa: E402

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
BT = dict(halflife=180, lam=2.0, sg=2.0)
HL_D = 240; LAM_D = 6.0


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    sf = pd.read_parquet(PROC / "seq_feats.parquet"); sf["date"] = pd.to_datetime(sf.date)
    dr = pd.read_parquet(PROC / "drafts_team.parquet")
    g = sf.merge(dr[["gameid"] + [f"{s}_{r}" for s in ("blue", "red") for r in dm.ROLES]], on="gameid").sort_values("date").reset_index(drop=True)
    home = home_groups(g)
    ok_lg = g.league.isin(set(GROUP) | INTL)
    start = pd.Timestamp("2025-03-01")
    months = pd.period_range(start, g.date.max(), freq="M")
    bt = np.full(len(g), np.nan); dl = np.full(len(g), np.nan)
    for m in months:
        a, b = m.start_time, (m + 1).start_time
        te = np.where((g.date >= a) & (g.date < b))[0]
        if not len(te):
            continue
        rating, side, _ = fit_bt(g, a, home, **BT)
        sub = g.iloc[te]
        bt[te] = np.array([rating(x) - rating(z) for x, z in zip(sub.blue, sub.red)]) + side
        tr = g[(g.date < a) & (g.date >= a - pd.Timedelta(days=900)) & ok_lg]
        age = (a - tr.date).dt.total_seconds().to_numpy() / 86400
        mod = dm.fit(*dm.to_lists(tr), tr.y.to_numpy(), 0.5 ** (age / HL_D), LAM_D)
        dl[te] = dm.logit(mod, *dm.to_lists(sub))
        print(m, len(te), flush=True)
    g["bt"] = bt; g["draft"] = dl
    g.to_parquet(PROC / "wf_preds.parquet", index=False)
    tier = g.blue.map(lambda t: home.get(t, "?") in MAJOR) & g.red.map(lambda t: home.get(t, "?") in MAJOR)
    T = g[tier & g.bt.notna()].copy(); y = T.y.to_numpy()
    sig = lambda z: 1 / (1 + np.exp(-z))
    print(f"\ntier-1 test games: {len(T)}")
    for name, p in [("Elo seq", sig(T.elo_logit)), ("BT", sig(T.bt)), ("draft seul", sig(T.draft + 0.1))]:
        print(f"{name:12s} ll={ll(p, y):.4f} acc={((p > .5) == y).mean():.3f}")
    # stack walk-forward (poids appris sur les mois precedents)
    feats = {"BT": ["bt"], "BT+TS": ["bt", "ts_logit"], "BT+draft": ["bt", "draft"], "BT+TS+draft": ["bt", "ts_logit", "draft"]}
    T["ym"] = T.date.dt.to_period("M")
    mm = sorted(T.ym.unique())[6:]
    for name, cols in feats.items():
        ps, ys = [], []
        for m in mm:
            tr = T[T.ym < m]; te = T[T.ym == m]
            lr = LogisticRegression(C=1e3).fit(tr[cols], tr.y)
            ps.append(lr.predict_proba(te[cols])[:, 1]); ys.append(te.y.to_numpy())
        p = np.concatenate(ps); yy = np.concatenate(ys)
        print(f"stack {name:12s} ll={ll(p, yy):.4f} acc={((p > .5) == yy).mean():.3f} (n={len(yy)})")
    lr = LogisticRegression(C=1e3).fit(T[["bt", "ts_logit", "draft"]], T.y)
    print("poids stack final [bt, ts, draft]:", lr.coef_[0].round(3), "b0", lr.intercept_.round(3))


if __name__ == "__main__":
    main()
