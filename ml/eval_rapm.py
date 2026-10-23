"""Gain des approches RAPM / embeddings dans le stack (walk-forward mensuel, memes mois que le stack de production)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as st
warnings.filterwarnings("ignore")
PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
F = pd.read_parquet(PROC / "v4_feats.parquet").merge(pd.read_parquet(PROC / "v4_pbt.parquet"), on="gameid").merge(pd.read_parquet(PROC / "v4_rapm.parquet"), on="gameid")
F["date"] = pd.to_datetime(F.date)
T = F[F.radj_champ.notna() & F.pbt.notna() & F.bt.notna()].copy(); T["ym"] = T.date.dt.to_period("M")
def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
PRE = ["bt", "pbt", "ts", "d_fw", "h2h"]; POST = PRE + ["draft", "comfort", "comfort_min"]
specs = {"PRE (reference avant draft)": PRE, "PRE + radj_player": PRE + ["radj_player"], "PRE - pbt + radj_player": ["bt", "ts", "d_fw", "h2h", "radj_player"],
         "POST (reference apres draft)": POST, "POST + radj_champ": POST + ["radj_champ"], "POST - draft + radj_champ": [c for c in POST if c != "draft"] + ["radj_champ"],
         "POST + emb": POST + ["emb"], "POST + radj_champ + emb": POST + ["radj_champ", "emb"], "POST + radj_champ + radj_player + emb": POST + ["radj_champ", "radj_player", "emb"]}
res = {}
for name, cols in specs.items():
    ps, ys = [], []
    for m in [x for x in sorted(T.ym.unique()) if x >= pd.Period("2024-01", "M")]:
        tr, te = T[(T.ym < m) & (T.date >= "2023-06-01")], T[T.ym == m]
        age = (te.date.min() - tr.date).dt.total_seconds().to_numpy() / 86400
        mu = tr[cols].mean(); sd = tr[cols].std().replace(0, 1)
        lr = LogisticRegression(C=10, max_iter=500).fit(((tr[cols] - mu) / sd).fillna(0), tr.y, sample_weight=0.5 ** (age / 900))
        ps.append(lr.predict_proba(((te[cols] - mu) / sd).fillna(0))[:, 1]); ys.append(te.y.to_numpy())
    res[name] = (np.concatenate(ps), np.concatenate(ys))
rng = np.random.default_rng(4)
for base in ("PRE (reference avant draft)", "POST (reference apres draft)"):
    p0, y0 = res[base]; ix = rng.integers(0, len(y0), (400, len(y0)))
    print(f"\n--- n={len(y0)} games (2024-01 -> 2026-10) ; reference: {base}  ll={ll(p0, y0):.4f} acc={((p0 > .5) == y0).mean():.3f}")
    for name, (p, y) in res.items():
        if (base.startswith("PRE") != name.startswith("PRE")): continue
        bs = np.array([ll(p[i], y[i]) - ll(p0[i], y0[i]) for i in ix]); lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"   {name:42s} ll {ll(p, y):.4f} acc {((p > .5) == y).mean():.3f} diff {ll(p, y) - ll(p0, y0):+.4f} IC95% [{lo:+.4f}, {hi:+.4f}]{'  <-- mieux' if hi < 0 else ('  <-- pire' if lo > 0 else '')}")
# standalone : pouvoir predictif du seul morceau draft (au-dela de bt)
for c in ("draft", "radj_champ", "emb", "radj_player", "comfort"):
    ps, ys = [], []
    for m in [x for x in sorted(T.ym.unique()) if x >= pd.Period("2024-01", "M")]:
        tr, te = T[(T.ym < m) & (T.date >= "2023-06-01")], T[T.ym == m]
        lr = LogisticRegression(C=10).fit(tr[["bt", c]].fillna(0), tr.y); ps.append(lr.predict_proba(te[["bt", c]].fillna(0))[:, 1]); ys.append(te.y.to_numpy())
    p, y = np.concatenate(ps), np.concatenate(ys); lr = LogisticRegression(C=10).fit(T[["bt", c]].fillna(0), T.y)
    print(f"bt + {c:12s}: ll {ll(p, y):.4f} acc {((p > .5) == y).mean():.3f}  (coef {c} = {lr.coef_[0][1]:+.3f})")
