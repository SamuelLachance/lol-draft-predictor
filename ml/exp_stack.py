"""Selection de variables du stack final (LR regularisee, walk-forward mensuel, entrainee 2023+)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as st
warnings.filterwarnings("ignore")
PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
F = pd.read_parquet(PROC / "v4_feats.parquet").merge(pd.read_parquet(PROC / "v4_pbt.parquet"), on="gameid"); F["date"] = pd.to_datetime(F.date)
T = F[F.home_b.isin(st.MAJOR) & F.home_r.isin(st.MAJOR) & F.pbt.notna() & F.bt.notna()].copy(); T["ym"] = T.date.dt.to_period("M")
def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
def run(cols, C=1e3, hl=600):
    ps, ys = [], []
    for m in [x for x in sorted(T.ym.unique()) if x >= pd.Period("2025-09", "M")]:
        tr, te = T[(T.ym < m) & (T.date >= "2023-06-01")], T[T.ym == m]
        age = (te.date.min() - tr.date).dt.total_seconds().to_numpy() / 86400
        mu = tr[cols].mean(); sd = tr[cols].std().replace(0, 1)
        X = ((tr[cols] - mu) / sd).fillna(0); Xt = ((te[cols] - mu) / sd).fillna(0)
        lr = LogisticRegression(C=C).fit(X, tr.y, sample_weight=0.5 ** (age / hl))
        ps.append(lr.predict_proba(Xt)[:, 1]); ys.append(te.y.to_numpy())
    p, y = np.concatenate(ps), np.concatenate(ys); return ll(p, y), ((p > .5) == y).mean()
base = ["bt", "pbt", "ts", "draft", "d_fw", "h2h"]
cands = {"d_f10": ["d_f10"], "d_f5": ["d_f5"], "d_streak": ["d_streak"], "d_rest": ["d_rest"], "d_load7": ["d_load7"], "elo": ["elo"], "ts_sig": ["ts_sig"], "ts_min/max": ["ts_min", "ts_max"],
         "meta_wr": ["meta_wr"], "meta_pres": ["meta_pres"], "prof": ["prof"], "comfort": ["comfort", "comfort_min"], "playoffs": ["playoffs"], "intl": ["intl"],
         "comp_attrs": ["c_engage", "c_mobility", "c_tankiness", "c_poke", "c_scaling", "c_imm", "c_front", "c_ad"], "ts_roles": [f"ts_{r}" for r in ["top", "jng", "mid", "bot", "sup"]]}
b = run(base); print("base", base, f"ll={b[0]:.4f} acc={b[1]:.3f}")
keep = list(base); best = b
for name, cols in cands.items():
    r = run(keep + cols); d = r[0] - best[0]
    print(f"+{name:12s} ll={r[0]:.4f} acc={r[1]:.3f} d={d:+.4f}")
    if d < -0.0007: keep += cols; best = r
print("RETENU:", keep, f"ll={best[0]:.4f} acc={best[1]:.3f}")
for C in (0.05, 0.2, 1, 10):
    r = run(keep, C=C); print("C", C, f"ll={r[0]:.4f} acc={r[1]:.3f}")
for hl in (300, 900, 1500): r = run(keep, hl=hl); print("hl", hl, f"ll={r[0]:.4f} acc={r[1]:.3f}")
