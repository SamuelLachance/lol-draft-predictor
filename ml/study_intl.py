"""Les matchs INTERNATIONAUX sont-ils plus incertains (favoris surestimes) ? Test dans le stack de production (walk-forward)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as st
warnings.filterwarnings("ignore")
PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
F = pd.read_parquet(PROC / "v4_feats.parquet").merge(pd.read_parquet(PROC / "v4_pbt.parquet"), on="gameid"); F["date"] = pd.to_datetime(F.date)
T = F[F.home_b.isin(st.MAJOR) & F.home_r.isin(st.MAJOR) & F.pbt.notna() & F.bt.notna() & F.league.isin(set(st.GROUP) | st.INTL)].copy()
T["ym"] = T.date.dt.to_period("M"); T["W"] = (T.league == "WLDs").astype(float); T["I"] = T.intl
for c in ("bt", "pbt", "ts"): T[c + "I"] = T[c] * T.I; T[c + "W"] = T[c] * T.W
def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
PRE = ["bt", "pbt", "ts", "d_fw", "h2h"]
specs = {"v4 avant draft": PRE, "+ ecrasement intl (bt,pbt,ts x intl)": PRE + ["btI", "pbtI", "tsI"], "+ ecrasement intl (bt seul)": PRE + ["btI"],
         "+ ecrasement Worlds (bt,pbt,ts x Worlds)": PRE + ["btW", "pbtW", "tsW"], "+ intl (drapeau)": PRE + ["I"]}
months = [m for m in sorted(T.ym.unique()) if m >= pd.Period("2024-01", "M")]
out = {}
for name, cols in specs.items():
    ps, ys, ii, ww = [], [], [], []
    for m in months:
        tr, te = T[(T.ym < m) & (T.date >= "2023-06-01")], T[T.ym == m]
        age = (te.date.min() - tr.date).dt.total_seconds().to_numpy() / 86400
        mu = tr[cols].mean(); sd = tr[cols].std().replace(0, 1)
        lr = LogisticRegression(C=10, max_iter=500).fit(((tr[cols] - mu) / sd).fillna(0), tr.y, sample_weight=0.5 ** (age / 900))
        ps.append(lr.predict_proba(((te[cols] - mu) / sd).fillna(0))[:, 1]); ys.append(te.y.to_numpy()); ii.append(te.I.to_numpy() == 1); ww.append(te.W.to_numpy() == 1)
    out[name] = (np.concatenate(ps), np.concatenate(ys), np.concatenate(ii), np.concatenate(ww))
p0, y0, i0, w0 = out["v4 avant draft"]; rng = np.random.default_rng(2)
for lab, mask in (("matchs internationaux", i0), ("Worlds (main event)", w0), ("ligues domestiques", ~i0)):
    idx0 = np.where(mask)[0]; ix = rng.choice(idx0, (400, len(idx0)))
    print(f"\n[{lab}] n={len(idx0)}")
    for name, (p, y, _, _) in out.items():
        d = ll(p[mask], y[mask]) - ll(p0[mask], y0[mask]); bs = np.array([ll(p[i], y[i]) - ll(p0[i], y0[i]) for i in ix])
        flag = "  <-- mieux" if np.percentile(bs, 97.5) < 0 else ("  <-- pire" if np.percentile(bs, 2.5) > 0 else "")
        print(f"   {name:42s} ll {ll(p[mask], y[mask]):.4f} acc {((p[mask] > .5) == y[mask]).mean():.3f} diff {d:+.4f} IC95% [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]{flag}")
# coefficient d'ecrasement
cols = PRE + ["btI", "pbtI", "tsI"]; tr = T[T.date >= "2023-06-01"]; mu = tr[cols].mean(); sd = tr[cols].std().replace(0, 1)
lr = LogisticRegression(C=10, max_iter=500).fit(((tr[cols] - mu) / sd).fillna(0), tr.y); raw = lr.coef_[0] / sd.to_numpy()
print("\ncoefficients bruts :", {c: round(v, 3) for c, v in zip(cols, raw)})
print("=> pente intl relative bt: ", round((raw[0] + raw[5]) / raw[0], 3), "| pbt:", round((raw[1] + raw[6]) / raw[1], 3), "| ts:", round((raw[2] + raw[7]) / raw[2], 3))
