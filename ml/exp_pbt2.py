"""Stack complet en walk-forward : bt + pbt (joueurs) + ts + draft, entraine sur 2023+. Sauvegarde les predictions pbt."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as st, exp_v4 as e4
from exp_pbt import fit_pbt, pred
PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
HL, LAM = 365, 8


def ll(p, yy): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())


def main():
    dt, pg = e4.load()
    F = pd.read_parquet(PROC / "v4_feats.parquet"); F["date"] = pd.to_datetime(F.date)
    F["bp"] = F.gameid.map(lambda g: pg.get(g, {}).get("bp")); F["rp"] = F.gameid.map(lambda g: pg.get(g, {}).get("rp"))
    F["has_pl"] = F.bp.map(lambda l: bool(l) and all(l)) & F.rp.map(lambda l: bool(l) and all(l))
    F = F[F.league.isin(set(st.GROUP) | st.INTL)].sort_values("date").reset_index(drop=True)
    home = st.home_groups(F)
    tier = F.home_b.isin(st.MAJOR) & F.home_r.isin(st.MAJOR)
    pb = np.full(len(F), np.nan)
    for m in pd.period_range("2023-01", F.date.max(), freq="M"):
        a, b = m.start_time, (m + 1).start_time
        te = np.where((F.date >= a) & (F.date < b) & tier)[0]
        if not len(te): continue
        pr, td, off, side = fit_pbt(F, a, home, halflife=HL, lam=LAM)
        pb[te] = pred(F.iloc[te], pr, td, off, side, home)
        print(m, len(te), flush=True)
    F["pbt"] = pb
    F[["gameid", "pbt"]].to_parquet(PROC / "v4_pbt.parquet", index=False)
    T = F[tier & F.pbt.notna() & F.bt.notna()].copy(); T["ym"] = T.date.dt.to_period("M"); y = T.y.to_numpy()
    sets = {"bt": ["bt"], "bt+ts+draft (v3)": ["bt", "ts", "draft"], "bt+pbt": ["bt", "pbt"], "bt+pbt+ts": ["bt", "pbt", "ts"], "bt+pbt+ts+draft": ["bt", "pbt", "ts", "draft"],
            "bt+pbt+ts+draft+form+h2h": ["bt", "pbt", "ts", "draft", "d_fw", "h2h"]}
    for name, cols in sets.items():
        ps, ys, iv = [], [], []
        for m in [x for x in sorted(T.ym.unique()) if x >= pd.Period("2025-09", "M")]:
            tr, te = T[(T.ym < m) & (T.date >= "2023-06-01")], T[T.ym == m]
            age = (te.date.min() - tr.date).dt.total_seconds().to_numpy() / 86400
            lr = LogisticRegression(C=1e3).fit(tr[cols].fillna(0), tr.y, sample_weight=0.5 ** (age / 600))
            ps.append(lr.predict_proba(te[cols].fillna(0))[:, 1]); ys.append(te.y.to_numpy()); iv.append(te.intl.to_numpy() == 1)
        p, yy, i = np.concatenate(ps), np.concatenate(ys), np.concatenate(iv)
        print(f"{name:28s} ll={ll(p, yy):.4f} acc={((p > .5) == yy).mean():.3f} (n={len(yy)}) | intl ll={ll(p[i], yy[i]):.4f} acc={((p[i] > .5) == yy[i]).mean():.3f}")


if __name__ == "__main__":
    main()
