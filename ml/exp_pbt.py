"""Test : Bradley-Terry au niveau JOUEUR (somme des niveaux des 5 joueurs + decalage de ligue + petit ecart d'equipe)."""
from __future__ import annotations
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as st, exp_v4 as e4

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


def fit_pbt(G, asof, home, halflife=180, lam=4.0, sp_=1.0, st_=0.5, sg=2.0, window=1100):
    asof = pd.Timestamp(asof)
    h = G[(G.date < asof) & (G.date >= asof - pd.Timedelta(days=window)) & G.has_pl]
    teams = sorted(set(h.blue) | set(h.red)); groups = sorted({home.get(t, "?") for t in teams})
    pl = sorted({p for l in h.bp for p in l} | {p for l in h.rp for p in l})
    ti = {t: i for i, t in enumerate(teams)}; gi = {x: i for i, x in enumerate(groups)}; pi = {p: i for i, p in enumerate(pl)}
    nt, ng, npl = len(teams), len(groups), len(pl); n = len(h)
    r, c, v = [], [], []
    for i, (b, rr, bp, rp) in enumerate(zip(h.blue, h.red, h.bp, h.rp)):
        r += [i, i]; c += [ti[b], ti[rr]]; v += [st_, -st_]
        r += [i, i]; c += [nt + gi[home.get(b, "?")], nt + gi[home.get(rr, "?")]]; v += [sg, -sg]
        for p in bp: r.append(i); c.append(nt + ng + pi[p]); v.append(sp_)
        for p in rp: r.append(i); c.append(nt + ng + pi[p]); v.append(-sp_)
    X = sp.csr_matrix((v, (r, c)), shape=(n, nt + ng + npl))
    age = (asof - h.date).dt.total_seconds().to_numpy() / 86400; w = 0.5 ** (age / halflife)
    m = LogisticRegression(C=1.0 / lam, max_iter=400).fit(X, h.y.to_numpy(), sample_weight=w)
    co = m.coef_[0]
    pr = {p: co[nt + ng + i] * sp_ for p, i in pi.items()}; tdev = {t: co[i] * st_ for t, i in ti.items()}
    off = {x: co[nt + i] * sg for x, i in gi.items()}
    return pr, tdev, off, float(m.intercept_[0])


def pred(sub, pr, tdev, off, side, home):
    out = []
    for b, rr, bp, rp in zip(sub.blue, sub.red, sub.bp, sub.rp):
        sb = off.get(home.get(b, "?"), 0) + tdev.get(b, 0) + sum(pr.get(p, 0) for p in bp)
        sr = off.get(home.get(rr, "?"), 0) + tdev.get(rr, 0) + sum(pr.get(p, 0) for p in rp)
        out.append(sb - sr + side)
    return np.array(out)


def main():
    dt, pg = e4.load()
    F = pd.read_parquet(PROC / "v4_feats.parquet"); F["date"] = pd.to_datetime(F.date)
    F = F.merge(dt[["gameid"]], on="gameid")
    F["bp"] = F.gameid.map(lambda g: pg.get(g, {}).get("bp")); F["rp"] = F.gameid.map(lambda g: pg.get(g, {}).get("rp"))
    F["has_pl"] = F.bp.map(lambda l: bool(l) and all(l)) & F.rp.map(lambda l: bool(l) and all(l))
    F = F[F.league.isin(set(st.GROUP) | st.INTL)].sort_values("date").reset_index(drop=True)
    home = st.home_groups(F)
    tier = F.home_b.isin(st.MAJOR) & F.home_r.isin(st.MAJOR) & F.has_pl
    y = F.y.to_numpy()
    def ll(p, yy): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())
    months = list(pd.period_range("2025-09", F.date.max(), freq="M"))
    base_ok = tier & F.bt.notna()
    for hl, lam, sp_, st_ in [(180, 4, 1.0, 0.5), (180, 8, 1.0, 0.5), (365, 8, 1.0, 0.5), (180, 4, 1.0, 1.0), (270, 6, 1.0, 0.7)]:
        pb = np.full(len(F), np.nan)
        for m in months:
            a, b = m.start_time, (m + 1).start_time
            te = np.where((F.date >= a) & (F.date < b) & tier)[0]
            if not len(te): continue
            pr, td, off, side = fit_pbt(F, a, home, halflife=hl, lam=lam, sp_=sp_, st_=st_)
            pb[te] = 1 / (1 + np.exp(-pred(F.iloc[te], pr, td, off, side, home)))
        ok = ~np.isnan(pb)
        pbt = 1 / (1 + np.exp(-F.bt.to_numpy()))
        print(f"pbt hl={hl} lam={lam} sp={sp_} st={st_}: ll={ll(pb[ok], y[ok]):.4f} acc={((pb[ok] > .5) == y[ok]).mean():.3f} | team-BT same games ll={ll(pbt[ok], y[ok]):.4f} acc={((pbt[ok] > .5) == y[ok]).mean():.3f} | avg ll={ll((pb[ok] + pbt[ok]) / 2, y[ok]):.4f} n={int(ok.sum())}", flush=True)


if __name__ == "__main__":
    main()
