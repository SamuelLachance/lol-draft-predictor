"""Walk-forward : compare l'Elo sequentiel (ancien) au Bradley-Terry a decalage de ligue (nouveau).

Pour chaque mois M, on ajuste sur les games AVANT M puis on predit les games de M (aucune fuite).
    python ml/eval_strength.py            # grille d'hyperparametres + comparaison
"""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from strength import fit_bt, home_groups, INTL, MAJOR, group_of   # noqa: E402

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def load():
    g = pd.read_parquet(PROC / "seq_feats.parquet")
    g["date"] = pd.to_datetime(g.date); return g.sort_values("date").reset_index(drop=True)


def walk(g, home, start="2025-03-01", **kw):
    months = pd.period_range(start, g.date.max(), freq="M")
    out = np.full(len(g), np.nan)
    for m in months:
        a, b = m.start_time, (m + 1).start_time
        idx = np.where((g.date >= a) & (g.date < b))[0]
        if not len(idx):
            continue
        rating, side, _ = fit_bt(g, a, home, **kw)
        sub = g.iloc[idx]
        d = np.array([rating(x) - rating(z) for x, z in zip(sub.blue, sub.red)]) + side
        out[idx] = 1 / (1 + np.exp(-d))
    return out


def main():
    g = load(); home = home_groups(g)
    tier = g.blue.map(lambda t: home.get(t, "?") in MAJOR) & g.red.map(lambda t: home.get(t, "?") in MAJOR)
    intl = g.league.isin(INTL) & tier
    test = (g.date >= "2025-03-01") & tier
    y = g.y.to_numpy()
    pe = 1 / (1 + np.exp(-g.elo_logit.to_numpy()))
    print(f"test tier-1 games {int(test.sum())} | international {int((test & intl).sum())}")
    print(f"ELO seq (ancien)   all ll={ll(pe[test], y[test]):.4f} acc={((pe[test] > .5) == y[test]).mean():.3f} | "
          f"intl ll={ll(pe[test & intl], y[test & intl]):.4f} acc={((pe[test & intl] > .5) == y[test & intl]).mean():.3f}")
    res = []
    for hl, lam, sg in itertools.product([180, 365, 730], [0.5, 2.0, 6.0], [2.0, 4.0]):
        p = walk(g, home, halflife=hl, lam=lam, sg=sg)
        ok = test & ~np.isnan(p)
        io = ok & intl
        res.append((ll(p[ok], y[ok]), ll(p[io], y[io]), ((p[ok] > .5) == y[ok]).mean(), ((p[io] > .5) == y[io]).mean(), hl, lam, sg))
        print(f"BT hl={hl} lam={lam} sg={sg}: all ll={res[-1][0]:.4f} acc={res[-1][2]:.3f} | intl ll={res[-1][1]:.4f} acc={res[-1][3]:.3f}", flush=True)
    best = min(res, key=lambda r: r[0] + r[1])
    print("meilleur (all+intl) :", best)


if __name__ == "__main__":
    main()
