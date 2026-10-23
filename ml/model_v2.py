"""Modele v2 : force d'equipe inter-ligues (BT) + TrueSkill roster + draft, valide en walk-forward.

    p(blue gagne) = sigmoid( w0 + w_bt * bt + w_ts * ts + w_draft * draft )

- bt     : Bradley-Terry a decalage de ligue (strength.py), ajuste AVANT chaque mois teste
- ts     : logit TrueSkill des rosters (calcule de facon causale par build_board)
- draft  : avantage de draft (draftmodel.py), ajuste AVANT chaque mois teste
Les poids du stack sont appris sur les predictions walk-forward (jamais sur du in-sample).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import strength as st
import draftmodel as dm

BT = dict(halflife=180, lam=2.0, sg=2.0)       # choisis par ml/eval_strength.py (grille walk-forward)
HL_D, LAM_D, WIN_D = 240, 6.0, 900             # draft : demi-vie, ridge, fenetre (jours)
START = "2025-03-01"


def sig(z):
    return 1 / (1 + np.exp(-z))


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_draft(g: pd.DataFrame, asof) -> dict:
    ok = g.league.isin(set(st.GROUP) | st.INTL)
    tr = g[(g.date < asof) & (g.date >= asof - pd.Timedelta(days=WIN_D)) & ok]
    age = (asof - tr.date).dt.total_seconds().to_numpy() / 86400
    return dm.fit(*dm.to_lists(tr), tr.y.to_numpy(), 0.5 ** (age / HL_D), LAM_D)


def walk_forward(g: pd.DataFrame, home: dict) -> pd.DataFrame:
    """g : gameid,date,league,blue,red,y,elo_logit,ts_logit + blue_/red_ roles. Retourne g + bt, draft."""
    g = g.sort_values("date").reset_index(drop=True)
    bt = np.full(len(g), np.nan); dl = np.full(len(g), np.nan)
    for m in pd.period_range(START, g.date.max(), freq="M"):
        a, b = m.start_time, (m + 1).start_time
        te = np.where((g.date >= a) & (g.date < b))[0]
        if not len(te):
            continue
        rating, side, _ = st.fit_bt(g, a, home, **BT)
        sub = g.iloc[te]
        bt[te] = np.array([rating(x) - rating(z) for x, z in zip(sub.blue, sub.red)]) + side
        dl[te] = dm.logit(fit_draft(g, a), *dm.to_lists(sub))
    g["bt"] = bt; g["draft"] = dl
    return g


def stack(wf: pd.DataFrame, home: dict):
    """poids du stack + metriques honnetes (walk-forward du stack lui-meme)."""
    tier = wf.blue.map(lambda t: home.get(t, "?") in st.MAJOR) & wf.red.map(lambda t: home.get(t, "?") in st.MAJOR)
    T = wf[tier & wf.bt.notna()].copy(); T["ym"] = T.date.dt.to_period("M")
    intl = T.league.isin(st.INTL).to_numpy()
    out = {}
    for name, cols in {"bt": ["bt"], "bt+ts": ["bt", "ts_logit"], "bt+ts+draft": ["bt", "ts_logit", "draft"]}.items():
        ps, ys, iv = [], [], []
        for m in sorted(T.ym.unique())[6:]:
            tr, te = T[T.ym < m], T[T.ym == m]
            lr = LogisticRegression(C=1e3).fit(tr[cols], tr.y)
            ps.append(lr.predict_proba(te[cols])[:, 1]); ys.append(te.y.to_numpy()); iv.append(te.league.isin(st.INTL).to_numpy())
        p, y, i = np.concatenate(ps), np.concatenate(ys), np.concatenate(iv)
        out[name] = {"logloss": round(ll(p, y), 4), "acc": round(float(((p > .5) == y).mean()), 3), "n": int(len(y)),
                     "acc_intl": round(float(((p[i] > .5) == y[i]).mean()), 3) if i.any() else None, "n_intl": int(i.sum())}
    base = T.iloc[[k for k in range(len(T)) if T.ym.iloc[k] >= sorted(T.ym.unique())[6]]]
    pe = sig(base.elo_logit.to_numpy()); yb = base.y.to_numpy()
    out["elo_seq_ancien"] = {"logloss": round(ll(pe, yb), 4), "acc": round(float(((pe > .5) == yb).mean()), 3), "n": int(len(yb))}
    w = {}
    for name, cols in {"pre": ["bt", "ts_logit"], "post": ["bt", "ts_logit", "draft"]}.items():
        lr = LogisticRegression(C=1e3).fit(T[cols], T.y)
        w[name] = {"w0": float(lr.intercept_[0]), **{c: float(v) for c, v in zip(cols, lr.coef_[0])}}
    return w, out
