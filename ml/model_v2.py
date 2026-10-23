"""Modele v4 : stack logistique de signaux complementaires, valide en walk-forward (aucune fuite).

    z = w0 + w_bt*bt + w_pbt*pbt + w_ts*ts + w_fw*d_fw + w_h2h*h2h [+ w_draft*draft + w_c*comfort + w_cm*comfort_min]
    p(blue gagne) = sigmoid(z)

 bt      Bradley-Terry d'equipe a decalage de ligue (strength.fit_bt)            \\ refits mensuels AVANT chaque mois
 pbt     Bradley-Terry au niveau joueur (somme des 5 joueurs, strength.fit_pbt)  /  (aucune fuite)
 ts      TrueSkill des 5 joueurs (features.build)
 d_fw    forme ponderee (10 derniers) bleu - rouge ; h2h : confrontations directes 2 ans
 draft   valeur de draft (champion x role, matchups, synergies) ; comfort : maitrise joueur-champion
Les poids sont appris sur les predictions walk-forward des matchs des ligues majeures depuis 2023-06 (decroissance
temporelle, demi-vie 900 j). Les variables testees mais ecartees (faute de gain) : meta glissante, attributs de
composition, Elo sequentiel, repos/charge, serie, TrueSkill par role, LightGBM sur ~60 variables (ml/eval_v4.py).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import strength as st
import draftmodel as dm

BT = dict(halflife=180, lam=2.0, sg=2.0)         # grille walk-forward (ml/eval_strength.py)
PBT = dict(halflife=365, lam=8.0)                # grille walk-forward (ml/exp_pbt.py)
HL_D, LAM_D, WIN_D = 240, 6.0, 900               # draft : demi-vie, ridge, fenetre (jours)
BT_FROM = "2021-01-01"                           # bt walk-forward pour toutes les games depuis cette date
START = "2023-01-01"                              # debut des refits walk-forward (bt, pbt, draft)
STACK_FROM = "2023-06-01"; HL_STACK = 900.0
EVAL_FROM = pd.Period("2025-09", "M")             # mois de test du stack (walk-forward)
PRE = ["bt", "pbt", "ts", "d_fw", "h2h"]
POST = PRE + ["draft", "comfort", "comfort_min"]


def sig(z):
    return 1 / (1 + np.exp(-z))


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_draft(g: pd.DataFrame, asof) -> dict:
    ok = g.league.isin(set(st.GROUP) | st.INTL)
    tr = g[(g.date < asof) & (g.date >= asof - pd.Timedelta(days=WIN_D)) & ok]
    age = (asof - tr.date).dt.total_seconds().to_numpy() / 86400
    return dm.fit(*dm.to_lists(tr), tr.y.to_numpy(), 0.5 ** (age / HL_D), LAM_D)


def tier_mask(g, home):
    """matchs OFFICIELS des ligues majeures + evenements internationaux, entre equipes de ligues majeures (population d'entrainement/test du stack)."""
    return (g.blue.map(lambda t: home.get(t, "?") in st.MAJOR) & g.red.map(lambda t: home.get(t, "?") in st.MAJOR)
            & g.league.isin(set(st.GROUP) | st.INTL))


def walk_forward(G: pd.DataFrame, home: dict) -> pd.DataFrame:
    """G : gameid,date,league,blue,red,y,bp,rp,has_pl + picks (blue_/red_<role>) + ts,d_fw,h2h,comfort,comfort_min,elo.
    Ajoute bt, pbt (avec leur biais de side) et draft, calcules AVANT chaque mois, pour les matchs des ligues majeures."""
    G = G.sort_values("date").reset_index(drop=True)
    tier = tier_mask(G, home).to_numpy()
    bt = np.full(len(G), np.nan); pb = np.full(len(G), np.nan); dl = np.full(len(G), np.nan)
    for m in pd.period_range(BT_FROM, G.date.max(), freq="M"):
        a, b = m.start_time, (m + 1).start_time
        ta = np.where((G.date >= a) & (G.date < b))[0]
        if not len(ta):
            continue
        suba = G.iloc[ta]
        rating, side, _ = st.fit_bt(G, a, home, **BT)
        bt[ta] = np.array([rating(x) - rating(z) for x, z in zip(suba.blue, suba.red)]) + side     # toutes les games (controle du RAPM)
        te = np.where((G.date >= a) & (G.date < b) & tier & (G.date >= START))[0]
        if not len(te):
            continue
        sub = G.iloc[te]
        pr, td, off, pside = st.fit_pbt(G, a, home, **PBT)
        pb[te] = np.array([st.pbt_diff(b_, r_, bp, rp, pr, td, off, home) for b_, r_, bp, rp in zip(sub.blue, sub.red, sub.bp, sub.rp)]) + pside
        dl[te] = dm.logit(fit_draft(G, a), *dm.to_lists(sub))
        print("  walk-forward", m, len(te), flush=True)
    G["bt"] = bt; G["pbt"] = pb; G["draft"] = dl
    return G


def _fit_std(T, cols):
    """LR sur variables standardisees -> coefficients en echelle BRUTE (+ moyennes pour imputer les valeurs manquantes)."""
    mu = T[cols].mean(); sd = T[cols].std().replace(0, 1)
    X = ((T[cols] - mu) / sd).fillna(0)
    age = (T.date.max() - T.date).dt.total_seconds().to_numpy() / 86400
    lr = LogisticRegression(C=10.0, max_iter=500).fit(X, T.y, sample_weight=0.5 ** (age / HL_STACK))
    w = lr.coef_[0] / sd.to_numpy(); b0 = float(lr.intercept_[0] - (lr.coef_[0] * mu.to_numpy() / sd.to_numpy()).sum())
    return {"w0": b0, **{c: float(v) for c, v in zip(cols, w)}, "fill": {c: float(mu[c]) for c in cols}}


def apply(wt: dict, df: pd.DataFrame) -> np.ndarray:
    """probabilite p(blue) pour un DataFrame de variables (NaN -> moyenne d'entrainement)."""
    z = np.full(len(df), wt["w0"], dtype=float)
    for c, fill in wt["fill"].items():
        z += wt[c] * df[c].fillna(fill).to_numpy()
    return sig(z)


def stack(WF: pd.DataFrame, home: dict):
    """poids du stack (pre-draft / post-draft) + metriques honnetes (walk-forward du stack lui-meme)."""
    T = WF[tier_mask(WF, home) & WF.bt.notna() & WF.pbt.notna() & (WF.date >= STACK_FROM)].copy()
    T["ym"] = T.date.dt.to_period("M"); intl = T.league.isin(st.INTL)
    out = {}
    variants = {"bt": ["bt"], "v3 (bt+ts+draft)": ["bt", "ts", "draft"], "pre-draft": PRE, "post-draft": POST}
    for name, cols in variants.items():
        ps, ys, iv = [], [], []
        for m in [x for x in sorted(T.ym.unique()) if x >= EVAL_FROM]:
            tr, te = T[T.ym < m], T[T.ym == m]
            wt = _fit_std(tr, cols); ps.append(apply(wt, te)); ys.append(te.y.to_numpy()); iv.append(te.league.isin(st.INTL).to_numpy())
        p, y, i = np.concatenate(ps), np.concatenate(ys), np.concatenate(iv)
        out[name] = {"logloss": round(ll(p, y), 4), "acc": round(float(((p > .5) == y).mean()), 3), "n": int(len(y)),
                     "acc_intl": round(float(((p[i] > .5) == y[i]).mean()), 3) if i.any() else None, "n_intl": int(i.sum())}
    ev = T[T.ym >= EVAL_FROM]; pe = sig(ev.elo.to_numpy()); yb = ev.y.to_numpy()
    out["elo_seq_ancien"] = {"logloss": round(ll(pe, yb), 4), "acc": round(float(((pe > .5) == yb).mean()), 3), "n": int(len(yb))}
    return {"pre": _fit_std(T, PRE), "post": _fit_std(T, POST)}, out
