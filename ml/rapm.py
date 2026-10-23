"""RAPM des champions : effet NET d'un champion sur la probabilite de victoire, controle pour la force d'equipe (bt),
le niveau de chacun des 10 joueurs et la maitrise (courbe d'apprentissage : games deja jouees par ce joueur sur ce champion).

    logit P(bleu) = a*bt + sum_joueurs(+-niveau) + sum_champions(+-[effet champion x role + beta_c * log(1 + maitrise)])

Ridge logistique creux. Sert a la "valeur cachee" : ecart entre l'effet ajuste et ce que montrent le winrate brut / la presence.
"""
from __future__ import annotations
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

ROLES = ["top", "jng", "mid", "bot", "sup"]
SCALE = {"bt": 3.0, "p": 0.8, "c": 1.0, "m": 0.6}
LAM = 6.0


def mastery(bp, rp, bc, rc, has):
    """log(1 + nb de games JOUEES par ce joueur sur ce champion AVANT la game) pour chaque slot (donnees triees par date)."""
    cnt = {}; mb = np.zeros((len(bc), 5)); mr = np.zeros((len(bc), 5))
    for i in range(len(bc)):
        if not has[i]: continue
        for k in range(5):
            mb[i, k] = cnt.get((bp[i][k], bc[i, k]), 0); mr[i, k] = cnt.get((rp[i][k], rc[i, k]), 0)
        for k in range(5):
            cnt[(bp[i][k], bc[i, k])] = cnt.get((bp[i][k], bc[i, k]), 0) + 1; cnt[(rp[i][k], rc[i, k])] = cnt.get((rp[i][k], rc[i, k]), 0) + 1
    return np.log1p(mb), np.log1p(mr)


def fit(bt, y, dates, bp, rp, bc, rc, mb, mr, idx, lam=LAM, hl=365.0):
    vocab = {}; r, c, v = [], [], []
    def col(key):
        j = vocab.get(key)
        if j is None: j = vocab[key] = len(vocab) + 1
        return j
    for row, i in enumerate(idx):
        r.append(row); c.append(0); v.append(bt[i] * SCALE["bt"])
        for k, role in enumerate(ROLES):
            for sgn, pl, ch, m in ((1.0, bp[i][k], bc[i, k], mb[i, k]), (-1.0, rp[i][k], rc[i, k], mr[i, k])):
                r += [row, row, row]; c += [col(("p", pl)), col(("c", role, ch)), col(("m", ch))]
                v += [sgn * SCALE["p"], sgn * SCALE["c"], sgn * m * SCALE["m"]]
    X = sp.csr_matrix((v, (r, c)), shape=(len(idx), len(vocab) + 1))
    age = (dates[idx[-1]] - dates[idx]).astype("timedelta64[s]").astype(float) / 86400
    lr = LogisticRegression(C=1.0 / lam, max_iter=300).fit(X, y[idx], sample_weight=0.5 ** (age / hl))
    return vocab, lr.coef_[0]


def champion_table(vocab, coef) -> pd.DataFrame:
    """effet champion x role (logit, a maitrise nulle) et pente de maitrise par champion (logit par unite de log(1+games))."""
    rows = [{"role": k[1], "champion": k[2], "adj": float(coef[j] * SCALE["c"])} for k, j in vocab.items() if k[0] == "c"]
    mast = {k[1]: float(coef[j] * SCALE["m"]) for k, j in vocab.items() if k[0] == "m"}
    df = pd.DataFrame(rows); df["learn"] = df.champion.map(mast); return df
