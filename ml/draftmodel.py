"""Modele de draft v2 : ridge logistique creux sur champion-role, matchups de lane et synergies.

Chaque feature est antisymetrique (bleu = +1, rouge = -1) : le modele apprend UN avantage de draft,
sans biais de side (porte par l'intercept). Pondere par recence (la meta change vite).
"""
from __future__ import annotations
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

ROLES = ["top", "jng", "mid", "bot", "sup"]
SYN = [("top", "jng"), ("jng", "mid"), ("mid", "bot"), ("bot", "sup"), ("jng", "sup"), ("mid", "sup"), ("jng", "bot")]


def game_keys(row_blue: list[str], row_red: list[str]):
    """liste (cle, signe) des features d'une draft : blue/red = 5 champions en ordre de roles."""
    out = []
    for r, a, b in zip(ROLES, row_blue, row_red):
        out.append((f"c|{r}|{a}", 1.0)); out.append((f"c|{r}|{b}", -1.0))
        lo, hi = (a, b) if a <= b else (b, a)
        out.append((f"m|{r}|{lo}|{hi}", 1.0 if a == lo else -1.0))
    bi = dict(zip(ROLES, row_blue)); ri = dict(zip(ROLES, row_red))
    for r1, r2 in SYN:
        out.append((f"s|{r1}|{r2}|{bi[r1]}|{bi[r2]}", 1.0)); out.append((f"s|{r1}|{r2}|{ri[r1]}|{ri[r2]}", -1.0))
    return out


SCALE = {"c": 1.0, "m": 0.55, "s": 0.55}      # penalite plus forte sur matchups/synergies (donnees rares)


def design(rows_blue, rows_red, vocab: dict | None = None, grow: bool = True):
    vocab = {} if vocab is None else vocab
    ri, ci, vi = [], [], []
    for i, (b, r) in enumerate(zip(rows_blue, rows_red)):
        for k, s in game_keys(b, r):
            j = vocab.get(k)
            if j is None:
                if not grow:
                    continue
                j = vocab[k] = len(vocab)
            ri.append(i); ci.append(j); vi.append(s * SCALE[k[0]])
    return ri, ci, vi, vocab


def fit(blue: list, red: list, y: np.ndarray, w: np.ndarray, lam: float = 6.0):
    ri, ci, vi, vocab = design(blue, red)
    X = sp.csr_matrix((vi, (ri, ci)), shape=(len(blue), len(vocab)))
    m = LogisticRegression(C=1.0 / lam, max_iter=400)
    m.fit(X, y, sample_weight=w)
    coef = m.coef_[0]
    return {"vocab": vocab, "coef": coef, "b0": float(m.intercept_[0])}


def logit(model, blue: list, red: list) -> np.ndarray:
    ri, ci, vi, _ = design(blue, red, model["vocab"], grow=False)
    X = sp.csr_matrix((vi, (ri, ci)), shape=(len(blue), len(model["vocab"])))
    return X @ model["coef"]


def to_lists(df: pd.DataFrame):
    b = df[[f"blue_{r}" for r in ROLES]].astype(str).values.tolist()
    r = df[[f"red_{r}" for r in ROLES]].astype(str).values.tolist()
    return b, r
