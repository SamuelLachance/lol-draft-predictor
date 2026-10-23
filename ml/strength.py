"""Force d'equipe inter-ligues : Bradley-Terry ridge a decalage par ligue, pondere par recence.

rating(equipe) = offset(ligue d'origine) + ecart(equipe)

Les decalages de ligue ne sont identifies que par les games inter-ligues (Worlds, MSI, FST, EWC...),
les ecarts d'equipe par toutes les games. Une equipe d'une ligue inferieure ne peut donc plus
depasser une equipe de ligue majeure juste parce qu'elle domine des adversaires faibles.
"""
from __future__ import annotations
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

INTL = {"WLDs", "MSI", "FST", "EWC", "RR", "IWCQ", "AG", "APAC"}
# ligue -> groupe de calibration (ligues successives fusionnees ; le reste = son propre groupe)
GROUP = {"LCK": "KR", "KeSPA": "KR", "LPL": "CN", "LEC": "EU", "EU LCS": "EU",
         "LCS": "NA", "NA LCS": "NA", "LTA N": "NA", "LTA": "NA",
         "CBLOL": "BR", "LTA S": "BR", "LCP": "LCP", "PCS": "PCS", "VCS": "VCS", "LJL": "LJL",
         "TCL": "TR", "LLA": "LLA"}
# groupes qui envoient des equipes a Worlds/MSI = ligues "majeures" affichees dans les classements
MAJOR = {"KR", "CN", "EU", "NA", "BR", "LCP"}
REGION = {"KR": "KR", "CN": "CN", "EU": "EMEA", "NA": "AMERICAS", "BR": "AMERICAS", "LCP": "APAC"}


# noms alternatifs d'une meme organisation (Leaguepedia vs Oracle's Elixir) -> nom d'affichage canonique
ALIASES = {"LYON (2024 American Team)": "LYON"}


def alias(name):
    return ALIASES.get(name, name)


def group_of(lg: str) -> str:
    return GROUP.get(lg, lg)


def home_groups(g: pd.DataFrame, since: str = "2024-06-01") -> dict[str, str]:
    """groupe d'origine = groupe modal des games DOMESTIQUES recentes (fallback : toutes dates)."""
    rows = pd.concat([g[["blue", "league", "date"]].rename(columns={"blue": "t"}),
                      g[["red", "league", "date"]].rename(columns={"red": "t"})])
    dom = rows[~rows.league.isin(INTL)].copy()
    dom["grp"] = dom.league.map(group_of)
    out = {}
    for sub in (dom[dom.date >= since], dom):
        for t, s in sub.groupby("t").grp:
            out.setdefault(t, s.mode().iloc[0])
    return out


def fit_bt(g: pd.DataFrame, asof, home: dict[str, str], halflife: float = 365, lam: float = 1.0,
           sg: float = 3.0, window: int = 1100, min_games: int = 1):
    """Ajuste sur les games AVANT `asof`. Retourne (rating(team)->float en logit, side_bias, offsets)."""
    asof = pd.Timestamp(asof)
    h = g[(g.date < asof) & (g.date >= asof - pd.Timedelta(days=window))]
    if not len(h):
        return (lambda t: 0.0), 0.0, {}
    teams = sorted(set(h.blue) | set(h.red)); groups = sorted({home.get(t, "?") for t in teams})
    ti = {t: i for i, t in enumerate(teams)}; gi = {x: i for i, x in enumerate(groups)}
    n = len(h); nt = len(teams)
    b = h.blue.map(ti).to_numpy(); r = h.red.map(ti).to_numpy()
    gb = h.blue.map(lambda t: gi[home.get(t, "?")]).to_numpy(); gr = h.red.map(lambda t: gi[home.get(t, "?")]).to_numpy()
    ar = np.arange(n)
    rows = np.r_[ar, ar, ar, ar]; cols = np.r_[b, r, nt + gb, nt + gr]
    vals = np.r_[np.ones(n), -np.ones(n), sg * np.ones(n), -sg * np.ones(n)]
    X = sp.csr_matrix((vals, (rows, cols)), shape=(n, nt + len(groups)))
    age = (asof - h.date).dt.total_seconds().to_numpy() / 86400.0
    w = 0.5 ** (age / halflife)
    m = LogisticRegression(C=1.0 / lam, fit_intercept=True, max_iter=300)
    m.fit(X, h.y.to_numpy(), sample_weight=w)
    coef = m.coef_[0]; dev = dict(zip(teams, coef[:nt])); off = {x: float(coef[nt + i] * sg) for x, i in gi.items()}
    side = float(m.intercept_[0])

    def rating(t):
        return off.get(home.get(t, "?"), 0.0) + dev.get(t, 0.0)
    return rating, side, off


def to_elo(logit: float) -> float:
    """logit -> echelle Elo (400/ln10), centre sur 1500."""
    return 1500 + logit * 400 / np.log(10)
