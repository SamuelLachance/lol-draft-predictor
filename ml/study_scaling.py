"""Etude : "aux Worlds le scaling gagne plus souvent" — verification historique (2014-2026) puis apport au modele.

    python ml/study_scaling.py bt        # 1) force d'equipe walk-forward sur l'historique LP (cache data/processed/hist_bt.parquet)
    python ml/study_scaling.py analyze   # 2) analyses de la theorie
Definition du "scaling" d'un champion : coefficient delta_c = effet de la DUREE de la game sur sa probabilite de victoire
(regression logistique ridge sur les games domestiques, a force d'equipe egale, offset bt). delta_c > 0 : meilleur en game longue.
Pour eviter toute fuite : delta est estime sur les games DOMESTIQUES uniquement (ou hors annee testee), jamais sur les games evaluees.
"""
from __future__ import annotations
import json, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as st
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; PROC = ROOT / "data" / "processed"; HIST = ROOT / "data" / "games_hist.parquet"
ROLES = ["top", "jng", "mid", "bot", "sup"]


def league_code(t: str) -> str:
    t = str(t)
    if re.search(r"Play-?In", t, re.I) and re.search(r"World|Worlds", t): return "WLDs_PI"
    if re.search(r"World Championship|Worlds", t): return "WLDs"
    if re.search(r"Mid-Season|MSI", t): return "MSI"
    if re.search(r"Rift Rivals", t): return "RR"
    for pat, code in [(r"EU LCS|\bLEC\b", "LEC"), (r"NA LCS|\bLCS\b|\bLTA\b|LTA N", "LCS"), (r"LTA S", "LTA S"), (r"\bLCK\b|Champions Korea|KeSPA", "LCK"),
                      (r"\bLPL\b|Demacia|Tencent", "LPL"), (r"\bLMS\b", "LMS"), (r"\bLCP\b", "LCP"), (r"\bPCS\b", "PCS"), (r"\bVCS\b", "VCS"),
                      (r"\bLJL\b", "LJL"), (r"CBLOL", "CBLOL"), (r"\bLLA\b", "LLA"), (r"\bTCL\b", "TCL"), (r"\bLCL\b", "LCL"), (r"\bOPL\b", "OPL")]:
        if re.search(pat, t): return code
    return "OTH"


INTL_ALL = {"WLDs", "WLDs_PI", "MSI", "RR"}


def load() -> pd.DataFrame:
    h = pd.read_parquet(HIST)
    h["blue_picks"] = h.blue_picks.map(json.loads); h["red_picks"] = h.red_picks.map(json.loads)
    h["lg"] = h.tournament.map(league_code)
    h = h[h.lg != "OTH"].copy()
    h["year"] = h.date.dt.year
    h["kind"] = np.where(h.lg == "WLDs", "worlds", np.where(h.lg == "WLDs_PI", "playin", np.where(h.lg.isin(["MSI", "RR"]), "msi", "domestic")))
    return h.sort_values("date").reset_index(drop=True)


def stage_bt():
    h = load()
    G = h.rename(columns={"lg": "league"}).assign(y=h.blue_win)
    G["league"] = G.league.replace({"WLDs_PI": "WLDs"})
    home = st.home_groups(G, since="2015-01-01")
    bt = np.full(len(G), np.nan)
    for m in pd.period_range("2016-01", G.date.max(), freq="M"):
        a, b = m.start_time, (m + 1).start_time
        te = np.where((G.date >= a) & (G.date < b))[0]
        if not len(te): continue
        rating, side, _ = st.fit_bt(G, a, home, halflife=180, lam=2.0, sg=2.0)
        sub = G.iloc[te]; bt[te] = np.array([rating(x) - rating(z) for x, z in zip(sub.blue, sub.red)]) + side
        print(m, len(te), flush=True)
    out = pd.DataFrame({"gameid": G.gameid, "bt": bt}); out.to_parquet(PROC / "hist_bt.parquet", index=False)


def champ_index(domestic: pd.DataFrame, lam=3.0):
    """delta_c (scaling) et beta_c (force brute) par champion : y ~ bt + sum_c (beta_c + delta_c * z_len) (+1 bleu / -1 rouge)."""
    d = domestic.dropna(subset=["length", "bt"]).copy()
    d["z"] = d.groupby("year").length.transform(lambda s: (s - s.mean()) / (s.std() or 1))
    champs = sorted({c for l in d.blue_picks for c in l} | {c for l in d.red_picks for c in l}); ci = {c: i for i, c in enumerate(champs)}; nc = len(champs)
    r, c, v = [], [], []
    for i, (bp, rp, z) in enumerate(zip(d.blue_picks, d.red_picks, d.z)):
        for ch in bp: r += [i, i]; c += [ci[ch], nc + ci[ch]]; v += [1.0, z]
        for ch in rp: r += [i, i]; c += [ci[ch], nc + ci[ch]]; v += [-1.0, -z]
    X = sp.hstack([sp.csr_matrix((v, (r, c)), shape=(len(d), 2 * nc)), sp.csr_matrix(d.bt.to_numpy()[:, None])]).tocsr()
    lr = LogisticRegression(C=1.0 / lam, max_iter=400).fit(X, d.blue_win.to_numpy())
    co = lr.coef_[0]
    return {ch: float(co[nc + ci[ch]]) for ch in champs}, {ch: float(co[ci[ch]]) for ch in champs}, {ch: int(sum(ch in l for l in d.blue_picks) + sum(ch in l for l in d.red_picks)) for ch in champs}


def side_score(picks_b, picks_r, delta):
    return np.array([np.mean([delta.get(c, 0.0) for c in b]) - np.mean([delta.get(c, 0.0) for c in r]) for b, r in zip(picks_b, picks_r)])


def stage_analyze():
    h = load().merge(pd.read_parquet(PROC / "hist_bt.parquet"), on="gameid")
    h = h[h.date >= "2016-01-01"].copy()
    dom = h[h.kind == "domestic"]; wor = h[h.kind == "worlds"]
    print(f"games: domestique {len(dom)} | Worlds (main event) {len(wor)} | play-ins {int((h.kind == 'playin').sum())} | MSI/RR {int((h.kind == 'msi').sum())}")
    # ---------- (1) duree des games
    print("\n[1] DUREE MOYENNE (min) — Worlds main event vs ligues domestiques majeures, meme annee")
    major = dom[dom.lg.isin(["LCK", "LPL", "LEC", "LCS"])]
    rows = []
    for y, g in wor.groupby("year"):
        m_ = major[major.year == y]
        if len(m_) and g.length.notna().any():
            rows.append((y, g.length.mean() / 60, m_.length.mean() / 60, len(g)))
    tb = pd.DataFrame(rows, columns=["annee", "worlds", "domestique", "n_worlds"]); tb["ecart"] = tb.worlds - tb.domestique
    print(tb.round(2).to_string(index=False)); print(f"ecart moyen {tb.ecart.mean():+.2f} min ; Worlds plus long {int((tb.ecart > 0).sum())}/{len(tb)} annees")
    # ---------- (2) indice de scaling par champion (aucune fuite : domestique seulement)
    ydelta = {}
    for y in sorted(h.year.unique()):
        if y < 2017: continue
        ydelta[y] = champ_index(dom[dom.year != y])[0]            # leave-one-year-out (domestique des autres annees)
    delta_all, beta_all, cnt = champ_index(dom)
    top = sorted([(d_, c) for c, d_ in delta_all.items() if cnt[c] >= 150], reverse=True)
    print("\n[2] INDICE DE SCALING (delta, >=150 picks) — plus fort :", [(c, round(d_, 2)) for d_, c in top[:10]])
    print("    plus faible (early/tempo) :", [(c, round(d_, 2)) for d_, c in top[-10:]])
    def dscore(df):
        out = np.zeros(len(df))
        for y in df.year.unique():
            m_ = (df.year == y).to_numpy(); dl = ydelta.get(y, delta_all)
            out[m_] = side_score(df.blue_picks[m_], df.red_picks[m_], dl)
        return out
    h["scal"] = dscore(h)
    # ---------- (3) les equipes picks-elles plus de scaling aux Worlds ?
    print("\n[3] SCALING MOYEN DES PICKS (delta moyen des 10 champions) — Worlds vs domestique, meme annee")
    h["pick_scal"] = [np.mean([ydelta.get(y, delta_all).get(c, 0.0) for c in b + r]) for y, b, r in zip(h.year, h.blue_picks, h.red_picks)]
    t3 = h[h.kind.isin(["worlds", "domestic"])].groupby(["year", "kind"]).pick_scal.mean().unstack()
    t3["ecart"] = t3.worlds - t3.domestic; print(t3.dropna().round(3).to_string()); print(f"ecart moyen {t3.ecart.mean():+.3f}")
    # ---------- (4) le scaling gagne-t-il plus aux Worlds ? regression y ~ bt + scal (+ scal x worlds), IC par bootstrap
    print("\n[4] EFFET DU SCALING SUR LA VICTOIRE a force d'equipe egale : logit P(bleu) = a + b*bt + c*scal + d*scal*Worlds")
    T = h[h.kind.isin(["worlds", "domestic", "msi"]) & h.bt.notna()].copy(); T["W"] = (T.kind == "worlds").astype(float); T["M"] = (T.kind == "msi").astype(float)
    def fit(df):
        X = np.c_[df.bt, df.scal, df.scal * df.W, df.scal * df.M, df.W, df.M]
        lr = LogisticRegression(C=1e3, max_iter=400).fit(X, df.blue_win); return lr.coef_[0]
    co = fit(T); rng = np.random.default_rng(7); boots = []
    for _ in range(200):
        boots.append(fit(T.sample(len(T), replace=True, random_state=int(rng.integers(1e9)))))
    boots = np.array(boots); ci = np.percentile(boots, [2.5, 97.5], axis=0)
    names = ["bt", "scal (domestique)", "scal x Worlds (supplement)", "scal x MSI (supplement)", "Worlds", "MSI"]
    for n_, c_, lo, hi in zip(names, co, ci[0], ci[1]): print(f"   {n_:28s} {c_:+.3f}  IC95% [{lo:+.3f}, {hi:+.3f}]")
    print(f"   -> scaling aux Worlds = {co[1] + co[2]:+.3f} vs domestique {co[1]:+.3f} (supplement significatif: {'OUI' if (ci[0][2] > 0 or ci[1][2] < 0) else 'NON'})")
    # sous-echantillons : annee par annee (effet scaling Worlds seul)
    print("\n    Effet du scaling par edition des Worlds (coefficient sur scal, a force egale) :")
    for y, g in T[T.W == 1].groupby("year"):
        if len(g) < 40: continue
        lr = LogisticRegression(C=1e3).fit(np.c_[g.bt, g.scal], g.blue_win)
        print(f"     {y}: n={len(g):3d}  coef scal {lr.coef_[0][1]:+.2f}  | %games ou le bleu plus 'scaling' gagne: {((g.scal > 0) == (g.blue_win == 1))[g.scal.abs() > .02].mean():.1%}")
    # ---------- (5) le scaling gagne-t-il quand la game est LONGUE ? (controle)
    print("\n[5] SANITY : P(equipe la plus scaling gagne) selon la duree de la game (domestique | Worlds)")
    for lab, df in (("domestique", T[T.W == 0]), ("Worlds", T[T.W == 1])):
        df = df[df.length.notna() & (df.scal.abs() > .02)]
        res = []
        for lo, hi in ((0, 28), (28, 33), (33, 38), (38, 99)):
            g = df[(df.length / 60 >= lo) & (df.length / 60 < hi)]
            res.append(f"{lo}-{hi}min: {((g.scal > 0) == (g.blue_win == 1)).mean():.1%} (n={len(g)})")
        print(f"   {lab:11s}", " | ".join(res))
    h[["gameid", "date", "kind", "lg", "scal", "pick_scal", "bt", "length", "blue_win"]].to_parquet(PROC / "hist_scal.parquet", index=False)
    json.dump({"delta": delta_all, "beta": beta_all, "count": cnt}, open(PROC / "champ_scaling.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    {"bt": stage_bt, "analyze": stage_analyze}[sys.argv[1]]()
