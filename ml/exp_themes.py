"""Coherence et FIDELITE au theme : une equipe gagne-t-elle plus quand elle draft dans son style / une draft "pure" ?

 theme d'une draft = vecteur z-normalise (engage/all-in, poke/controle, scaling/hypercarry) (scoring par attributs/familles)
 coherence  = ecart entre le theme dominant et le 2e (draft "pure" vs "melangee")
 fidelite   = similarite (cosinus) et distance entre la draft du jour et le theme HABITUEL de l'equipe (moyenne ponderee recence,
              demi-vie 90 j, calculee uniquement sur le passe)
    python ml/exp_themes.py
"""
from __future__ import annotations
import json, sys, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as st
from exp_draft_parts import theme_scores
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; PROC = ROOT / "data" / "processed"
ROLES = ["top", "jng", "mid", "bot", "sup"]; HL = 90.0


def main():
    F = pd.read_parquet(PROC / "v4_feats.parquet").merge(pd.read_parquet(PROC / "v4_pbt.parquet"), on="gameid"); F["date"] = pd.to_datetime(F.date)
    F = F[F.league.isin(set(st.GROUP) | st.INTL)].sort_values("date").reset_index(drop=True)
    A = {c["name"]: c for c in json.load(open(ROOT / "ml" / "champion_attrs.json", encoding="utf-8"))["champions"]}
    pb = F[[f"blue_{r}" for r in ROLES]].astype(str).values.tolist(); pr = F[[f"red_{r}" for r in ROLES]].astype(str).values.tolist()
    tb = np.array([theme_scores(l, A) for l in pb]); tr = np.array([theme_scores(l, A) for l in pr])
    mu = np.nanmean(np.vstack([tb, tr]), 0); sd = np.nanstd(np.vstack([tb, tr]), 0); Zb = (tb - mu) / sd; Zr = (tr - mu) / sd
    days = (F.date - F.date.iloc[0]).dt.total_seconds().to_numpy() / 86400
    hist = {}                                               # equipe -> [somme ponderee des vecteurs, somme des poids, dernier jour]
    def mean_at(team, t):
        h = hist.get(team)
        if h is None: return None, 0
        f = 0.5 ** ((t - h[2]) / HL); return h[0] * f / (h[1] * f), h[1] * f
    cols = {k: np.full(len(F), np.nan) for k in ("fid_cos_b", "fid_cos_r", "fid_dist_b", "fid_dist_r", "coh_b", "coh_r", "n_b", "n_r")}
    for i in range(len(F)):
        t = days[i]
        for side, team, Z in (("b", F.blue.iat[i], Zb[i]), ("r", F.red.iat[i], Zr[i])):
            if np.isnan(Z).any(): continue
            s = np.sort(Z); cols["coh_" + side][i] = s[-1] - s[-2]
            m, w = mean_at(team, t); cols["n_" + side][i] = w
            if m is not None and w >= 3:
                cols["fid_cos_" + side][i] = float(Z @ m / (np.linalg.norm(Z) * np.linalg.norm(m) + 1e-9)); cols["fid_dist_" + side][i] = -float(np.linalg.norm(Z - m))
        for team, Z in ((F.blue.iat[i], Zb[i]), (F.red.iat[i], Zr[i])):
            if np.isnan(Z).any(): continue
            h = hist.get(team)
            if h is None: hist[team] = [Z.copy(), 1.0, t]
            else:
                f = 0.5 ** ((t - h[2]) / HL); hist[team] = [h[0] * f + Z, h[1] * f + 1, t]
    for k, v in cols.items(): F[k] = v
    F["d_fid_cos"] = F.fid_cos_b - F.fid_cos_r; F["d_fid_dist"] = F.fid_dist_b - F.fid_dist_r; F["d_coh"] = F.coh_b - F.coh_r
    tier = F.home_b.isin(st.MAJOR) & F.home_r.isin(st.MAJOR) & F.bt.notna()
    G = F[tier & (F.date >= "2022-01-01")].dropna(subset=["d_fid_cos", "d_coh"]).copy()
    print(f"games 2022+ : {len(G)}")
    # ------------- effets a force d'equipe egale (statsmodels, IC95%)
    print("\n[1] Effet a force d'equipe egale (logit, par ecart-type de l'ecart bleu-rouge)")
    for name in ("d_fid_cos", "d_fid_dist", "d_coh"):
        for lab, d in (("toutes games", G), ("Worlds/MSI", G[G.intl == 1]), ("2025+", G[G.date >= "2025-01-01"])):
            X = pd.DataFrame({"bt": d.bt, "x": d[name] / d[name].std()}); fit = sm.Logit(d.y.to_numpy(), X).fit(disp=0); lo, hi = fit.conf_int().loc["x"]
            print(f"   {name:11s} {lab:13s} n={len(d):5d} coef {fit.params['x']:+.3f} IC95% [{lo:+.3f}, {hi:+.3f}]{'  <--' if lo > 0 or hi < 0 else ''}")
    # cote "fidele vs infidele" : la draft la plus fidele a son theme gagne-t-elle plus ?
    g = G[G.d_fid_cos.abs() > 0.15]; print(f"\n   Taux brut : l'equipe la plus fidele a son theme gagne {((g.d_fid_cos > 0) == (g.y == 1)).mean():.1%} (n={len(g)}) ; la plus coherente {((G.d_coh > 0) == (G.y == 1))[G.d_coh.abs() > .3].mean():.1%}")
    # ------------- test hors echantillon (walk-forward) dans le stack
    T = F[tier & (F.date >= "2023-06-01")].dropna(subset=["d_fid_cos", "d_coh", "pbt"]).copy(); T["ym"] = T.date.dt.to_period("M"); y0 = None
    base = ["bt", "pbt", "ts", "d_fw", "h2h", "comfort", "comfort_min"]
    specs = {"base": base, "+ fidelite (cos)": base + ["d_fid_cos"], "+ fidelite (distance)": base + ["d_fid_dist"], "+ coherence": base + ["d_coh"], "+ les trois": base + ["d_fid_cos", "d_fid_dist", "d_coh"]}
    def ll(p, yy): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())
    res = {}
    for name, cs in specs.items():
        ps, ys = [], []
        for m in [x for x in sorted(T.ym.unique()) if x >= pd.Period("2024-01", "M")]:
            trn, te = T[(T.ym < m)], T[T.ym == m]; age = (te.date.min() - trn.date).dt.total_seconds().to_numpy() / 86400
            m_ = trn[cs].mean(); s_ = trn[cs].std().replace(0, 1)
            lr = LogisticRegression(C=10, max_iter=500).fit(((trn[cs] - m_) / s_).fillna(0), trn.y, sample_weight=0.5 ** (age / 900))
            ps.append(lr.predict_proba(((te[cs] - m_) / s_).fillna(0))[:, 1]); ys.append(te.y.to_numpy())
        res[name] = (np.concatenate(ps), np.concatenate(ys))
    p0, y0 = res["base"]; ix = np.random.default_rng(1).integers(0, len(y0), (400, len(y0)))
    print(f"\n[2] Hors echantillon (walk-forward) n={len(y0)} ; base ll={ll(p0, y0):.4f} acc={((p0 > .5) == y0).mean():.3f}")
    for name, (p, yy) in res.items():
        bs = np.array([ll(p[i], yy[i]) - ll(p0[i], y0[i]) for i in ix]); lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"   {name:24s} ll {ll(p, yy):.4f} acc {((p > .5) == yy).mean():.3f} diff {ll(p, yy) - ll(p0, y0):+.4f} IC95% [{lo:+.4f}, {hi:+.4f}]{'  <-- mieux' if hi < 0 else ''}")
    # ------------- identite : equipes fideles vs versatiles (consistance du theme) et fidelite moyenne
    print("\n[3] Equipes les plus FIDELES a leur theme (cosinus moyen 2025+, >=60 games) :")
    r = pd.concat([pd.DataFrame({"team": F.blue, "f": F.fid_cos_b, "d": F.date}), pd.DataFrame({"team": F.red, "f": F.fid_cos_r, "d": F.date})]); r = r[(r.d >= "2025-01-01") & r.f.notna()]
    s = r.groupby("team").f.agg(["mean", "size"]); s = s[s["size"] >= 60].sort_values("mean", ascending=False)
    print("   fideles:", [(t, round(v, 2)) for t, v in s["mean"].head(6).items()], "\n   versatiles:", [(t, round(v, 2)) for t, v in s["mean"].tail(6).items()])


if __name__ == "__main__":
    main()
