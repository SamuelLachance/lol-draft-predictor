"""Que valent les synergies, les counters (matchups) et les THEMES de composition ? Ablation walk-forward (refit trimestriel).

 variantes du modele de draft : champion seul | + matchups de lane (counters) | + synergies | complet
 themes : engage/all-in, poke/controle, scaling/hypercarry (meme scoring que le draft assistant) -> matrice theme-contre-theme
    python ml/exp_draft_parts.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as st, model_v2 as mv, draftmodel as dm
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; PROC = ROOT / "data" / "processed"
ROLES = ["top", "jng", "mid", "bot", "sup"]


def theme_scores(champs, A):
    al = po = hy = n = 0.0
    for c in champs:
        a = A.get(c)
        if not a: continue
        n += 1
        al += 2 * a["engage"] + (1.5 if a["family"] in ("engage", "catch") else 0) + (1 if a["family"] == "assassin" else 0) + .5 * a["cc"] * a["engage"] + .5 * a["pick"]
        po += 1.5 * a["poke"] + a["disengage"] + (1.5 if a["family"] in ("poke", "disengage") else 0) + (.7 if a["family"] == "enchanter" else 0) + .5 * a["anti_dive"] - a["engage"]
        hy += (3 if a["family"] == "hypercarry" else 0) + (2 if a["family"] == "enchanter" else 0) + .4 * a["scaling"] + .3 * a["waveclear"] + .5 * a["global"]
    return (al / n, po / n, hy / n) if n >= 3 else (np.nan, np.nan, np.nan)


def main():
    F = pd.read_parquet(PROC / "v4_feats.parquet").merge(pd.read_parquet(PROC / "v4_pbt.parquet"), on="gameid"); F["date"] = pd.to_datetime(F.date)
    F = F[F.league.isin(set(st.GROUP) | st.INTL)].sort_values("date").reset_index(drop=True)
    home = st.home_groups(F)
    tier = (F.home_b.isin(st.MAJOR) & F.home_r.isin(st.MAJOR) & F.bt.notna() & F.pbt.notna()).to_numpy()
    # ---------- variantes du modele de draft
    old = dict(dm.SCALE); variants = {"champion seul": (1, 0, 0), "+ counters (matchups)": (1, 1, 0), "+ synergies": (1, 0, 1), "complet": (1, 1, 1)}
    out = {k: np.full(len(F), np.nan) for k in variants}
    cuts = [pd.Timestamp(f"{y}-{m:02d}-01") for y in range(2023, 2027) for m in (1, 4, 7, 10)][1:] + [pd.Timestamp("2027-01-01")]
    for a, b in zip(cuts[:-1], cuts[1:]):
        te = np.where((F.date >= a) & (F.date < b) & tier)[0]
        if not len(te): continue
        for name, (sc, sm, ss) in variants.items():
            dm.SCALE.update({"c": old["c"] * sc, "m": old["m"] * sm, "s": old["s"] * ss})
            mod = mv.fit_draft(F, a); out[name][te] = dm.logit(mod, *dm.to_lists(F.iloc[te]))
        print("fit", a.date(), flush=True)
    dm.SCALE.update(old)
    # ---------- themes
    raw = json.load(open(ROOT / "web" / "public" / "draft" / "data" / "champions.json", encoding="utf-8"))["champions"]
    A = {c["name"]: c for c in raw}
    pb = F[[f"blue_{r}" for r in ROLES]].astype(str).values.tolist(); pr = F[[f"red_{r}" for r in ROLES]].astype(str).values.tolist()
    tb = np.array([theme_scores(l, A) for l in pb]); tr = np.array([theme_scores(l, A) for l in pr])
    ok = ~np.isnan(tb).any(1) & ~np.isnan(tr).any(1)
    # theme = argmax du score z-normalise (0 engage, 1 poke, 2 scaling)
    mu = np.nanmean(np.vstack([tb, tr]), 0); sd = np.nanstd(np.vstack([tb, tr]), 0)
    th_b = np.where(ok, np.nanargmax((tb - mu) / sd, 1), -1); th_r = np.where(ok, np.nanargmax((tr - mu) / sd, 1), -1)
    names = ["engage/all-in", "poke/controle", "scaling/hypercarry"]
    for i in range(3):
        for j in range(i + 1, 3):
            F[f"th_{i}{j}"] = np.where((th_b == i) & (th_r == j), 1.0, np.where((th_b == j) & (th_r == i), -1.0, 0.0))
    F["th_gap"] = np.where(ok, (tb[:, 0] - tr[:, 0]) * 0, 0)
    for k, v in out.items(): F["d_" + k] = v
    T = F[tier & F["d_complet"].notna()].copy(); T["ym"] = T.date.dt.to_period("M"); y = T.y.to_numpy()
    def ll(p, yy): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())
    base = ["bt", "pbt", "ts", "d_fw", "h2h", "comfort", "comfort_min"]; TH = ["th_01", "th_02", "th_12"]
    specs = {"sans draft": base}
    for k in variants: specs["draft : " + k] = base + ["d_" + k]
    specs["complet + themes"] = base + ["d_complet"] + TH; specs["themes seuls"] = base + TH
    res = {}
    for name, cols in specs.items():
        ps, ys = [], []
        for m in [x for x in sorted(T.ym.unique()) if x >= pd.Period("2024-01", "M")]:
            trn, te = T[(T.ym < m) & (T.date >= "2023-06-01")], T[T.ym == m]
            age = (te.date.min() - trn.date).dt.total_seconds().to_numpy() / 86400
            mu_ = trn[cols].mean(); sd_ = trn[cols].std().replace(0, 1)
            lr = LogisticRegression(C=10, max_iter=500).fit(((trn[cols] - mu_) / sd_).fillna(0), trn.y, sample_weight=0.5 ** (age / 900))
            ps.append(lr.predict_proba(((te[cols] - mu_) / sd_).fillna(0))[:, 1]); ys.append(te.y.to_numpy())
        res[name] = (np.concatenate(ps), np.concatenate(ys))
    p0, y0 = res["sans draft"]; rng = np.random.default_rng(9); ix = rng.integers(0, len(y0), (400, len(y0)))
    print(f"\nn={len(y0)} games (2024-01 -> 2026-10) ; reference 'sans draft' ll={ll(p0, y0):.4f} acc={((p0 > .5) == y0).mean():.3f}")
    for name, (p, yy) in res.items():
        bs = np.array([ll(p[i], yy[i]) - ll(p0[i], y0[i]) for i in ix]); lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"   {name:28s} ll {ll(p, yy):.4f} acc {((p > .5) == yy).mean():.3f} diff vs sans draft {ll(p, yy) - ll(p0, y0):+.4f} IC95% [{lo:+.4f}, {hi:+.4f}]{'  <-- mieux' if hi < 0 else ''}")
    # ---------- matrice theme-contre-theme (toutes games 2021+, a force d'equipe egale)
    G = F[ok & F.bt.notna() & (F.date >= "2021-01-01") & F.league.isin(set(st.GROUP) | st.INTL)].copy(); G["tb"] = th_b[G.index]; G["tr"] = th_r[G.index]
    print("\nPart des themes (tier-1 2025+):", {names[i]: f"{((th_b[(F.date >= '2025-01-01').to_numpy() & ok] == i).mean()):.0%}" for i in range(3)})
    print("Matrice : P(victoire du theme en ligne contre le theme en colonne) corrigee de la force d'equipe (logit bt controle)")
    import statsmodels.api as sm
    X = pd.DataFrame({"bt": G.bt, **{f"t{i}{j}": np.where((G.tb == i) & (G.tr == j), 1.0, np.where((G.tb == j) & (G.tr == i), -1.0, 0.0)) for i in range(3) for j in range(i + 1, 3)}})
    fit = sm.Logit(G.y.to_numpy(), X).fit(disp=0)
    for i in range(3):
        for j in range(i + 1, 3):
            c = fit.params[f"t{i}{j}"]; lo, hi = fit.conf_int().loc[f"t{i}{j}"]; n_ = int((X[f"t{i}{j}"] != 0).sum())
            print(f"   {names[i]:20s} vs {names[j]:20s}: effet {c:+.3f} logit ({100 * (1 / (1 + np.exp(-c)) - .5):+.1f} pts) IC95% [{lo:+.3f}, {hi:+.3f}] n={n_}")
    # winrate par theme (a force egale) : theme x intl
    for lab, d in (("toutes games", G), ("Worlds/MSI", G[G.intl == 1])):
        s = pd.concat([pd.DataFrame({"t": d.tb, "w": d.y}), pd.DataFrame({"t": d.tr, "w": 1 - d.y})]).groupby("t").w.agg(["mean", "size"])
        print(f"   winrate brut par theme ({lab}):", {names[int(i)]: f"{r['mean']:.1%} (n={int(r['size'])})" for i, r in s.iterrows()})


if __name__ == "__main__":
    main()
