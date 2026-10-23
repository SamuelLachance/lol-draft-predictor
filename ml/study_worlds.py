"""Experiences specifiques aux Worlds (suite de study_scaling.py) :
 A. scaling : effet par ecart-type, par epoque, IC bootstrap ; indice alternatif (attributs manuels) ; stabilite de l'indice
 B. diagnostics Worlds : cote bleu, calibration des favoris (pente), upsets, region, experience
 C. apport au modele : le scaling (x Worlds) ameliore-t-il le stack en walk-forward sur les Worlds ?
    python ml/study_worlds.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_scaling as S, strength as st, features as ft
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; PROC = ROOT / "data" / "processed"
rng = np.random.default_rng(11)


def boot(fn, df, n=300):
    res = [fn(df.sample(len(df), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(n)]
    return np.percentile(np.array(res), [2.5, 97.5], axis=0)


def main():
    H = S.load().merge(pd.read_parquet(PROC / "hist_bt.parquet"), on="gameid"); H = H[H.date >= "2016-01-01"].copy()
    sc = pd.read_parquet(PROC / "hist_scal.parquet")[["gameid", "scal", "pick_scal"]]
    H = H.merge(sc, on="gameid"); H["W"] = (H.kind == "worlds").astype(float); H["M"] = (H.kind == "msi").astype(float)
    sd = H[H.kind == "domestic"].scal.std(); H["sz"] = H.scal / sd
    T = H[H.kind.isin(["domestic", "worlds", "msi"]) & H.bt.notna()].copy()
    # ============ A. scaling
    print("=== A. EFFET DU SCALING (par ecart-type d'ecart de 'scaling' entre les deux drafts), a force d'equipe egale ===")
    def coef_sz(df): return LogisticRegression(C=1e3, max_iter=300).fit(np.c_[df.bt, df.sz], df.blue_win).coef_[0][1]
    for lab, df in (("domestique (toutes annees)", T[T.kind == "domestic"]), ("domestique 2021+", T[(T.kind == "domestic") & (T.year >= 2021)]),
                    ("Worlds (toutes annees)", T[T.kind == "worlds"]), ("Worlds 2016-2020", T[(T.kind == "worlds") & (T.year <= 2020)]),
                    ("Worlds 2021-2025", T[(T.kind == "worlds") & (T.year >= 2021)]), ("MSI", T[T.kind == "msi"]), ("Play-ins", H[(H.kind == "playin") & H.bt.notna()])):
        c = coef_sz(df); lo, hi = boot(coef_sz, df, 200)
        print(f"  {lab:28s} n={len(df):5d}  coef/SD = {c:+.3f}  IC95% [{lo:+.3f}, {hi:+.3f}]   (1 SD de scaling ~ {100 * (1 / (1 + np.exp(-c)) - .5):+.1f} pts de win%)")
    dom, wor = T[T.kind == "domestic"], T[T.kind == "worlds"]
    def diff(df):
        a = df[df.kind == "worlds"]; b = df[df.kind == "domestic"]
        return coef_sz(a) - coef_sz(b)
    b_ = pd.concat([wor, dom.sample(len(wor) * 6, random_state=1)])
    d = diff(b_); lo, hi = boot(diff, b_, 200)
    print(f"  Difference Worlds - domestique (coef/SD) = {d:+.3f}  IC95% [{lo:+.3f}, {hi:+.3f}]  -> {'SIGNIFICATIF' if lo > 0 or hi < 0 else 'non significatif'}")
    # taux brut de victoire de la draft la plus 'scaling'
    print("\n  Taux brut : l'equipe avec la draft la plus 'scaling' gagne (|ecart| > 0.5 SD) :")
    for lab, df in (("domestique", dom), ("Worlds", wor), ("MSI", T[T.kind == "msi"])):
        g = df[df.sz.abs() > .5]; k = int(((g.sz > 0) == (g.blue_win == 1)).sum()); n = len(g); p = k / n
        print(f"    {lab:11s} {p:.1%}  (n={n}, IC95% +/- {1.96 * np.sqrt(p * (1 - p) / n) * 100:.1f} pts)")
    # indice alternatif : attributs manuels des champions (scaling 0-3, famille hypercarry)
    attrs = ft.load_attrs(ROOT / "ml" / "champion_attrs.json")
    def attr_diff(b, r):
        f = lambda l: np.nanmean([attrs[c]["scaling"] if c in attrs else np.nan for c in l])
        return f(b) - f(r)
    T["attr"] = [attr_diff(b, r) for b, r in zip(T.blue_picks, T.red_picks)]
    cov = np.mean([c in attrs for l in T.blue_picks for c in l]); print(f"\n  Indice ALTERNATIF = attribut 'scaling' (0-3) fait main, couverture {cov:.0%} des picks :")
    X = T.dropna(subset=["attr"]).copy(); X["az"] = X.attr / X.attr.std()
    def coef_a(df): return LogisticRegression(C=1e3, max_iter=300).fit(np.c_[df.bt, df.az], df.blue_win).coef_[0][1]
    for lab, df in (("domestique", X[X.kind == "domestic"]), ("Worlds", X[X.kind == "worlds"]), ("Worlds 2021-2025", X[(X.kind == "worlds") & (X.year >= 2021)])):
        c = coef_a(df); lo, hi = boot(coef_a, df, 200); print(f"    {lab:18s} n={len(df):5d} coef/SD = {c:+.3f}  IC95% [{lo:+.3f}, {hi:+.3f}]")
    # stabilite de l'indice delta (avant 2022 vs apres)
    dom_all = H[H.kind == "domestic"]
    d1 = S.champ_index(dom_all[dom_all.year <= 2021])[0]; d2 = S.champ_index(dom_all[dom_all.year >= 2022])[0]
    com = [c for c in d1 if c in d2]; print(f"\n  Stabilite de l'indice delta (<=2021 vs >=2022, {len(com)} champions) : correlation = {np.corrcoef([d1[c] for c in com], [d2[c] for c in com])[0, 1]:.2f}")

    # ============ B. diagnostics Worlds
    print("\n=== B. DIAGNOSTICS SPECIFIQUES AUX WORLDS ===")
    print("  Win% du cote BLEU :", {lab: f"{df.blue_win.mean():.1%} (n={len(df)})" for lab, df in (("domestique", dom), ("Worlds", wor), ("MSI", T[T.kind == 'msi']))})
    for lab, df in (("Worlds 2016-2020", wor[wor.year <= 2020]), ("Worlds 2021-2025", wor[wor.year >= 2021])): print(f"    {lab}: {df.blue_win.mean():.1%}")
    def slope(df): return LogisticRegression(C=1e3, max_iter=300).fit(np.c_[df.bt], df.blue_win).coef_[0][0]
    print("  Pente de calibration de la force d'equipe (1.00 = parfaitement calibre ; <1 = favoris surestimes) :")
    for lab, df in (("domestique", dom), ("Worlds", wor), ("MSI", T[T.kind == "msi"]), ("Worlds 2021-2025", wor[wor.year >= 2021])):
        c = slope(df); lo, hi = boot(slope, df, 200); print(f"    {lab:18s} {c:.3f}  IC95% [{lo:.3f}, {hi:.3f}]")
    wor2 = wor.copy(); wor2["p"] = 1 / (1 + np.exp(-wor2.bt)); wor2["fav_wins"] = ((wor2.p > .5) == (wor2.blue_win == 1))
    print(f"  Le favori (bt) gagne : Worlds {wor2.fav_wins.mean():.1%} (n={len(wor2)}) vs domestique {((1 / (1 + np.exp(-dom.bt)) > .5) == (dom.blue_win == 1)).mean():.1%}")
    # ============ C. apport au modele (stack reduit, walk-forward par edition des Worlds, delta fit AVANT l'edition)
    print("\n=== C. LE SCALING AMELIORE-T-IL LE MODELE AUX WORLDS ? (walk-forward par edition, indice calcule avant l'edition) ===")
    F = pd.read_parquet(PROC / "v4_feats.parquet"); F["date"] = pd.to_datetime(F.date)
    F = F[F.bt.notna() & F.league.isin(set(st.GROUP) | st.INTL)].copy()
    picks_b = F[[f"blue_{r}" for r in S.ROLES]].astype(str).values.tolist(); picks_r = F[[f"red_{r}" for r in S.ROLES]].astype(str).values.tolist()
    F["scal"] = np.nan
    dates_fit = [pd.Timestamp(f"{y}-{m:02d}-01") for y in range(2021, 2026) for m in (1, 4, 7, 10)] + [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01"), pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-01")]
    arr = np.full(len(F), np.nan)
    dom_hist = H[H.kind == "domestic"]
    for a, b in zip(dates_fit, dates_fit[1:] + [pd.Timestamp("2027-01-01")]):
        dl = S.champ_index(dom_hist[dom_hist.date < a])[0]
        m = ((F.date >= a) & (F.date < b)).to_numpy()
        arr[m] = S.side_score([picks_b[i] for i in np.where(m)[0]], [picks_r[i] for i in np.where(m)[0]], dl)
    F["scal"] = arr / sd; F["W"] = (F.league == "WLDs").astype(float); F["scalW"] = F.scal * F.W; F["intl"] = F.league.isin(st.INTL).astype(float); F["scalI"] = F.scal * F.intl
    F = F[F.scal.notna()]
    tier = F.home_b.isin(st.MAJOR) & F.home_r.isin(st.MAJOR)
    base = ["bt", "ts", "draft", "d_fw", "h2h"]
    def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    variants = {"stack reduit": base, "+ scaling": base + ["scal"], "+ scaling x Worlds": base + ["scal", "scalW"], "+ scaling x intl": base + ["scal", "scalI"]}
    for scope in ("Worlds (5 editions 2021-2025)", "tous les matchs internationaux", "ensemble des matchs majeurs 2023-2025"):
        res = {}
        for name, cols in variants.items():
            ps, ys = [], []
            for a in [pd.Timestamp(f"{y}-{m:02d}-01") for y in range(2022, 2026) for m in range(1, 13)] + [pd.Timestamp(f"2026-{m:02d}-01") for m in range(1, 11)]:
                b = a + pd.offsets.MonthBegin(1)
                te = F[(F.date >= a) & (F.date < b) & tier]
                if scope.startswith("Worlds"): te = te[te.league == "WLDs"]
                elif scope.startswith("tous"): te = te[te.intl == 1]
                else: te = te[te.date >= "2023-01-01"]
                if not len(te): continue
                tr = F[(F.date < a) & tier & (F.date >= "2021-06-01")]
                mu = tr[cols].mean(); s_ = tr[cols].std().replace(0, 1)
                age = (a - tr.date).dt.total_seconds().to_numpy() / 86400
                lr = LogisticRegression(C=10, max_iter=500).fit(((tr[cols] - mu) / s_).fillna(0), tr.y, sample_weight=0.5 ** (age / 900))
                ps.append(lr.predict_proba(((te[cols] - mu) / s_).fillna(0))[:, 1]); ys.append(te.y.to_numpy())
            p, y = np.concatenate(ps), np.concatenate(ys); res[name] = (p, y)
        print(f"  [{scope}] n={len(res['stack reduit'][1])}")
        p0, y0 = res["stack reduit"]
        for name, (p, y) in res.items():
            d = ll(p, y) - ll(p0, y)
            idx = np.random.default_rng(3).integers(0, len(y), (300, len(y))); bs = [ll(p[i], y[i]) - ll(p0[i], y0[i]) for i in idx]
            print(f"     {name:22s} log-loss {ll(p, y):.4f}  acc {((p > .5) == y).mean():.3f}  diff vs reduit {d:+.4f}  IC95% [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")


if __name__ == "__main__":
    main()
