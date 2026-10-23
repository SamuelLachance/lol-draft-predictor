"""Test HORS ECHANTILLON de la theorie "le scaling gagne plus aux Worlds" : walk-forward par edition (2017-2025).

Pour chaque edition Y des Worlds : entrainement sur TOUS les matchs avant le 1er match de l'edition, test sur les games des Worlds Y.
Definitions du scaling testees (declarees a l'avance, toutes rapportees) :
  attr   : moyenne de l'attribut manuel 'scaling' (0-3) des 5 picks, bleu - rouge       (attributs manuels)
  hyper  : nombre de champions de famille 'hypercarry' ou 'scaling>=3', bleu - rouge
  delta  : indice mesure sur les donnees (effet de la duree) — instable, rappel
Specifications : base = bt ; +intl-slope (bt x international) ; +X ; +X x Worlds
    python ml/study_worlds2.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_scaling as S, features as ft
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; PROC = ROOT / "data" / "processed"


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    H = S.load().merge(pd.read_parquet(PROC / "hist_bt.parquet"), on="gameid"); H = H[(H.date >= "2016-01-01") & H.bt.notna()].copy()
    H = H.merge(pd.read_parquet(PROC / "hist_scal.parquet")[["gameid", "scal"]], on="gameid")
    attrs = json.load(open(ROOT / "ml" / "champion_attrs.json", encoding="utf-8"))["champions"]
    A = {c["name"]: c for c in attrs}
    f_attr = lambda l: np.nanmean([A[c]["scaling"] if c in A else np.nan for c in l])
    f_hyp = lambda l: sum(1 for c in l if c in A and (A[c]["family"] == "hypercarry" or A[c]["scaling"] >= 3))
    H["attr"] = [f_attr(b) - f_attr(r) for b, r in zip(H.blue_picks, H.red_picks)]
    H["hyper"] = [f_hyp(b) - f_hyp(r) for b, r in zip(H.blue_picks, H.red_picks)]
    for c in ("attr", "hyper", "scal"):
        H[c] = H[c].fillna(0) / H[H.kind == "domestic"][c].std()
    H["W"] = (H.kind == "worlds").astype(float); H["I"] = H.kind.isin(["worlds", "msi", "playin"]).astype(float)
    H["btI"] = H.bt * H.I
    for c in ("attr", "hyper", "scal"):
        H[c + "W"] = H[c] * H.W; H[c + "I"] = H[c] * H.I
    specs = {"base (bt)": ["bt"], "+ pente intl (bt x intl)": ["bt", "btI"]}
    for c in ("attr", "hyper", "scal"):
        specs[f"+ {c}"] = ["bt", c]; specs[f"+ {c} x Worlds"] = ["bt", c, c + "W"]; specs[f"+ {c} x intl"] = ["bt", c, c + "I"]
    specs["+ attr x Worlds + pente intl"] = ["bt", "btI", "attr", "attrW"]
    test_sets = {"Worlds (main event)": lambda e: e.kind == "worlds", "Worlds + MSI + play-ins": lambda e: e.kind.isin(["worlds", "msi", "playin"])}
    res = {k: {s: ([], []) for s in specs} for k in test_sets}
    for y in range(2017, 2026):
        wy = H[(H.kind == "worlds") & (H.year == y)]
        if not len(wy): continue
        start = wy.date.min(); tr = H[H.date < start]; end = wy.date.max() + pd.Timedelta(days=1)
        for tname, sel in test_sets.items():
            te = H[(H.date >= start) & (H.date < end) & sel(H)]
            if not len(te): continue
            for s, cols in specs.items():
                lr = LogisticRegression(C=1e3, max_iter=400).fit(tr[cols], tr.blue_win)
                res[tname][s][0].append(lr.predict_proba(te[cols])[:, 1]); res[tname][s][1].append(te.blue_win.to_numpy())
    rng = np.random.default_rng(5)
    for tname in test_sets:
        p0 = np.concatenate(res[tname]["base (bt)"][0]); y0 = np.concatenate(res[tname]["base (bt)"][1])
        idx = rng.integers(0, len(y0), (400, len(y0)))
        print(f"\n=== {tname} : {len(y0)} games, 9 editions (hors echantillon) — base log-loss {ll(p0, y0):.4f}, acc {((p0 > .5) == y0).mean():.3f}")
        for s in specs:
            p = np.concatenate(res[tname][s][0]); d = ll(p, y0) - ll(p0, y0)
            bs = np.array([ll(p[i], y0[i]) - ll(p0[i], y0[i]) for i in idx])
            sig = "  <-- mieux" if np.percentile(bs, 97.5) < 0 else ("  <-- pire" if np.percentile(bs, 2.5) > 0 else "")
            print(f"   {s:30s} log-loss {ll(p, y0):.4f}  acc {((p > .5) == y0).mean():.3f}  diff {d:+.4f}  IC95% [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]{sig}")
    # coefficients finaux (toutes editions) pour l'interpretation
    tr = H[H.kind.isin(["domestic", "worlds"])]
    for c in ("attr", "hyper"):
        lr = LogisticRegression(C=1e3, max_iter=400).fit(tr[["bt", c, c + "W"]], tr.blue_win)
        print(f"\ncoefficients ({c}) toutes annees : domestique {lr.coef_[0][1]:+.3f}/SD ; supplement Worlds {lr.coef_[0][2]:+.3f}/SD")
    # attr par edition (coefficient simple sur les Worlds)
    print("\nEffet 'attr' par edition des Worlds (coef/SD, a force egale) :")
    for y, g in H[H.kind == "worlds"].groupby("year"):
        lr = LogisticRegression(C=1e3).fit(g[["bt", "attr"]], g.blue_win); print(f"   {y}: n={len(g):3d}  {lr.coef_[0][1]:+.2f}")


if __name__ == "__main__":
    main()
