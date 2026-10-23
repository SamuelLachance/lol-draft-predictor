"""Modele de probabilite LIVE (en cours de game) appris sur de vraies timelines lolesports (data/raw/timelines, cf. fetch_timelines.py).

P(bleu gagne | prior pre-game du modele, minute, ecart d'or, kills, tours, dragons/ame, barons, inhibiteurs, niveaux, joueurs en vie, CS).
Le prior est la proba POST-draft walk-forward du modele v4 (wf_preds) -> le modele live apprend a quelle vitesse l'etat du jeu l'emporte sur le prior.
Validation : split temporel par game (jamais de fuite entre echantillons d'une meme game). Export : web/public/model/live_v1.json (regression logistique a interactions temporelles + arbres LightGBM, melanges en logit).
Usage : python ml/train_live.py
"""
from __future__ import annotations
import glob, json, re, sys, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["lp", "tm", "gd", "gdr", "kd", "kt", "td", "dd", "sd", "bd", "ind", "ld", "ad", "csd", "pb", "pr", "dgd3", "bbuf"]
MIN_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 300
MONO = {"bbuf": 1, "lp": 1, "gd": 1, "gdr": 1, "kd": 1, "td": 1, "bd": 1, "ind": 1, "ld": 1, "sd": 1}


def nk(s): return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", str(s).replace("ø", "o").replace("Ø", "O")).lower())
def same(a, b):
    x, y = nk(a), nk(b)
    return bool(x and y and (x == y or (min(len(x), len(y)) >= 3 and (x in y or y in x))))


def row_feats(b, r, tmin, lp, prev=None):
    """b / r = [or, kills, tours, barons, inhibiteurs, dragons[], niveaux, vivants, cs] ; renvoie le vecteur de features (point de vue bleu) - IDENTIQUE a match/index.html"""
    soul = lambda d: 1 if len(d) >= 4 else 0
    return [lp, tmin, (b[0] - r[0]) / 1000.0, (b[0] - r[0]) / max(1, b[0] + r[0]), b[1] - r[1], b[1] + r[1], b[2] - r[2], len(b[5]) - len(r[5]), soul(b[5]) - soul(r[5]),
            b[3] - r[3], b[4] - r[4], b[6] - r[6], b[7] - r[7], b[8] - r[8], b[7], r[7],
            ((b[0] - r[0]) - (prev[0][0] - prev[1][0])) / 1000.0 if prev else 0.0,                       # dynamique de l'ecart d'or sur ~3 min
            ((1 if b[3] > prev[0][3] else 0) - (1 if r[3] > prev[1][3] else 0)) if prev else 0]          # buff de Baron pris dans les ~3 dernieres minutes


def load():
    base = pd.read_parquet(ROOT / "data/processed/v4_base.parquet", columns=["gameid", "date", "league", "blue", "red", "y"]).merge(
        pd.read_parquet(ROOT / "data/processed/wf_preds.parquet", columns=["gameid", "p_pre", "p_post"]), on="gameid")
    base["date"] = pd.to_datetime(base["date"]); base = base[base.date >= "2024-12-25"].reset_index(drop=True)
    byd = base.sort_values("date")
    rows, n_ok, n_all = [], 0, 0
    for f in glob.glob(str(ROOT / "data/raw/timelines/*.json")):
        g = json.load(open(f, encoding="utf-8")); n_all += 1
        t0 = pd.Timestamp(g["t0"]); c = byd[(byd.date > t0 - pd.Timedelta(hours=4)) & (byd.date < t0 + pd.Timedelta(hours=4))]
        c = c[[same(x, g["blue"]) and same(y, g["red"]) or same(x, g["red"]) and same(y, g["blue"]) for x, y in zip(c.blue, c.red)]]
        if c.empty: continue
        c = c.iloc[int((c.date - t0).abs().values.argmin())]
        flip = not (same(c.blue, g["blue"]) and same(c.red, g["red"]))
        p = 1 - c.p_post if flip else c.p_post; pp = 1 - c.p_pre if flip else c.p_pre; y = 1 - c.y if flip else c.y
        last = g["rows"][-1]
        # controle : l'issue de la base doit etre coherente avec l'etat final (tours/inhibiteurs) sinon on jette (mauvais appariement)
        fb, fr = last["b"], last["r"]; fin = (fb[4] - fr[4]) * 10 + (fb[2] - fr[2])
        if fin != 0 and (fin > 0) != (y == 1): continue
        n_ok += 1; lp = float(np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98)))); tz = pd.Timestamp(g["t0"])
        gr = g["rows"]
        for k, r in enumerate(gr):
            tm = (pd.Timestamp(r["ts"]).tz_convert(None) - tz).total_seconds() / 60.0
            if tm < 0.5 or r["st"] == "finished": continue              # pas de frames de fin de game (issue deja jouee) : on apprend ce que voit un spectateur en direct
            rows.append([g["id"], g["league"], c.date, y, tm, float(pp), *row_feats(r["b"], r["r"], tm, lp, (gr[k - 3]["b"], gr[k - 3]["r"]) if k >= 3 else None)])
    D = pd.DataFrame(rows, columns=["game", "league", "date", "y", "tm_", "p_pre"] + FEATS)
    print(f"{n_ok}/{n_all} games appariees ({len(D)} echantillons)"); return D


def ll(y, p): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
sg = lambda z: 1 / (1 + np.exp(-z))
LR_COLS = ["lp", "gd", "gdr", "kd", "td", "dd", "sd", "bd", "ind", "ld", "ad", "csd", "dgd3", "bbuf"]
TAUS = (12.0, 30.0)


def basis(tm):
    t = np.asarray(tm, float); return np.stack([np.ones_like(t), np.exp(-t / TAUS[0]), np.exp(-t / TAUS[1])], 1)


def lr_design(d, sd):
    B = basis(d.tm.values); cols = [B]
    for c in LR_COLS: cols.append(B * (d[c].values / sd[c])[:, None])
    return np.hstack(cols)


def fit_lr(tr, C=1.0):
    sd = {c: float(max(tr[c].std(), 1e-6)) for c in LR_COLS}
    m = LogisticRegression(C=C, max_iter=3000, fit_intercept=False).fit(lr_design(tr, sd), tr.y); return m, sd


def lr_z(m, sd, d): return lr_design(d, sd) @ m.coef_[0]


GBM_PRM = dict(objective="binary", learning_rate=0.03, num_leaves=7, min_data_in_leaf=400, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
               lambda_l2=50, monotone_constraints=[MONO.get(c, 0) for c in FEATS], verbose=-1, seed=7)


def fit_gbm(tr, nround):
    return lgb.train(GBM_PRM, lgb.Dataset(tr[FEATS], tr.y), nround)


def main():
    D = load()
    if D.game.nunique() < MIN_GAMES: sys.exit("pas assez de games")
    D = D.sort_values("date").reset_index(drop=True); prior = lambda d: sg(d.lp)
    games = D.drop_duplicates("game").game.tolist(); rng = np.random.RandomState(3)
    fold = {g: i % 5 for i, g in enumerate(rng.permutation(games))}; D["fold"] = D.game.map(fold)
    # --- 1) nb d'arbres : early stopping sur un tiers des games d'apprentissage (fold 0 de validation)
    tr0, va0 = D[D.fold != 0], D[D.fold == 0]
    g0 = lgb.train(GBM_PRM, lgb.Dataset(tr0[FEATS], tr0.y), 3000, valid_sets=[lgb.Dataset(va0[FEATS], va0.y)], callbacks=[lgb.early_stopping(150, verbose=False)])
    nb = max(60, g0.best_iteration); print("arbres GBM :", nb)
    # --- 2) validation croisee 5 plis par GAME : LR, GBM, melange
    zl, zg = np.zeros(len(D)), np.zeros(len(D))
    for k in range(5):
        tr, te = D[D.fold != k], D[D.fold == k]
        m, sd = fit_lr(tr); zl[te.index] = lr_z(m, sd, te)
        zg[te.index] = fit_gbm(tr, nb).predict(te[FEATS], raw_score=True)
    y = D.y.values
    print("CV logloss : prior", round(ll(y, prior(D)), 4), "| LR", round(ll(y, sg(zl)), 4), "| GBM", round(ll(y, sg(zg)), 4))
    best = min(((ll(y, sg(w * zl + (1 - w) * zg)), w) for w in np.linspace(0, 1, 11)))
    mix = float(best[1]); zm = mix * zl + (1 - mix) * zg
    print(f"melange w_LR={mix:.1f} -> CV logloss {best[0]:.4f} | acc {((zm > 0) == y).mean():.4f} | prior acc {((D.lp > 0) == y).mean():.4f}")
    for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 90)]:
        m_ = ((D.tm >= lo) & (D.tm < hi)).values
        if m_.sum() > 50: print(f"  {lo:2d}-{hi:2d} min n={m_.sum():6d} prior {ll(y[m_], prior(D)[m_]):.4f} LR {ll(y[m_], sg(zl[m_])):.4f} GBM {ll(y[m_], sg(zg[m_])):.4f} mix {ll(y[m_], sg(zm[m_])):.4f} acc {((zm[m_] > 0) == y[m_]).mean():.3f}")
    q = pd.cut(pd.Series(sg(zm)), [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    print(pd.DataFrame({"q": q.values, "p": sg(zm), "y": y}).groupby("q", observed=True).agg(n=("y", "size"), annonce=("p", "mean"), observe=("y", "mean")).round(3).to_string())
    # --- 3) holdout temporel honnete : apprentissage sur les 80 % de games les plus anciens, test sur les 20 % recents
    cut = D.drop_duplicates("game").date.iloc[int(len(games) * .8)]; tr, te = D[D.date < cut], D[D.date >= cut]
    m, sd = fit_lr(tr); zl_t = lr_z(m, sd, te); zg_t = fit_gbm(tr, nb).predict(te[FEATS], raw_score=True); zm_t = mix * zl_t + (1 - mix) * zg_t
    print(f"holdout temporel (apres {str(cut)[:10]}, {te.game.nunique()} games) : prior {ll(te.y, prior(te)):.4f} LR {ll(te.y, sg(zl_t)):.4f} GBM {ll(te.y, sg(zg_t)):.4f} mix {ll(te.y, sg(zm_t)):.4f} | acc {((zm_t > 0) == te.y).mean():.4f} (prior {((te.lp > 0) == te.y).mean():.4f})")
    # --- 4) ajustement final sur toutes les games + export
    mF, sdF = fit_lr(D); gF = fit_gbm(D, nb); dump = gF.dump_model()
    def node(n):
        if "leaf_value" in n: return float(f"{n['leaf_value']:.7g}")
        return [n["split_feature"], float(f"{n['threshold']:.9g}"), node(n["left_child"]), node(n["right_child"])]
    W = mF.coef_[0]; nc = len(LR_COLS)
    out = {"version": 2, "feats": FEATS, "mix": mix, "taus": list(TAUS), "lr": {"cols": [FEATS.index(c) for c in LR_COLS], "sd": [sdF[c] for c in LR_COLS],
           "bias": [float(x) for x in W[:3]], "w": [[float(x) for x in W[3 + 3 * i:6 + 3 * i]] for i in range(nc)]},
           "trees": [] if mix >= 0.999 else [node(t["tree_structure"]) for t in dump["tree_info"]],      # le melange n'apporte rien -> pas d'arbres exportes "n_games": int(D.game.nunique()), "asof": str(D.date.max().date()),
           "metrics": {"logloss_test": ll(y, sg(zm)), "logloss_prior": ll(y, prior(D)), "acc_test": float(((zm > 0) == y).mean()), "acc_prior": float(((D.lp > 0) == y).mean())}}
    (ROOT / "web/public/model").mkdir(parents=True, exist_ok=True); path = ROOT / "web/public/model/live_v1.json"
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    # controle : evaluation "navigateur" (LR + parcours d'arbres) == modeles Python
    def ev(t, x):
        while isinstance(t, list): t = t[2] if x[t[0]] <= t[1] else t[3]
        return t
    xs = D[FEATS].values[:400]; e1, e2 = np.exp(-D.tm.values[:400] / TAUS[0]), np.exp(-D.tm.values[:400] / TAUS[1])
    zlj = np.array([out["lr"]["bias"][0] + out["lr"]["bias"][1] * a + out["lr"]["bias"][2] * b + sum((wi[0] + wi[1] * a + wi[2] * b) * x[ci] / s for wi, ci, s in zip(out["lr"]["w"], out["lr"]["cols"], out["lr"]["sd"])) for x, a, b in zip(xs, e1, e2)])
    zgj = np.array([sum(ev(t, x) for t in out["trees"]) for x in xs])
    print("ecart max export vs python : LR", float(np.abs(zlj - lr_z(mF, sdF, D.iloc[:400])).max()), "| GBM", float(np.abs(zgj - gF.predict(D[FEATS].iloc[:400], raw_score=True)).max()))
    print("exporte", path.name, path.stat().st_size // 1024, "Ko")


if __name__ == "__main__":
    main()
