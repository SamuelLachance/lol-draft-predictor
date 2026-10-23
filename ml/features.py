"""Variables CAUSALES par game (calculees AVANT le resultat) pour le modele v4.

Une seule passe chronologique : pour chaque game on lit l'etat (ratings, forme, meta, maitrise joueur-champion...),
on emet les variables, puis on met l'etat a jour avec le resultat. Rien du futur ne fuit.

Groupes de variables :
  equipe   : Elo sequentiel, forme (5/10 derniers, pondere), serie, repos, charge, confrontations directes (H2H)
  joueurs  : TrueSkill par ROLE (mu/sigma), ecart de niveau par role, meilleur/pire duel
  meta     : winrate champion x role en fenetre glissante (demi-vie 21 j, tier-1) et presence (priorite)
  draft    : attributs de composition (engage, CC, mobilite, poke, scaling, frontline, AD/AP, immobiles...)
  maitrise : historique joueur x champion (nb de games, winrate shrinke vers la meta du champion)
"""
from __future__ import annotations
import json, math
from collections import defaultdict, deque
from pathlib import Path
import numpy as np, pandas as pd
import trueskill as ts

ROLES = ["top", "jng", "mid", "bot", "sup"]
ATTR_KEYS = ["cc", "engage", "disengage", "mobility", "waveclear", "poke", "scaling", "tankiness", "sustain", "burst", "global", "anti_dive", "pick"]
HL_META = 21.0          # demi-vie (jours) du winrate/presence champion
K_SHR = 8.0
ENV = ts.TrueSkill(draw_probability=0.0)


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4); return math.log(p / (1 - p))


def wl(a, b):
    mu = sum(r.mu for r in a) - sum(r.mu for r in b)
    s2 = sum(r.sigma ** 2 for r in a) + sum(r.sigma ** 2 for r in b) + (len(a) + len(b)) * ENV.beta ** 2
    return 0.5 * (1 + math.erf(mu / math.sqrt(2 * s2)))


def load_attrs(path: Path) -> dict:
    d = json.load(open(path, encoding="utf-8"))["champions"]
    out = {}
    for c in d:
        out[c["name"]] = {**{k: float(c.get(k, 0)) for k in ATTR_KEYS}, "ad": 1.0 if c["damage"] == "ad" else 0.0,
                          "ap": 1.0 if c["damage"] == "ap" else 0.0, "imm": 1.0 if c["mobility"] == 0 else 0.0,
                          "front": 1.0 if c["tankiness"] >= 2 else 0.0}
    return out


def comp_attrs(champs, attrs):
    rows = [attrs[c] for c in champs if c in attrs]
    keys = ATTR_KEYS + ["ad", "ap", "imm", "front"]
    if len(rows) < 3:
        return {k: np.nan for k in keys}
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


class Decay:
    """compteur (victoires, games) a decroissance exponentielle."""
    __slots__ = ("w", "g", "t")

    def __init__(self):
        self.w = 0.0; self.g = 0.0; self.t = None

    def at(self, t):
        if self.t is None:
            return 0.0, 0.0
        f = 0.5 ** ((t - self.t) / HL_META)
        return self.w * f, self.g * f

    def add(self, t, win):
        w, g = self.at(t); self.w = w + win; self.g = g + 1; self.t = t


def build(dt: pd.DataFrame, pg: dict, attrs: dict, tier1: set, intl: set) -> pd.DataFrame:
    """dt : trie par date, colonnes gameid,date(_date),league,blue_team,red_team,blue_win,playoffs,blue_<role>,red_<role>.
    pg : gameid -> {"bp":[5 joueurs], "rp":[5 joueurs]}. Retourne un DataFrame (une ligne par game) des variables."""
    n = len(dt)
    gid = dt.gameid.to_numpy(); y = dt.blue_win.to_numpy().astype(int)
    dates = dt._date.dt.to_pydatetime(); days = np.array([(d - dates[0]).total_seconds() / 86400 for d in dates])
    lg = dt.league.to_numpy(); po = dt.playoffs.fillna(0).to_numpy().astype(float)
    bt_ = dt.blue_team.astype(str).to_numpy(); rt_ = dt.red_team.astype(str).to_numpy()
    BP = {r: dt[f"blue_{r}"].astype(str).to_numpy() for r in ROLES}; RP = {r: dt[f"red_{r}"].astype(str).to_numpy() for r in ROLES}

    elo = defaultdict(lambda: 1500.0)
    form = defaultdict(lambda: deque(maxlen=10))          # (jour, resultat)
    last_day = {}; recent = defaultdict(lambda: deque(maxlen=40))
    h2h = defaultdict(list)
    PR = defaultdict(lambda: ENV.create_rating())
    pc = defaultdict(lambda: [0, 0])                      # (joueur,champion) -> [games, wins]
    pgames = defaultdict(int)
    meta = defaultdict(Decay); tot = Decay()
    rows = []
    for i in range(n):
        t = days[i]; a, b = bt_[i], rt_[i]; f = {}
        # --- Elo
        ea, eb = elo[a], elo[b]; pe = 1 / (1 + 10 ** ((eb - ea) / 400)); f["elo"] = logit(pe)
        # --- forme / repos / charge
        for key, tm in (("b", a), ("r", b)):
            fm = form[tm]; res = [r for _, r in fm]
            f[f"{key}_f5"] = np.mean(res[-5:]) if res else np.nan
            f[f"{key}_f10"] = np.mean(res) if res else np.nan
            if res:
                w = np.array([0.85 ** k for k in range(len(res))][::-1]); f[f"{key}_fw"] = float((np.array(res) * w).sum() / w.sum())
            else:
                f[f"{key}_fw"] = np.nan
            streak = 0
            for r in res[::-1]:
                if (r == res[-1]):
                    streak += 1
                else:
                    break
            f[f"{key}_streak"] = (streak if res and res[-1] == 1 else -streak) if res else 0
            f[f"{key}_rest"] = min(30.0, t - last_day[tm]) if tm in last_day else 30.0
            f[f"{key}_load7"] = sum(1 for d in recent[tm] if t - d <= 7)
            f[f"{key}_ngames"] = len(recent[tm])
        for k in ("f5", "f10", "fw", "streak", "rest", "load7"):
            f[f"d_{k}"] = f[f"b_{k}"] - f[f"r_{k}"] if not (np.isnan(f[f"b_{k}"]) or np.isnan(f[f"r_{k}"])) else np.nan
        # --- H2H (2 ans)
        hk = frozenset((a, b)); hh = [(d, w) for d, w in h2h[hk] if t - d <= 730]
        wa = sum(1 for d, w in hh if w == a)
        f["h2h_n"] = len(hh); f["h2h"] = (wa + 1) / (len(hh) + 2) - 0.5
        # --- joueurs : TrueSkill par role
        pp = pg.get(gid[i]); bl = [BP[r][i] for r in ROLES]; rl = [RP[r][i] for r in ROLES]
        pb = pr = None
        if pp and all(pp["bp"]) and all(pp["rp"]):
            pb = [PR[p] for p in pp["bp"]]; pr = [PR[p] for p in pp["rp"]]
            f["ts"] = logit(wl(pb, pr))
            for k, r in enumerate(ROLES):
                f[f"ts_{r}"] = pb[k].mu - pr[k].mu
            dd = [pb[k].mu - pr[k].mu for k in range(5)]; f["ts_max"] = max(dd); f["ts_min"] = min(dd)
            f["ts_sig"] = float(sum(x.sigma for x in pb) - sum(x.sigma for x in pr))
            f["ts_gp_b"] = float(np.mean([pgames[p] for p in pp["bp"]])); f["ts_gp_r"] = float(np.mean([pgames[p] for p in pp["rp"]]))
        else:
            f["ts"] = 0.0
            for r in ROLES: f[f"ts_{r}"] = np.nan
            f["ts_max"] = f["ts_min"] = f["ts_sig"] = f["ts_gp_b"] = f["ts_gp_r"] = np.nan
        # --- meta (champion x role, fenetre glissante) + maitrise joueur-champion + attributs de compo
        tw, tg = tot.at(t); tg = max(tg, 1.0)
        mb, mr, pb_, pr_, ub, ur, cb, cr = [], [], [], [], [], [], [], []
        for k, r in enumerate(ROLES):
            for side, ch, pl, m_, p_, u_, c_ in (("b", BP[r][i], pp["bp"][k] if pp else None, mb, pb_, ub, cb), ("r", RP[r][i], pp["rp"][k] if pp else None, mr, pr_, ur, cr)):
                w_, g_ = meta[(r, ch)].at(t)
                wr = (w_ + K_SHR * 0.5) / (g_ + K_SHR); m_.append(logit(wr)); u_.append(g_ / tg)      # niveau meta + presence
                if pl:
                    gg, ww = pc[(pl, ch)]
                    p_.append(logit((ww + 4 * wr) / (gg + 4))); c_.append(math.log1p(gg))
        f["meta_wr"] = float(np.mean(mb) - np.mean(mr)); f["meta_pres"] = float(np.sum(ub) - np.sum(ur))
        f["prof"] = float(np.mean(pb_) - np.mean(pr_)) if pb_ and pr_ else np.nan
        f["comfort"] = float(np.mean(cb) - np.mean(cr)) if cb and cr else np.nan
        f["comfort_min"] = float(min(cb) - min(cr)) if cb and cr else np.nan
        ab, ar = comp_attrs([BP[r][i] for r in ROLES], attrs), comp_attrs([RP[r][i] for r in ROLES], attrs)
        for k in ab: f[f"c_{k}"] = ab[k] - ar[k]
        f["c_ad_bal_b"] = abs(ab["ad"] - .5) if not np.isnan(ab["ad"]) else np.nan; f["c_ad_bal_r"] = abs(ar["ad"] - .5) if not np.isnan(ar["ad"]) else np.nan
        # --- contexte
        f["playoffs"] = po[i]; f["intl"] = 1.0 if lg[i] in intl else 0.0
        f["y"] = y[i]; f["gameid"] = gid[i]
        rows.append(f)
        # ================= mise a jour avec le resultat =================
        w_b = y[i]; elo[a] = ea + 30 * (w_b - pe); elo[b] = eb + 30 * ((1 - w_b) - (1 - pe))
        form[a].append((t, w_b)); form[b].append((t, 1 - w_b)); last_day[a] = last_day[b] = t; recent[a].append(t); recent[b].append(t)
        h2h[hk].append((t, a if w_b else b))
        if pb is not None:
            nb, nr = ENV.rate([pb, pr], ranks=[0, 1] if w_b else [1, 0])
            for p, x in zip(pp["bp"], nb): PR[p] = x
            for p, x in zip(pp["rp"], nr): PR[p] = x
            for p in pp["bp"] + pp["rp"]: pgames[p] += 1
        if pp and all(pp["bp"]) and all(pp["rp"]):
            for k, r in enumerate(ROLES):
                pc[(pp["bp"][k], BP[r][i])][0] += 1; pc[(pp["bp"][k], BP[r][i])][1] += w_b
                pc[(pp["rp"][k], RP[r][i])][0] += 1; pc[(pp["rp"][k], RP[r][i])][1] += 1 - w_b
        if lg[i] in tier1:
            for r in ROLES:
                meta[(r, BP[r][i])].add(t, w_b); meta[(r, RP[r][i])].add(t, 1 - w_b)
            tot.add(t, 0)
    F = pd.DataFrame(rows)
    F["date"] = dt._date.values; F["league"] = lg; F["blue"] = bt_; F["red"] = rt_
    return F
