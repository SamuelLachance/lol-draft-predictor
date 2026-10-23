"""Pages de match Worlds : analyse par match (probas, enjeux Swiss, champions contestes, plan de jeu).

Entrees (toutes dans web/public/ ou data/, donc rejouable toutes les ~10 min en CI) :
  data/worlds_schedule.json  calendrier/resultats (ml/fetch_rich.py)
  data/games_rich.parquet    games pro avec picks/bans/objectifs
  web/public/data/{teams,players,meta}.json, web/public/model/draft_v2.json, web/public/draft/data/champions.json
Sortie : web/public/data/worlds.json  (+ live.json = version legere pour le polling navigateur)

    python ml/build_worlds.py
"""
from __future__ import annotations
import json, math, random, re, sys, unicodedata
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as st, draftmodel as dm

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "public"
DATA = WEB / "data"
ROLES = ["top", "jng", "mid", "bot", "sup"]
RL = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support"}
SINCE = pd.Timestamp("2026-03-01")           # fenetre des profils d'equipe
HALF = 75.0                                   # demi-vie (jours) du poids de recence des profils
K_LIVE = 0.10                                 # ajustement du rating (logit) par game jouee depuis le dernier build complet
N_SIM, N_COND = 4000, 1500
RNG = random.Random(26)


def T(fr, en):
    return {"fr": fr, "en": en}


def sig(z):
    return 1 / (1 + math.exp(-z))


def norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).casefold()
    return re.sub(r"[^a-z0-9]", "", s)


def jl(x):
    return json.loads(x) if isinstance(x, str) else list(x)


def pname(x):
    """'Kratos (Ngo Duc Khanh)' -> 'kratos' ; compare insensible a la casse/accents."""
    return norm(re.sub(r"\(.*?\)", "", str(x)))


def series_p(p, bo):
    if bo == 1:
        return p
    need = bo // 2 + 1
    q = 1 - p
    return sum(math.comb(need - 1 + k, k) * p ** need * q ** k for k in range(need))


# ------------------------------------------------------------------ chargement
def load():
    teams = json.load(open(DATA / "teams.json", encoding="utf-8"))
    players = json.load(open(DATA / "players.json", encoding="utf-8"))
    meta = json.load(open(DATA / "meta.json", encoding="utf-8"))
    dv = json.load(open(WEB / "model" / "draft_v2.json", encoding="utf-8"))
    attrs = {c["name"]: c for c in json.load(open(WEB / "draft" / "data" / "champions.json", encoding="utf-8"))["champions"]}
    sched = json.load(open(ROOT / "data" / "worlds_schedule.json", encoding="utf-8"))
    g = pd.read_parquet(ROOT / "data" / "games_rich.parquet")
    for c in ("blue_picks", "red_picks", "blue_bans", "red_bans", "blue_players", "red_players"):
        g[c] = g[c].map(jl)
    g["blue"] = g.blue.map(st.alias); g["red"] = g.red.map(st.alias)
    return teams, players, meta, dv, attrs, sched, g


def match_team(name, tidx_norm):
    """nom du calendrier -> nom de equipe du modele (alias + comparaison normalisee)."""
    name = st.alias(name)
    k = norm(name)
    if k in tidx_norm:
        return tidx_norm[k]
    for kk, v in tidx_norm.items():              # ex. 'los' <-> 'løs' : prefixe/inclusion
        if k and (kk.startswith(k) or k.startswith(kk)) and min(len(k), len(kk)) >= 3:
            return v
    return name


# ------------------------------------------------------------------ profils d'equipe
def team_games(g, team):
    """toutes les games (depuis SINCE) de `team`, orientees : own/opp picks, bans, joueurs, objectifs."""
    rows = []
    for r in g[(g.date >= SINCE) & ((g.blue == team) | (g.red == team))].itertuples():
        me = "blue" if r.blue == team else "red"
        op = "red" if me == "blue" else "blue"
        gg = lambda k, s: getattr(r, f"{s}_{k}")
        rows.append({"date": r.date, "side": me, "win": int(r.blue_win == (me == "blue")), "opp": getattr(r, op),
                     "league": r.league, "length": r.length, "tour": r.tournament,
                     "picks": gg("picks", me), "opicks": gg("picks", op), "bans": gg("bans", me), "obans": gg("bans", op),
                     "players": gg("players", me), "oplayers": gg("players", op),
                     "dragons": gg("dragons", me), "odragons": gg("dragons", op), "barons": gg("barons", me),
                     "obarons": gg("barons", op), "towers": gg("towers", me), "otowers": gg("towers", op),
                     "heralds": gg("riftheralds", me), "oheralds": gg("riftheralds", op),
                     "kills": gg("kills", me), "okills": gg("kills", op), "gold": gg("gold", me), "ogold": gg("gold", op)})
    return rows


def wavg(vals, ws):
    ws = np.asarray(ws, float); v = np.asarray(vals, float); m = ~np.isnan(v)
    return float((v[m] * ws[m]).sum() / ws[m].sum()) if m.any() and ws[m].sum() > 0 else None


def style_of(champs, attrs):
    """scores d'identite (meme logique que le draft assistant) -> allin / poke / hyper (+ AD/AP, mobilite)."""
    allin = poke = hyper = ad = ap = n = 0.0
    mob = eng = cc = 0.0
    for c in champs:
        a = attrs.get(c)
        if not a:
            continue
        n += 1
        allin += 2 * a["engage"] + (1.5 if a["family"] in ("engage", "catch") else 0) + (1 if a["family"] == "assassin" else 0) + .5 * a["cc"] * a["engage"] + .5 * a["pick"]
        poke += 1.5 * a["poke"] + a["disengage"] + (1.5 if a["family"] in ("poke", "disengage") else 0) + (.7 if a["family"] == "enchanter" else 0) + .5 * a["anti_dive"] - a["engage"]
        hyper += (3 if a["family"] == "hypercarry" else 0) + (2 if a["family"] == "enchanter" else 0) + .4 * a["scaling"] + .3 * a["waveclear"] + .5 * a["global"]
        ad += a["damage"] == "ad"; ap += a["damage"] == "ap"; mob += a["mobility"]; eng += a["engage"]; cc += a["cc"]
    if not n:
        return None
    return {"allin": allin / n, "poke": poke / n, "hyper": hyper / n, "ad": ad / n, "ap": ap / n, "mob": mob / n, "engage": eng / n, "cc": cc / n, "n": int(n)}


def build_profile(team, g, players_by_team, attrs, field):
    rows = team_games(g, team)
    if len(rows) < 6:                                  # equipe peu vue : elargit la fenetre
        rows = []
        for r in g[(g.blue == team) | (g.red == team)].tail(40).itertuples():
            pass
        # repli simple : toutes dates
        global SINCE
        old = SINCE; SINCE = pd.Timestamp("2025-01-01"); rows = team_games(g, team); SINCE = old
    if not rows:
        return None
    now = g.date.max()
    w = np.array([0.5 ** ((now - r["date"]).days / HALF) for r in rows])
    n = len(rows); wins = sum(r["win"] for r in rows)
    P = {"games": n, "wins": wins, "wr": wins / n}
    # side
    for s in ("blue", "red"):
        rs = [r for r in rows if r["side"] == s]
        P[f"{s}_g"] = len(rs); P[f"{s}_w"] = sum(r["win"] for r in rs)
    # joueurs par role (nom le plus frequent recent) + picks par role
    role_stats = {}
    for i, ro in enumerate(ROLES):
        cnt = defaultdict(lambda: [0.0, 0, 0.0])      # champion -> [poids, games, wins]
        pc = Counter()
        for r, wt in zip(rows, w):
            try:
                ch = r["picks"][i]; pl = r["players"][i]
            except IndexError:
                continue
            e = cnt[ch]; e[0] += wt; e[1] += 1; e[2] += r["win"]; pc[pname(pl)] += wt
        tot = sum(e[0] for e in cnt.values()) or 1
        pool = sorted(cnt.items(), key=lambda kv: -kv[1][0])
        role_stats[ro] = {"pool": [{"champion": c, "w": round(e[0] / tot, 3), "games": e[1], "wins": int(e[2])} for c, e in pool[:8]],
                          "unique": len(cnt), "top3": round(sum(e[0] for _, e in pool[:3]) / tot, 3)}
    P["roles"] = role_stats
    # bans faits / subis ; phase 1 = 3 premiers bans
    bm, bm1, bs = Counter(), Counter(), Counter()
    for r, wt in zip(rows, w):
        for k, c in enumerate(r["bans"]):
            bm[c] += wt
            if k < 3: bm1[c] += wt
        for c in r["obans"]:
            bs[c] += wt
    W = w.sum()
    P["bans_made"] = [{"champion": c, "rate": round(v / W, 3)} for c, v in bm.most_common(10)]
    P["bans_p1"] = [{"champion": c, "rate": round(v / W, 3)} for c, v in bm1.most_common(8)]
    P["bans_against"] = [{"champion": c, "rate": round(v / W, 3)} for c, v in bs.most_common(10)]
    # tempo / objectifs
    L = [r["length"] / 60 if r["length"] else None for r in rows]
    P["len"] = wavg(L, w)
    P["kpm"] = wavg([(r["kills"] + r["okills"]) / (r["length"] / 60) if r["length"] else None for r in rows], w)
    P["dragons"] = wavg([r["dragons"] for r in rows], w); P["heralds"] = wavg([r["heralds"] for r in rows], w)
    P["barons"] = wavg([r["barons"] for r in rows], w); P["towers"] = wavg([r["towers"] for r in rows], w)
    P["gold_share"] = wavg([r["gold"] / (r["gold"] + r["ogold"]) if r["gold"] and r["ogold"] else None for r in rows], w)
    P["dragon_share"] = wavg([r["dragons"] / (r["dragons"] + r["odragons"]) if (r["dragons"] + r["odragons"]) else None for r in rows], w)
    short = [r for r in rows if r["length"] and r["length"] < 30 * 60]; longg = [r for r in rows if r["length"] and r["length"] > 34 * 60]
    P["short"] = {"g": len(short), "w": sum(r["win"] for r in short)}; P["long"] = {"g": len(longg), "w": sum(r["win"] for r in longg)}
    # style de compo
    allc = [c for r in rows for c in r["picks"]]
    P["style"] = style_of(allc, attrs)
    P["recent"] = [{"date": str(r["date"])[:10], "opp": r["opp"], "win": r["win"], "side": r["side"], "tour": r["tour"]} for r in rows[-8:][::-1]]
    P["since"] = str(rows[0]["date"])[:10]
    return P


def identity_bullets(team, P, field, tm):
    """points forts / plan de jeu en langage naturel (FR/EN) a partir des ecarts vs les equipes de Worlds."""
    out = []
    def z(k):
        vals = [f[k] for f in field if f.get(k) is not None]
        if len(vals) < 4 or P.get(k) is None:
            return 0, None
        mu, sd = float(np.mean(vals)), float(np.std(vals) or 1)
        return (P[k] - mu) / sd, mu
    zl, mu = z("len")
    if P.get("len") and abs(zl) > .6:
        d = P["len"] - mu
        out.append(("tempo", T(f"{'Parties rapides' if d<0 else 'Parties longues'} : {P['len']:.1f} min en moyenne ({abs(d):.1f} min {'de moins' if d<0 else 'de plus'} que la moyenne des équipes de Worlds) — {'cherche à fermer vite' if d<0 else 'joue la montée en puissance'}.",
                               f"{'Fast games' if d<0 else 'Long games'}: {P['len']:.1f} min on average ({abs(d):.1f} min {'shorter' if d<0 else 'longer'} than the Worlds field) — {'looks to close fast' if d<0 else 'plays for scaling'}.")))
    zk, mk = z("kpm")
    if P.get("kpm") and abs(zk) > .6:
        out.append(("fights", T(f"{'Beaucoup de combats' if zk>0 else 'Jeu contrôlé, peu de kills'} : {P['kpm']:.2f} kills/min (moyenne {mk:.2f}).",
                                 f"{'Very fight-heavy' if zk>0 else 'Controlled, low-kill style'}: {P['kpm']:.2f} kills/min (field {mk:.2f}).")))
    zd, md = z("dragon_share")
    if P.get("dragon_share") is not None and abs(zd) > .7:
        out.append(("obj", T(f"{'Priorité aux dragons' if zd>0 else 'Cède souvent les dragons'} : {P['dragon_share']*100:.0f}% des dragons pris (moyenne {md*100:.0f}%).",
                              f"{'Dragon priority' if zd>0 else 'Often concedes dragons'}: {P['dragon_share']*100:.0f}% of dragons taken (field {md*100:.0f}%).")))
    zb, mb = z("barons")
    if P.get("barons") is not None and abs(zb) > .8:
        out.append(("baron", T(f"{'Joue le Baron' if zb>0 else 'Baron rarement pris'} : {P['barons']:.2f} par game.", f"{'Baron-focused' if zb>0 else 'Rarely takes Baron'}: {P['barons']:.2f} per game.")))
    zg, mg = z("gold_share")
    if P.get("gold_share") and zg > .7:
        out.append(("lead", T(f"Prend l'avantage économique : {P['gold_share']*100:.1f}% de l'or total.", f"Builds economic leads: {P['gold_share']*100:.1f}% of total gold.")))
    # side
    bg, rg = P["blue_g"], P["red_g"]
    if bg >= 8 and rg >= 8:
        bw, rw = P["blue_w"] / bg, P["red_w"] / rg
        if abs(bw - rw) > .15:
            s = "bleu" if bw > rw else "rouge"; se = "Blue" if bw > rw else "Red"
            out.append(("side", T(f"Préfère le side {s} : {max(bw,rw)*100:.0f}% de victoires contre {min(bw,rw)*100:.0f}% de l'autre côté.",
                                   f"Prefers {se} side: {max(bw,rw)*100:.0f}% win rate vs {min(bw,rw)*100:.0f}% on the other.")))
    # short/long
    sh, lo = P["short"], P["long"]
    if sh["g"] >= 8 and lo["g"] >= 8:
        a, b = sh["w"] / sh["g"], lo["w"] / lo["g"]
        if abs(a - b) > .15:
            out.append(("sl", T(f"{'Domine les parties courtes (<30 min)' if a>b else 'Domine les parties longues (>34 min)'} : {max(a,b)*100:.0f}% de WR contre {min(a,b)*100:.0f}% dans l'autre cas.",
                                 f"{'Excels in short games (<30 min)' if a>b else 'Excels in long games (>34 min)'}: {max(a,b)*100:.0f}% WR vs {min(a,b)*100:.0f}% otherwise.")))
    # pools
    narrow = [(ro, s["top3"]) for ro, s in P["roles"].items() if s["top3"] > .8 and s["unique"] >= 3]
    if narrow:
        ro, v = max(narrow, key=lambda x: x[1])
        out.append(("pool", T(f"Pool très concentré au {RL[ro]} : 3 champions = {v*100:.0f}% de ses picks (prévisible).", f"Very narrow {RL[ro]} pool: 3 champions = {v*100:.0f}% of picks (predictable).")))
    wide = [(ro, s["unique"]) for ro, s in P["roles"].items() if s["unique"] >= 14]
    if wide:
        ro, v = max(wide, key=lambda x: x[1])
        out.append(("flex", T(f"Large pool au {RL[ro]} : {v} champions différents depuis mars (difficile à cibler en ban).", f"Wide {RL[ro]} pool: {v} different champions since March (hard to target with bans).")))
    st_ = P.get("style")
    if st_:
        sc = {"allin": st_["allin"], "poke": st_["poke"], "hyper": st_["hyper"]}
        top = max(sc, key=sc.get)
        lab = {"allin": T("engage / all-in", "engage / all-in"), "poke": T("poke / contrôle", "poke / control"), "hyper": T("scaling / hypercarry", "scaling / hypercarry")}[top]
        out.append(("comp", T(f"Compositions dominantes : {lab['fr']} (AD {st_['ad']*100:.0f}% / AP {st_['ap']*100:.0f}%).", f"Dominant comp style: {lab['en']} (AD {st_['ad']*100:.0f}% / AP {st_['ap']*100:.0f}%).")))
    return [{"k": k, "t": t} for k, t in out]


# ------------------------------------------------------------------ proba de match
def team_logit(t, tm, adj):
    return tm[t]["bt"] + adj.get(t, 0.0)


def ts_logit(a, b, tm):
    A, B = tm[a], tm[b]
    if A.get("ts_mu") is None or B.get("ts_mu") is None:
        return 0.0
    BETA = 25 / 6
    mu = A["ts_mu"] - B["ts_mu"]; s2 = A["ts_sig"] ** 2 + B["ts_sig"] ** 2 + 10 * BETA ** 2
    p = 0.5 * (1 + math.erf(mu / math.sqrt(2 * s2))); p = min(.999, max(.001, p))
    return math.log(p / (1 - p))


def game_p(a, b, tm, adj, meta):
    """P(a bat b) en un game, symetrisee sur le side (a bleu / a rouge)."""
    w = meta["blend"]["pre"]; side = meta["blend"]["bt_side"]
    d = team_logit(a, tm, adj) - team_logit(b, tm, adj); t = ts_logit(a, b, tm)
    p_blue = sig(w["w0"] + w["bt"] * (d + side) + w["ts_logit"] * t)
    p_red = sig(w["w0"] + w["bt"] * (-d + side) + w["ts_logit"] * -t)
    return 0.5 * (p_blue + 1 - p_red), p_blue, 1 - p_red


# ------------------------------------------------------------------ simulation Swiss (regles OFFICIELLES)
# Riot : R1 tirage par tiers (deja fait, connu via le calendrier). Des le R2 : adversaire tire AU HASARD parmi les
# equipes du MEME BILAN, jamais deux fois le meme adversaire (meme region autorise). 3 victoires = qualifie, 3 defaites = elimine.
# Bo3 des qu'une equipe a 2 victoires ou 2 defaites (verifie sur le calendrier officiel), sinon Bo1.
# Quarts (tirage du 31/10) : les deux 3-0 contre deux des 3-2 tires au sort ; les 4 autres equipes tirees au sort entre elles.
def bo_for(w1, l1, w2, l2):
    return 3 if (max(w1, w2) == 2 or max(l1, l2) == 2) else 1


def draw_pairs(grp, played, rng):
    """appariement uniforme parmi les perfect matchings sans revanche (rejet puis backtracking) ; None si impossible."""
    for _ in range(200):
        g = grp[:]; rng.shuffle(g)
        ps = [(g[i], g[i + 1]) for i in range(0, len(g) - 1, 2)]
        if not any(frozenset(p) in played for p in ps):
            return ps
    def rec(rest):
        if not rest:
            return []
        a, others = rest[0], rest[1:]
        cands = [b for b in others if frozenset((a, b)) not in played]; rng.shuffle(cands)
        for b in cands:
            r = rec([x for x in others if x != b])
            if r is not None:
                return [(a, b)] + r
        return None
    g = grp[:]; rng.shuffle(g)
    return rec(g)


def simulate_swiss(teams16, results, fixtures_by_round, pf, n_sims, force=None, rng=RNG, past=None):
    """fixtures_by_round[N] = [(a,b,bo,winner|None)] pour les matchs DEJA connus (joues ou programmes).
    past = ensemble de frozenset((a,b)) deja joues (anti-revanche). force=(a,b,winner) impose un match a venir."""
    qual = Counter(); rec_dist = defaultdict(Counter); qualified_lists = []
    for _ in range(n_sims):
        W = {t: 0 for t in teams16}; Lc = {t: 0 for t in teams16}; played = set()
        for rnd in range(1, 6):
            done = set(); pairs = []
            for a, b, bo, win in fixtures_by_round.get(rnd, []):
                if a not in W or b not in W or W[a] >= 3 or Lc[a] >= 3 or W[b] >= 3 or Lc[b] >= 3:
                    continue
                pairs.append((a, b, bo, win)); done.add(a); done.add(b)
            rest = [t for t in teams16 if t not in done and W[t] < 3 and Lc[t] < 3]
            groups = defaultdict(list)
            for t in rest: groups[(W[t], Lc[t])].append(t)
            for grp in groups.values():
                ps = draw_pairs(grp, played, rng)
                if ps is None:                                   # etat incoherent : on ignore l'anti-revanche
                    g_ = grp[:]; rng.shuffle(g_); ps = [(g_[i], g_[i + 1]) for i in range(0, len(g_) - 1, 2)]
                for a, b in ps:
                    pairs.append((a, b, bo_for(W[a], Lc[a], W[b], Lc[b]), None))
            for a, b, bo, win in pairs:
                if win is None:
                    if force and {force[0], force[1]} == {a, b}:
                        win = force[2]
                    else:
                        win = a if rng.random() < pf(a, b, bo) else b
                lo = b if win == a else a
                W[win] += 1; Lc[lo] += 1; played.add(frozenset((a, b)))
        q = [(t, Lc[t]) for t in teams16 if W[t] >= 3]
        qualified_lists.append(q)
        for t in teams16:
            rec_dist[t][(W[t], Lc[t])] += 1
            if W[t] >= 3: qual[t] += 1
    return {"qual": {t: qual[t] / n_sims for t in teams16},
            "rec": {t: {f"{w}-{l}": c / n_sims for (w, l), c in d.items()} for t, d in rec_dist.items()},
            "qualified_lists": qualified_lists}


def knockout_odds(qlists, pf5, rng=RNG):
    """cotes de titre avec le tirage OFFICIEL des quarts ; demis : appariement des 4 vainqueurs tire au hasard (regle non publiee)."""
    title = Counter(); sf = Counter(); n = 0
    for q in qlists:
        if len(q) != 8:
            continue
        n += 1
        z = [t for t, l in q if l == 0]; two = [t for t, l in q if l == 2]; one = [t for t, l in q if l == 1]
        play = lambda a, b: a if rng.random() < pf5(a, b) else b
        if len(z) == 2 and len(two) == 3 and len(one) == 3:
            rng.shuffle(two); m = [(z[0], two[0]), (z[1], two[1])]
            rest = one + [two[2]]; rng.shuffle(rest); m += [(rest[0], rest[1]), (rest[2], rest[3])]
        else:
            ts_ = [t for t, _ in q]; rng.shuffle(ts_); m = [(ts_[i], ts_[i + 1]) for i in range(0, 8, 2)]
        w4 = [play(a, b) for a, b in m]
        for t in w4: sf[t] += 1
        rng.shuffle(w4)
        f = [play(w4[0], w4[1]), play(w4[2], w4[3])]
        title[play(f[0], f[1])] += 1
    return {t: title[t] / n for t in title} if n else {}, {t: sf[t] / n for t in sf} if n else {}


# ------------------------------------------------------------------ principal
def slug_of(mid, stage):
    m = re.search(r"(Play-In|Main Event)_(.+)_(\d+)$", str(mid or ""))
    if not m:
        return None
    rd = m.group(2)
    short = {"Quarterfinals": "qf", "Semifinals": "sf", "Finals": "f"}.get(rd) or ("r" + re.sub(r"\D", "", rd))
    return f"{'pi' if m.group(1)=='Play-In' else 'me'}-{short}-{m.group(3)}"


def main():
    teams, players, meta, dv, attrs, sched, g = load()
    tm = {t["team"]: t for t in teams}
    tn = {norm(t): t for t in tm}
    pidx = defaultdict(dict)
    for p in players:
        pidx[p["team"]][p["role"]] = p
    data_through = pd.Timestamp(meta["data_through"])
    # rating "live" : ajustement logit par game Worlds jouee apres le dernier build complet
    adj = defaultdict(float)
    wg = g[(g.league == "WLDs") & (g.date > data_through + pd.Timedelta(days=1))].sort_values("date")
    for r in wg.itertuples():
        a, b = r.blue, r.red
        if a in tm and b in tm:
            p = game_p(a, b, tm, adj, meta)[1]
            d = K_LIVE * ((1 if r.blue_win else 0) - p)
            adj[a] += d; adj[b] -= d

    # equipes du Main Event (Swiss)
    me = [m for m in sched if m["stage"] == "Main Event"]
    for m in sched:
        m["t1"] = match_team(m["t1"], tn) if m["t1"] and m["t1"] != "TBD" else None
        m["t2"] = match_team(m["t2"], tn) if m["t2"] and m["t2"] != "TBD" else None
    r1 = [m for m in me if m["round"] == "Round 1"]
    swiss = sorted({t for m in r1 for t in (m["t1"], m["t2"]) if t})
    # champs : presence dans les games recentes (champion hors-meta = valeur attribuee ailleurs)
    recent_g = g[g.date >= g.date.max() - pd.Timedelta(days=75)]
    field_pres = Counter()
    for r in recent_g.itertuples():
        for c in set(r.blue_picks + r.red_picks + r.blue_bans + r.red_bans):
            field_pres[c] += 1
    nrec = max(1, len(recent_g))
    wl_g = g[g.league == "WLDs"]
    world_pres = Counter()
    for r in wl_g[wl_g.date >= "2026-10-01"].itertuples():
        for c in set(r.blue_picks + r.red_picks + r.blue_bans + r.red_bans):
            world_pres[c] += 1
    nworld = max(1, len(wl_g[wl_g.date >= "2026-10-01"]))

    # profils
    profiles = {}
    for t in set(swiss) | {x for m in sched for x in (m["t1"], m["t2"]) if x}:
        if t in tm:
            profiles[t] = build_profile(t, g, pidx, attrs, None)
    field = [p for p in profiles.values() if p]
    for t, P in list(profiles.items()):
        if not P:
            continue
        P["bullets"] = identity_bullets(t, P, field, tm)

    # records Swiss (matches Main Event termines)
    W = {t: 0 for t in swiss}; Lc = {t: 0 for t in swiss}
    rounds = defaultdict(list)
    for m in me:
        mt = re.search(r"Round (\d)", m["round"] or "")
        done = m["winner"] in (1, 2, "1", "2")
        winner = None
        if done and m["t1"] and m["t2"]:
            winner = m["t1"] if str(m["winner"]) == "1" else m["t2"]
            lo = m["t2"] if winner == m["t1"] else m["t1"]
            if winner in W: W[winner] += 1
            if lo in Lc: Lc[lo] += 1
        if mt and m["t1"] and m["t2"]:
            rounds[int(mt.group(1))].append((m["t1"], m["t2"], m["bo"], winner))

    def pf(a, b, bo):
        return series_p(game_p(a, b, tm, adj, meta)[0], bo)
    sim = simulate_swiss(swiss, None, rounds, pf, N_SIM) if len(swiss) == 16 else None
    # TODO ci-dessous : cotes de titre
    title, semis = ({}, {})
    if sim:
        pf5 = lambda a, b: series_p(game_p(a, b, tm, adj, meta)[0], 5)
        title, semis = knockout_odds(sim["qualified_lists"], pf5)

    # matchs
    out_matches = []
    gby = {k: v for k, v in g[g.league == "WLDs"].groupby("matchid")}
    w_post = meta["blend"]["post"]
    for m in sched:
        mid = m["id"]; slug = slug_of(mid, m["stage"])
        item = {"id": mid, "slug": slug, "stage": m["stage"], "round": m["round"], "date": m["date"], "bo": m["bo"],
                "t1": m["t1"], "t2": m["t2"], "s1": m["s1"], "s2": m["s2"], "winner": m["winner"], "stream": m["stream"]}
        a, b = m["t1"], m["t2"]
        done = m["winner"] in (1, 2, "1", "2")
        item["status"] = "completed" if done else "upcoming"
        if a in tm and b in tm:
            pg, pa_blue, pb_blue_inv = game_p(a, b, tm, adj, meta)
            ps = series_p(pg, m["bo"])
            A, B = tm[a], tm[b]
            wgt = meta["blend"]["pre"]
            dbt = team_logit(a, tm, adj) - team_logit(b, tm, adj)
            contrib_bt = wgt["bt"] * dbt
            contrib_ts = wgt["ts_logit"] * ts_logit(a, b, tm)
            def pts(x): return round((sig(x) - .5) * 100, 1)
            factors = [
                {"k": "power", "t": T("Force d'équipe (Elo calibré par ligue)", "Team strength (league-calibrated Elo)"), "a": A["power"], "b": B["power"], "pts": pts(contrib_bt)},
                {"k": "ts", "t": T("Niveau des rosters (TrueSkill joueurs)", "Roster level (player TrueSkill)"), "a": A.get("ts_mu"), "b": B.get("ts_mu"), "pts": pts(contrib_ts)},
                {"k": "league", "t": T("Force de la ligue", "League strength"), "a": f"{A['region']} ({A['lg_off']:+d})", "b": f"{B['region']} ({B['lg_off']:+d})", "pts": None},
                {"k": "form", "t": T("Forme (10 derniers matchs)", "Form (last 10)"), "a": A["form"], "b": B["form"], "pts": None},
            ]
            h2h = g[(((g.blue == a) & (g.red == b)) | ((g.blue == b) & (g.red == a)))]
            hw = int(sum(((r.blue == a) == bool(r.blue_win)) for r in h2h.itertuples()))
            item["pred"] = {"game": round(pg, 4), "series": round(ps, 4), "a_blue": round(pa_blue, 4), "a_red": round(pb_blue_inv, 4),
                            "adj": {"a": round(adj.get(a, 0), 3), "b": round(adj.get(b, 0), 3)}, "factors": factors,
                            "h2h": {"games": int(len(h2h)), "a_wins": hw, "since": "2025-01"}}
            # --- champions contestes ---
            Pa, Pb = profiles.get(a), profiles.get(b)
            if Pa and Pb:
                cand = Counter()
                def pres(P_, c):
                    return 0
                rows = []
                allc = set()
                for P_ in (Pa, Pb):
                    for ro in ROLES:
                        allc.update(x["champion"] for x in P_["roles"][ro]["pool"][:5])
                    allc.update(x["champion"] for x in P_["bans_made"][:8]); allc.update(x["champion"] for x in P_["bans_against"][:8])
                allc.update(c for c, _ in world_pres.most_common(12))
                def team_pres(team, c):
                    gs = [r for r in team_games(g, team)]
                    return gs
                cache = {}
                for t in (a, b):
                    cache[t] = team_games(g, t)
                for c in allc:
                    rec = {"champion": c}
                    for key, t in (("a", a), ("b", b)):
                        gs = cache[t]; n = max(1, len(gs))
                        pk = [r for r in gs if c in r["picks"]]
                        bn = sum(c in r["bans"] for r in gs); ba = sum(c in r["obans"] for r in gs)
                        against = [r for r in gs if c in r["opicks"]]
                        rec[key] = {"picks": len(pk), "wins": sum(r["win"] for r in pk), "bans": bn, "banned_vs": ba, "faced": len(against),
                                    "n": n, "role": Counter(ROLES[[x for x in range(5)][r["picks"].index(c)]] for r in pk).most_common(1)[0][0] if pk else None}
                        # joueur principal
                        if pk:
                            pc = Counter(pname(r["players"][r["picks"].index(c)]) for r in pk)
                            rec[key]["player"] = pc.most_common(1)[0][0]
                            roster = {pname(x["player"]): x["player"] for x in (tm[t]["roster"] or [])}
                            rec[key]["player"] = roster.get(rec[key]["player"], rec[key]["player"])
                    fp = field_pres[c] / nrec; wp_ = world_pres[c] / nworld
                    pa = (rec["a"]["picks"] + rec["a"]["bans"] + rec["a"]["banned_vs"] + rec["a"]["faced"]) / (rec["a"]["n"] + 8) + 0.0
                    # presence conjointe : proba que c apparaisse (pick ou ban) dans une game de A ou de B (borne par la presence meta)
                    shrinkA = (rec["a"]["picks"] + rec["a"]["bans"] + rec["a"]["banned_vs"] + rec["a"]["faced"] + 8 * fp) / (rec["a"]["n"] + 8)
                    shrinkB = (rec["b"]["picks"] + rec["b"]["bans"] + rec["b"]["banned_vs"] + rec["b"]["faced"] + 8 * fp) / (rec["b"]["n"] + 8)
                    rec["presence"] = round(min(.99, (shrinkA + shrinkB) / 2), 3)
                    rec["field"] = round(fp, 3); rec["worlds"] = round(wp_, 3)
                    ctr = (rec["a"]["picks"] / rec["a"]["n"] + rec["b"]["picks"] / rec["b"]["n"]) / 2
                    rec["score"] = round(rec["presence"] * 0.8 + wp_ * 0.4, 3)
                    # contesté par les deux = le champion apparait (pick/ban/banni contre) dans >=35% des games de CHAQUE equipe
                    inv = lambda r_: (r_["picks"] + r_["bans"] + r_["banned_vs"]) / r_["n"]
                    rec["a"]["inv"] = round(inv(rec["a"]), 3); rec["b"]["inv"] = round(inv(rec["b"]), 3)
                    rec["both"] = bool(rec["a"]["inv"] >= .35 and rec["b"]["inv"] >= .35)
                    rows.append(rec)
                rows.sort(key=lambda r: -r["score"])
                item["contested"] = rows[:16]
                # --- bans probables : habitudes + (picks signature de l'adversaire) ---
                def ban_targets(me_, op_, Pme, Pop):
                    sc = Counter()
                    for x in Pme["bans_made"]: sc[x["champion"]] += 0.6 * x["rate"]
                    for ro in ROLES:
                        for x in Pop["roles"][ro]["pool"][:3]:
                            sc[x["champion"]] += 0.5 * x["w"] * (1 if x["games"] >= 3 else .4) * (Pop["roles"][ro]["top3"])
                    for x in Pop["bans_against"][:6]: sc[x["champion"]] += 0.25 * x["rate"]
                    tot = sum(sc.values()) or 1
                    res = []
                    for c, v in sc.most_common(8):
                        res.append({"champion": c, "p": round(min(.95, 5 * v / tot), 3)})
                    return res
                item["likely_bans"] = {"a": ban_targets(a, b, Pa, Pb), "b": ban_targets(b, a, Pb, Pa)}
                # --- picks probables par role ---
                lp = {}
                for key, P_, t in (("a", Pa, a), ("b", Pb, b)):
                    lp[key] = {}
                    for ro in ROLES:
                        pl_ = pidx[t].get(ro)
                        ros = next((x["player"] for x in tm[t]["roster"] if x["role"] == ro), None)
                        lp[key][ro] = {"player": ros, "pool": P_["roles"][ro]["pool"][:4], "unique": P_["roles"][ro]["unique"]}
                item["likely_picks"] = lp
                # --- duels de roles ---
                duels = []
                for ro in ROLES:
                    pa_, pb_ = pidx[a].get(ro), pidx[b].get(ro)
                    if pa_ and pb_:
                        duels.append({"role": ro, "a": {k: pa_[k] for k in ("player", "rating", "kda", "kp", "gd15", "wr", "games", "dmg")},
                                      "b": {k: pb_[k] for k in ("player", "rating", "kda", "kp", "gd15", "wr", "games", "dmg")},
                                      "edge": round(pa_["rating"] - pb_["rating"], 1)})
                item["duels"] = duels
            # enjeux Swiss
            if m["stage"] == "Main Event" and sim and a in swiss and b in swiss and not done and m["round"].startswith("Round"):
                wa, la, wb, lb = W[a], Lc[a], W[b], Lc[b]
                item["stakes"] = {"records": {"a": f"{wa}-{la}", "b": f"{wb}-{lb}"}, "qual_now": {"a": round(sim["qual"][a], 3), "b": round(sim["qual"][b], 3)},
                                  "decisive": bool(wa == 2 or wb == 2 or la == 2 or lb == 2)}
        # games joues
        gs = gby.get(mid)
        item["games"] = []
        if gs is not None:
            for r in gs.sort_values("game_n").itertuples():
                keys = dm.game_keys(r.blue_picks, r.red_picks)
                edge = 0.0; contrib = []
                for k, sgn in keys:
                    kind = k[0]; kk = k[2:]
                    v = dv.get(kind, {}).get(kk)
                    if v is None:
                        continue
                    edge += sgn * v
                    if kind == "c":
                        contrib.append({"champion": kk.split("|", 1)[1], "role": kk.split("|")[0], "side": "blue" if sgn > 0 else "red", "v": round(sgn * v, 3)})
                contrib.sort(key=lambda x: -abs(x["v"]))
                pg_ = None
                if r.blue in tm and r.red in tm:
                    d = team_logit(r.blue, tm, {}) - team_logit(r.red, tm, {})
                    wp0 = meta["blend"]["pre"]
                    pre = sig(wp0["w0"] + wp0["bt"] * (d + meta["blend"]["bt_side"]) + wp0["ts_logit"] * ts_logit(r.blue, r.red, tm))
                    post = sig(w_post["w0"] + w_post["bt"] * (d + meta["blend"]["bt_side"]) + w_post["ts_logit"] * ts_logit(r.blue, r.red, tm) + w_post["draft"] * edge)
                    pg_ = {"pre": round(pre, 4), "post": round(post, 4)}
                item["games"].append({"n": int(r.game_n), "blue": r.blue, "red": r.red, "blue_win": int(r.blue_win), "length": r.length,
                                      "blue_picks": r.blue_picks, "red_picks": r.red_picks, "blue_bans": r.blue_bans, "red_bans": r.red_bans,
                                      "blue_players": [re.sub(r"\s*\(.*?\)", "", x) for x in r.blue_players], "red_players": [re.sub(r"\s*\(.*?\)", "", x) for x in r.red_players],
                                      "kills": [int(r.blue_kills), int(r.red_kills)], "gold": [int(r.blue_gold), int(r.red_gold)],
                                      "towers": [int(r.blue_towers), int(r.red_towers)], "dragons": [int(r.blue_dragons), int(r.red_dragons)],
                                      "barons": [int(r.blue_barons), int(r.red_barons)], "vod": r.vod, "p": pg_,
                                      "draft_edge": round(edge, 3), "draft_top": contrib[:4]})
        out_matches.append(item)

    # enjeux conditionnels (victoire/defaite) pour les prochains matchs a deux equipes connues
    if sim:
        upcoming = [m for m in out_matches if m.get("stakes")][:10]
        for m in upcoming:
            a, b = m["t1"], m["t2"]
            res = {}
            for who, win in (("a_win", a), ("b_win", b)):
                s = simulate_swiss(swiss, None, rounds, pf, N_COND, force=(a, b, win))
                res[who] = {"a": round(s["qual"][a], 3), "b": round(s["qual"][b], 3)}
            m["stakes"]["if"] = res

    standings = []
    for t in swiss:
        standings.append({"team": t, "w": W[t], "l": Lc[t], "qual": round(sim["qual"][t], 3) if sim else None,
                          "title": round(title.get(t, 0), 3), "semis": round(semis.get(t, 0), 3),
                          "rec": {k: round(v, 3) for k, v in sorted(sim["rec"][t].items(), key=lambda kv: -kv[1])[:6]} if sim else {},
                          "power": tm[t]["power"] if t in tm else None, "region": tm[t]["region"] if t in tm else None})
    standings.sort(key=lambda s: (-(s["w"] - s["l"]), -s["w"], -(s["qual"] or 0)))

    # meta champions (Worlds + 75 derniers jours majeurs)
    def agg(df, since_label):
        picks, bans, wins = Counter(), Counter(), Counter(); role_c = defaultdict(Counter); n = len(df)
        for r in df.itertuples():
            for side in ("blue", "red"):
                won = (r.blue_win == 1) == (side == "blue")
                for i, c in enumerate(getattr(r, f"{side}_picks")):
                    picks[c] += 1; wins[c] += won; role_c[c][ROLES[i]] += 1
                for c in getattr(r, f"{side}_bans"):
                    bans[c] += 1
        rows = []
        for c in set(picks) | set(bans):
            pr = role_c[c].most_common(1)[0][0] if role_c[c] else None
            rows.append({"champion": c, "picks": picks[c], "bans": bans[c], "wins": wins[c], "role": pr,
                         "roles": {k: v for k, v in role_c[c].most_common(3)}, "pres": round((picks[c] + bans[c]) / max(1, n), 3),
                         "wr": round(wins[c] / picks[c], 3) if picks[c] else None})
        rows.sort(key=lambda r: -r["pres"])
        return {"n": n, "since": since_label, "rows": rows[:80]}
    meta_w = agg(g[(g.league == "WLDs") & (g.date >= "2026-10-01")], "Worlds 2026")
    major = g[g.league.isin(["LCK", "LPL", "LEC", "LCS", "LTA N", "LTA S", "LTA", "LCP", "CBLOL", "MSI", "EWC", "WLDs"]) & (g.date >= g.date.max() - pd.Timedelta(days=60))]
    meta_m = agg(major, "60 derniers jours (ligues majeures)")
    dvc = dv["c"]

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    worlds = {"generated": now, "data_through": meta["data_through"], "rating_adjusted_games": int(len(wg)),
              "swiss": swiss, "standings": standings, "matches": out_matches, "profiles": {t: p for t, p in profiles.items() if p},
              "meta_worlds": meta_w, "meta_recent": meta_m, "sim": {"n": N_SIM, "note": T("Règles officielles Riot : à partir du R2, adversaire tiré au hasard parmi les équipes du même bilan, sans revanche ; quarts : les 3-0 contre des 3-2 tirés au sort, les 4 autres au hasard ; demi-finales tirées au hasard (règle non publiée).", "Official Riot rules: from R2 on, opponent drawn at random among teams with the same record, no rematches; quarterfinals: the 3-0 teams vs 3-2 teams drawn at random, the other four at random; semifinal pairing random (rule not published).")},
              "ratings": {t: {"bt": tm[t]["bt"] + adj.get(t, 0.0), "ts_mu": tm[t].get("ts_mu"), "ts_sig": tm[t].get("ts_sig"), "power": tm[t]["power"]} for t in swiss if t in tm},
              "done_ids": [m["id"] for m in out_matches if m["status"] == "completed" and m["stage"] == "Main Event"],
              "model": {"metrics": meta.get("metrics"), "blend": meta["blend"], "data_through": meta["data_through"]}}
    json.dump(worlds, open(DATA / "worlds.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    live = {"generated": now, "version": now, "standings": [{k: s[k] for k in ("team", "w", "l", "qual", "title")} for s in standings],
            "matches": [{"id": m["id"], "slug": m["slug"], "t1": m["t1"], "t2": m["t2"], "s1": m["s1"], "s2": m["s2"], "status": m["status"], "date": m["date"], "bo": m["bo"],
                         "pred": (m.get("pred") or {}).get("series")} for m in out_matches]}
    json.dump(live, open(DATA / "live.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(f"worlds.json: {len(out_matches)} matchs | {len(profiles)} profils | swiss {len(swiss)} equipes | ajustements live {len(wg)} games")
    if sim:
        print("qualif :", sorted(((round(v, 2), k) for k, v in sim["qual"].items()), reverse=True)[:16])
        print("titre :", sorted(((round(v, 3), k) for k, v in title.items()), reverse=True)[:6])


if __name__ == "__main__":
    main()
