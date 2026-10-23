"""Collecte les timelines de games (flux livestats lolesports, rejouable pour les games passes) -> data/raw/timelines/<gameId>.json

Un echantillon toutes les STEP secondes : etat des deux equipes (or, kills, tours, dragons, barons, inhibiteurs, niveaux, joueurs en vie, CS).
Reprenable : un fichier par game, les games deja collectes sont ignores.
Usage : python ml/fetch_timelines.py [--since 2025-01-01] [--leagues lck,lpl,...] [--max N] [--workers 8]
"""
from __future__ import annotations
import argparse, datetime as dt, json, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "timelines"; OUT.mkdir(parents=True, exist_ok=True)
KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
ES = "https://esports-api.lolesports.com/persisted/gw/"
FEED = "https://feed.lolesports.com/livestats/v1/window/"
LEAGUES = {"worlds": "98767975604431411", "msi": "98767991325878492", "first_stand": "113464388705111224", "lck": "98767991310872058",
           "lpl": "98767991314006698", "lec": "98767991302996019", "lcs": "98767991299243165", "lta_n": "113470291645289904", "lta_s": "113475181634818701",
           "lcp": "113476371197627891", "cblol": "98767991332355509", "ewc": "116838530616006090", "lta_cross": "113475149040947852"}
STEP = 60
_lock = threading.Lock(); _stats = {"req": 0, "err": 0}


def _get(url, headers=None, tries=4):
    for k in range(tries):
        try:
            with _lock: _stats["req"] += 1
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=30).read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 400, 204): return None
            if e.code == 429: time.sleep(2 + 3 * k); continue
            time.sleep(1 + k)
        except Exception:
            time.sleep(1 + k)
    with _lock: _stats["err"] += 1
    return None


def es(path):
    b = _get(ES + path, {"x-api-key": KEY})
    return json.loads(b) if b else None


def window(gid, ts: dt.datetime):
    ts = ts.replace(microsecond=0) - dt.timedelta(seconds=ts.second % 10)       # l'API exige un instant multiple de 10 s (sinon 400)
    b = _get(FEED + gid + "?startingTime=" + ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
    if not b: return None
    try: d = json.loads(b)
    except Exception: return None
    return d if d.get("frames") else None


def P(s): return dt.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")


def events(league_id, since):
    """tous les matchs termines d'une ligue depuis `since`"""
    out, tok = [], None
    while True:
        d = es(f"getSchedule?hl=en-US&leagueId={league_id}" + (f"&pageToken={tok}" if tok else ""))
        if not d: break
        s = d["data"]["schedule"]; ev = s["events"]
        out += [e for e in ev if e.get("match") and e["state"] == "completed" and e["startTime"][:10] >= since]
        tok = (s.get("pages") or {}).get("older")
        if not tok or (ev and ev[0]["startTime"][:10] < since): break
    return out


def frame_row(f):
    def team(t):
        ps = t["participants"]
        return [t["totalGold"], t["totalKills"], t["towers"], t["barons"], t["inhibitors"], t["dragons"],
                sum(p["level"] for p in ps), sum(1 for p in ps if p["currentHealth"] > 0), sum(p["creepScore"] for p in ps)]
    return {"ts": f["rfc460Timestamp"], "st": f["gameState"], "b": team(f["blueTeam"]), "r": team(f["redTeam"])}


def find_start(gid, lo: dt.datetime):
    """premier instant ou le flux repond (debut de game, +-10 s) en partant de `lo` ; None si rien dans les 3 h."""
    t = lo; hi = None
    for _ in range(90):
        if window(gid, t): hi = t; break
        t += dt.timedelta(seconds=120)
    if hi is None: return None
    a, b = hi - dt.timedelta(seconds=120), hi
    while (b - a).total_seconds() > 10:
        m = a + dt.timedelta(seconds=round((b - a).total_seconds() / 2 / 10) * 10)
        if m <= a or m >= b: break
        if window(gid, m): b = m
        else: a = m
    return b


def collect_game(gid, lo: dt.datetime):
    t0 = find_start(gid, lo)
    if t0 is None: return None
    rows, t, ids = [], t0, None
    for _ in range(int(3 * 3600 / STEP)):
        w = window(gid, t)
        if w is None: break
        if ids is None and w.get("gameMetadata"): ids = (w["gameMetadata"]["blueTeamMetadata"]["esportsTeamId"], w["gameMetadata"]["redTeamMetadata"]["esportsTeamId"])
        r = frame_row(w["frames"][-1]); rows.append(r)
        if r["st"] == "finished": break
        if len(rows) > 3 and rows[-1]["ts"] == rows[-2]["ts"]: break
        t += dt.timedelta(seconds=STEP)
    return {"t0": t0.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows, "ids": ids}


def do_match(e, league):
    mid = e["match"]["id"]; det = es("getEventDetails?hl=en-US&id=" + mid)
    if not det: return 0
    m = det["data"]["event"]["match"]; names = {t["id"]: t["name"] for t in m.get("teams", [])}
    n = 0; lo = P(e["startTime"]); res = []
    for g in sorted(m.get("games", []), key=lambda x: x["number"]):
        f = OUT / f"{g['id']}.json"
        if g.get("state") != "completed":
            continue
        if f.exists():
            try: lo = P(json.loads(f.read_text())["rows"][-1]["ts"]) + dt.timedelta(seconds=90)
            except Exception: pass
            continue
        tl = collect_game(g["id"], lo)
        if not tl or len(tl["rows"]) < 5: continue
        bid, rid = tl.pop("ids") or (None, None)
        rec = {"id": g["id"], "match": mid, "n": g["number"], "league": league, "start": e["startTime"], "bo": (m.get("strategy") or {}).get("count"),
               "blue": names.get(bid), "red": names.get(rid), **tl}
        if not rec["blue"] or not rec["red"]: continue
        f.write_text(json.dumps(rec, separators=(",", ":")), encoding="utf-8"); n += 1
        lo = P(tl["rows"][-1]["ts"]) + dt.timedelta(seconds=90)
    return n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--since", default="2025-01-01"); ap.add_argument("--leagues", default=",".join(LEAGUES))
    ap.add_argument("--max", type=int, default=10 ** 9); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    todo = []
    for lg in a.leagues.split(","):
        ev = events(LEAGUES[lg], a.since); print(lg, len(ev), "matchs", flush=True); todo += [(e, lg) for e in ev]
    todo = todo[:a.max]; done = 0; t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(do_match, e, lg) for e, lg in todo]
        for i, f in enumerate(as_completed(futs)):
            try: done += f.result()
            except Exception as err: print("ERR", err, file=sys.stderr)
            if i % 20 == 0: print(f"{i}/{len(todo)} matchs | {done} games | {_stats['req']} req ({_stats['req']/(time.time()-t0):.1f}/s) err {_stats['err']}", flush=True)
    print("fini", done, "games")


if __name__ == "__main__":
    main()
