"""Games PRO enrichis (picks, bans, objectifs, duree, or) depuis Leaguepedia + calendrier Worlds.

Alimente les pages de match (draft, meta, identite de jeu). Source unique et fraiche :
Leaguepedia Special:CargoExport (contient les games du jour meme, pas de quota).

    python ml/fetch_rich.py                  # incremental (3 derniers jours) si data/games_rich.parquet existe
    python ml/fetch_rich.py --since 2025-01-01   # backfill complet
    python ml/fetch_rich.py --schedule-only      # calendrier/resultats Worlds seulement (job "live", leger)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_leaguepedia import export, league_of, val   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RICH = ROOT / "data" / "games_rich.parquet"
SCHED = ROOT / "data" / "worlds_schedule.json"
WORLDS_PAGE = "2026 Season World Championship"
G = "ScoreboardGames"
FIELDS = ["GameId", "Tournament", "DateTime_UTC", "Patch", "MatchId", "N_GameInMatch", "Team1", "Team2",
          "Team1Bans", "Team2Bans", "Team1Picks", "Team2Picks", "Team1Players", "Team2Players",
          "Gamelength", "Team1Dragons", "Team2Dragons", "Team1Barons", "Team2Barons",
          "Team1Towers", "Team2Towers", "Team1Gold", "Team2Gold", "Team1Kills", "Team2Kills",
          "Team1Inhibitors", "Team2Inhibitors", "Team1RiftHeralds", "Team2RiftHeralds",
          "WinTeam", "Winner", "VOD"]


def pages(tables, fields, where, order, cap=40000):
    out, off, LIM = [], 0, 500
    while off < cap:
        rows = export({"tables": tables, "fields": fields, "where": where,
                       "order_by": order, "limit": str(LIM), "offset": str(off)})
        if rows is None:
            raise SystemExit(f"Leaguepedia indisponible (offset {off}) — rien n'est ecrase.")
        out.extend(rows)
        print(f"  {tables} offset {off}: +{len(rows)} (total {len(out)})", flush=True)
        if len(rows) < LIM:
            break
        off += LIM
    return out


def _secs(s):
    try:
        m, sec = str(s).split(":"); return int(m) * 60 + int(sec)
    except Exception:
        return None


def fetch_games(since: str) -> pd.DataFrame:
    rows = pages(G, ",".join(f"{G}.{f}" for f in FIELDS), f"{G}.DateTime_UTC >= '{since}'", f"{G}.DateTime_UTC")
    recs = []
    for r in rows:
        lg = league_of(val(r, "Tournament"))
        if not lg:
            continue
        b1, b2 = r.get("Team1Picks") or [], r.get("Team2Picks") or []
        if len(b1) != 5 or len(b2) != 5:
            continue
        rec = {"gameid": r["GameId"], "league": lg, "tournament": val(r, "Tournament"),
               "date": val(r, "DateTime UTC", "DateTime_UTC"), "patch": str(val(r, "Patch") or ""),
               "matchid": val(r, "MatchId"), "game_n": val(r, "N GameInMatch", "N_GameInMatch"),
               "blue": r["Team1"], "red": r["Team2"], "blue_win": 1 if r.get("Winner") in (1, "1") else 0,
               "length": _secs(r.get("Gamelength")), "vod": val(r, "VOD")}
        for s, k in (("blue", "Team1"), ("red", "Team2")):
            # listes serialisees en JSON : pyarrow refuse les listes heterogenes (ex. noms numeriques)
            rec[f"{s}_picks"] = json.dumps([str(x) for x in (r.get(f"{k}Picks") or [])], ensure_ascii=False)
            rec[f"{s}_bans"] = json.dumps([str(x) for x in (r.get(f"{k}Bans") or [])], ensure_ascii=False)
            rec[f"{s}_players"] = json.dumps([str(x) for x in (r.get(f"{k}Players") or [])], ensure_ascii=False)
            for f in ("Dragons", "Barons", "Towers", "Gold", "Kills", "Inhibitors", "RiftHeralds"):
                rec[f"{s}_{f.lower()}"] = pd.to_numeric(r.get(f"{k}{f}"), errors="coerce")
        recs.append(rec)
    df = pd.DataFrame(recs)
    if len(df):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def fetch_schedule() -> list[dict]:
    M = "MatchSchedule"
    f = ["Team1", "Team2", "DateTime_UTC", "BestOf", "Winner", "OverviewPage", "Round", "Team1Score", "Team2Score",
         "MatchId", "Tab", "Team1Final", "Team2Final", "Stream"]
    rows = pages(M, ",".join(f), f"OverviewPage LIKE '{WORLDS_PAGE}%'", "DateTime_UTC")
    out = []
    for r in rows:
        stage = "Play-In" if "Play-In" in str(r.get("OverviewPage")) else "Main Event"
        tab = str(r.get("Tab") or "")
        out.append({"id": r.get("MatchId"), "stage": stage, "round": tab, "date": val(r, "DateTime UTC", "DateTime_UTC"),
                    "bo": int(r.get("BestOf") or 1), "t1": r.get("Team1"), "t2": r.get("Team2"),
                    "s1": r.get("Team1Score"), "s2": r.get("Team2Score"), "winner": r.get("Winner"),
                    "stream": r.get("Stream")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    ap.add_argument("--schedule-only", action="store_true")
    a = ap.parse_args()
    RICH.parent.mkdir(parents=True, exist_ok=True)
    sched = fetch_schedule()
    SCHED.write_text(json.dumps(sched, ensure_ascii=False), encoding="utf-8")
    print(f"calendrier Worlds : {len(sched)} matchs")
    if a.schedule_only:
        return 0
    since = a.since
    old = pd.read_parquet(RICH) if RICH.exists() else None
    if since is None:
        since = "2025-01-01" if old is None else (old.date.max() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    print(f"games enrichis depuis {since}")
    new = fetch_games(since)
    if not len(new):
        print("aucune game"); return 1
    out = new if old is None else pd.concat([old[~old.gameid.isin(set(new.gameid))], new], ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    out.to_parquet(RICH, index=False)
    print(f"{RICH.name}: {len(out)} games | {out.date.min()} -> {out.date.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
