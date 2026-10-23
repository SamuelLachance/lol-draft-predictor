"""Historique complet (2014 -> aujourd'hui) des games pro : compositions + DUREE + resultat, pour les etudes de meta.

    python ml/fetch_history.py          # ~10-15 min, reprise possible (cache par page) ; ecrit data/games_hist.parquet
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_leaguepedia import export, is_minor   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "games_hist.parquet"
CACHE = ROOT / "data" / "raw" / "hist_cache"
G = "ScoreboardGames"
FIELDS = ["GameId", "Tournament", "DateTime_UTC", "Patch", "Team1", "Team2", "Team1Picks", "Team2Picks", "Gamelength", "Winner"]
# tournois d'interet : internationaux + ligues domestiques majeures/regionales (Leaguepedia nomme "LCK 2019 Spring", "2016 Season ... LPL", etc.)
LIKES = ["World Championship", "Worlds", "Mid-Season", "MSI", "Rift Rivals", "LCK", "LPL", "LEC", "EU LCS", "NA LCS", "LCS", "LTA", "LMS", "LCP", "PCS", "VCS", "LJL",
         "CBLOL", "LLA", "TCL", "LCL", "OPL", "KeSPA", "Champions Korea", "Demacia", "Tencent"]


def secs(s):
    try:
        m, x = str(s).split(":"); return int(m) * 60 + int(x)
    except Exception:
        return None


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    like = " OR ".join(f"{G}.Tournament LIKE '%{x}%'" for x in LIKES)
    rows = []
    for year in range(2014, 2027):
        for half in ((f"{year}-01-01", f"{year}-07-01"), (f"{year}-07-01", f"{year + 1}-01-01")):
            off, tag = 0, f"{half[0]}"
            while True:
                cf = CACHE / f"{tag}_{off}.json"
                if cf.exists():
                    page = json.loads(cf.read_text(encoding="utf-8"))
                else:
                    page = export({"tables": G, "fields": ",".join(f"{G}.{f}" for f in FIELDS),
                                   "where": f"({like}) AND {G}.DateTime_UTC >= '{half[0]}' AND {G}.DateTime_UTC < '{half[1]}'",
                                   "order_by": f"{G}.DateTime_UTC", "limit": "500", "offset": str(off)})
                    if page is None:
                        print("abandon", tag, off, flush=True); break
                    cf.write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")
                rows.extend(page)
                print(tag, off, len(page), len(rows), flush=True)
                if len(page) < 500:
                    break
                off += 500
    recs = []
    for r in rows:
        p1, p2 = r.get("Team1Picks") or [], r.get("Team2Picks") or []
        if len(p1) != 5 or len(p2) != 5 or is_minor(r.get("Tournament")):
            continue
        recs.append({"gameid": r["GameId"], "tournament": r["Tournament"], "date": r.get("DateTime UTC") or r.get("DateTime_UTC"),
                     "patch": str(r.get("Patch") or ""), "blue": r["Team1"], "red": r["Team2"], "blue_win": 1 if str(r.get("Winner")) == "1" else 0,
                     "length": secs(r.get("Gamelength")), "blue_picks": json.dumps([str(x) for x in p1], ensure_ascii=False),
                     "red_picks": json.dumps([str(x) for x in p2], ensure_ascii=False)})
    df = pd.DataFrame(recs).drop_duplicates("gameid")
    df["date"] = pd.to_datetime(df.date, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"{OUT.name}: {len(df)} games | {df.date.min()} -> {df.date.max()} | avec duree: {df.length.notna().mean():.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
