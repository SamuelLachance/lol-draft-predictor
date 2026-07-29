"""Recupere les games PRO RECENTES depuis Leaguepedia (API Cargo) et les merge
dans board_input.parquet — c'est la source qui permet d'etre a jour CHAQUE JOUR.

Pourquoi Leaguepedia : le Google Drive d'Oracle's Elixir est quota-bloque
globalement ("Too many users"), donc inutilisable en automatique. Leaguepedia
expose les scoreboards en API publique et contient les games du JOUR MEME.

Limites assumees : Leaguepedia ne fournit ni golddiffat15 ni damageshare
directement -> golddiffat15 reste NaN, damageshare est recalcule depuis
DamageToChampions (part des degats de l'equipe). Les ratings d'equipe (Elo)
et de joueurs (TrueSkill) n'en dependent pas.

    python ml/fetch_leaguepedia.py                 # depuis la derniere date du snapshot
    python ml/fetch_leaguepedia.py --since 2026-06-01
    python ml/fetch_leaguepedia.py --upload        # + gh release upload board-data
"""
from __future__ import annotations
import argparse, json, re, subprocess, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "board_input.parquet"
CACHE = ROOT / "data" / "raw" / "lp_cache"
# Special:CargoExport (export en masse) et NON api.php : l'API action=cargoquery
# rate-limit tres agressivement les requetes anonymes, CargoExport passe.
BASE = "https://lol.fandom.com/wiki/Special:CargoExport"
UA = "lolcoach-board/1.0 (esports analytics research; contact samuellachance5@gmail.com)"
COLS = ["gameid", "league", "year", "split", "playoffs", "date", "patch", "side", "position",
        "playername", "teamname", "champion", "result", "kills", "deaths", "assists",
        "golddiffat15", "damageshare"]
ROLE = {"top": "top", "jungle": "jng", "jng": "jng", "mid": "mid", "middle": "mid",
        "bot": "bot", "adc": "bot", "support": "sup", "sup": "sup"}
# prefixe de tournoi Leaguepedia -> code ligue OE
LEAGUES = {"LCK": "LCK", "LPL": "LPL", "LEC": "LEC", "LCS": "LCS", "LCP": "LCP",
           "LJL": "LJL", "VCS": "VCS", "PCS": "PCS", "CBLOL": "CBLOL", "TCL": "TCL",
           "LLA": "LLA", "LTA North": "LTA N", "LTA South": "LTA S", "LTA": "LTA",
           "MSI": "MSI", "Worlds": "WLDs", "World Championship": "WLDs",
           "First Stand": "FST", "EWC": "EWC", "Esports World Cup": "EWC"}


def export(params: dict, tries: int = 6) -> list[dict] | None:
    params = dict(params); params["format"] = "json"
    url = BASE + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:
            print(f"    {type(e).__name__} -> retry dans {8*(k+1)}s", flush=True)
            time.sleep(8 * (k + 1))
    return None


def paged(tables: str, fields: str, where: str, order: str, tag: str, cap: int = 30000) -> list[dict]:
    """Pagine un export Cargo, avec cache disque par page (reprise possible)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out, off, LIM = [], 0, 500
    while off < cap:
        cf = CACHE / f"{tag}_{off}.json"
        if cf.exists():
            rows = json.loads(cf.read_text(encoding="utf-8"))
        else:
            rows = export({"tables": tables, "fields": fields, "where": where,
                           "order_by": order, "limit": str(LIM), "offset": str(off)})
            if rows is None:
                print(f"  [{tag}] abandon a l'offset {off}", flush=True); break
            cf.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            time.sleep(1.5)                                 # politesse entre pages
        out.extend(rows)
        print(f"  [{tag}] offset {off}: +{len(rows)} (total {len(out)})", flush=True)
        if len(rows) < LIM:
            break
        off += LIM
    return out


def val(row: dict, *names):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None


def league_of(tournament: str) -> str | None:
    t = str(tournament or "")
    for pref in sorted(LEAGUES, key=len, reverse=True):
        if t.startswith(pref + " ") or t == pref:
            return LEAGUES[pref]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="AAAA-MM-JJ (defaut: derniere date du snapshot)")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore le cache disque")
    args = ap.parse_args()

    if args.fresh and CACHE.exists():
        for f in CACHE.glob("*.json"): f.unlink()

    since = args.since
    base = pd.read_parquet(SNAP) if SNAP.exists() else None
    if since is None:
        if base is None:
            raise SystemExit("Pas de snapshot : precise --since AAAA-MM-JJ")
        dmax = pd.to_datetime(base["date"], errors="coerce").max()
        # petite fenetre de recouvrement : suffit pour le quotidien, garde le job leger
        since = (dmax - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    print(f"Leaguepedia : recuperation des games depuis {since}")

    G = "ScoreboardGames"
    games = paged(G, ",".join(f"{G}.{f}" for f in
                  ["GameId", "Tournament", "DateTime_UTC", "Team1", "Team2", "WinTeam", "Patch"]),
                  f"{G}.DateTime_UTC >= '{since}'", f"{G}.DateTime_UTC", "sg")
    if not games:
        print("Aucune game recuperee."); return 1

    gmeta = {}
    for r in games:
        gid = val(r, "GameId"); lg = league_of(val(r, "Tournament"))
        if not gid or not lg:
            continue
        gmeta[gid] = {"league": lg, "date": val(r, "DateTime UTC", "DateTime_UTC"),
                      "patch": val(r, "Patch"), "t1": val(r, "Team1"), "t2": val(r, "Team2"),
                      "win": val(r, "WinTeam")}
    print(f"  games tier-1/international retenues : {len(gmeta)} / {len(games)}")
    if not gmeta:
        print("Aucune game tier-1 sur la periode."); return 1

    P = "ScoreboardPlayers"
    players = paged(P, ",".join(f"{P}.{f}" for f in
                    ["GameId", "Link", "Champion", "Team", "Side", "Role",
                     "Kills", "Deaths", "Assists", "DamageToChampions", "DateTime_UTC"]),
                    f"{P}.DateTime_UTC >= '{since}'", f"{P}.DateTime_UTC", "sp")
    if not players:
        print("Aucun scoreboard joueur."); return 1

    rows = []
    for r in players:
        gid = val(r, "GameId")
        g = gmeta.get(gid)
        if not g:
            continue
        role = ROLE.get(str(val(r, "Role") or "").strip().lower())
        if not role:
            continue
        side_raw = str(val(r, "Side") or "").strip().lower()
        side = "blue" if side_raw in ("1", "blue") else "red" if side_raw in ("2", "red") else None
        team = val(r, "Team")
        if side is None or not team:
            continue
        rows.append({"gameid": gid, "league": g["league"], "date": g["date"], "patch": g["patch"],
                     "side": side, "position": role,
                     "playername": val(r, "Link"), "teamname": team,
                     "champion": val(r, "Champion"),
                     "result": 1 if str(team) == str(g["win"]) else 0,
                     "kills": val(r, "Kills"), "deaths": val(r, "Deaths"), "assists": val(r, "Assists"),
                     "dmg": val(r, "DamageToChampions")})
    if not rows:
        print("Aucune ligne joueur exploitable."); return 1
    df = pd.DataFrame(rows)
    for c in ["kills", "deaths", "assists", "dmg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # damageshare = part des degats dans l'equipe
    tot = df.groupby(["gameid", "side"])["dmg"].transform("sum")
    df["damageshare"] = (df["dmg"] / tot.where(tot > 0)).astype(float)
    df["golddiffat15"] = pd.NA

    # lignes "team" (1 par side) — build_drafts_team et les records d'equipe en ont besoin
    tm = (df.groupby(["gameid", "side"])
            .agg(league=("league", "first"), date=("date", "first"), patch=("patch", "first"),
                 teamname=("teamname", "first"), result=("result", "first"),
                 kills=("kills", "sum"), deaths=("deaths", "sum"), assists=("assists", "sum"))
            .reset_index())
    tm["position"] = "team"; tm["playername"] = None; tm["champion"] = None
    tm["damageshare"] = pd.NA; tm["golddiffat15"] = pd.NA

    fresh = pd.concat([df.drop(columns=["dmg"]), tm], ignore_index=True, sort=False)
    fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    fresh["year"] = pd.to_datetime(fresh["date"], errors="coerce").dt.year
    fresh["split"] = None; fresh["playoffs"] = 0
    fresh = fresh[[c for c in COLS if c in fresh.columns]]
    # dtypes homogenes (object/None, jamais le dtype "string" nullable)
    for c in ["gameid", "league", "split", "date", "patch", "side", "position", "playername", "teamname", "champion"]:
        fresh[c] = fresh[c].map(lambda x: None if pd.isna(x) else str(x)).astype(object)
    for c in ["year", "playoffs", "result", "kills", "deaths", "assists", "golddiffat15", "damageshare"]:
        fresh[c] = pd.to_numeric(fresh[c], errors="coerce")

    ng = fresh.gameid.nunique()
    print(f"  -> {len(fresh)} lignes / {ng} games "
          f"({pd.to_datetime(fresh.date).min()} a {pd.to_datetime(fresh.date).max()})")

    if base is not None:
        keep = base[~base.gameid.isin(set(fresh.gameid))]
        combined = pd.concat([keep, fresh], ignore_index=True, sort=False)
    else:
        combined = fresh
    combined = combined[[c for c in COLS if c in combined.columns]]
    combined.to_parquet(SNAP, index=False, compression="zstd")
    dmax = pd.to_datetime(combined["date"], errors="coerce").max()
    print(f"snapshot: {SNAP.stat().st_size/1e6:.1f} MB | {combined.gameid.nunique():,} games | "
          f"+{ng} frais | data_through {dmax.date() if pd.notna(dmax) else '?'}")

    if args.upload:
        try:
            subprocess.run(["gh", "release", "upload", "board-data", str(SNAP), "--clobber"], check=True)
            print("Release board-data mise a jour.")
        except Exception as e:
            print(f"upload gh echoue: {e}"); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
