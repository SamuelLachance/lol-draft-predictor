"""Rafraichit le snapshot board_input.parquet de facon INCREMENTALE et SURE.

Pourquoi : les fichiers Google Drive d'Oracle's Elixir sont quota-bloques depuis
les IP datacenter (donc peu fiables en CI), mais fonctionnent depuis une IP
residentielle. Ce script telecharge SEULEMENT les annees recentes, chaque annee
isolee (un echec n'annule pas le reste), puis MERGE les parties fraiches dans le
snapshot existant par gameid — on ne perd jamais l'historique et on ajoute les
nouveaux matchs. C'est la brique fiable pour "donnees a jour chaque jour".

Usage :
    python ml/refresh_snapshot.py                 # merge les annees recentes dans board_input.parquet
    python ml/refresh_snapshot.py --years 2025 2026
    python ml/refresh_snapshot.py --upload        # + gh release upload board-data (necessite gh + token)

L'ID Drive de l'annee courante change chaque saison et la page OE bloque les bots.
Pour ajouter/mettre a jour un ID sans toucher au code, creez data/oe_file_ids.json :
    { "2026": "LE_FILE_ID_DRIVE_DE_2026" }
(recuperez-le sur oracleselixir.com/tools/downloads dans un vrai navigateur).
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SNAP = ROOT / "data" / "board_input.parquet"
IDS_OVERRIDE = ROOT / "data" / "oe_file_ids.json"
# colonnes du snapshot (doit matcher make_board_input.COLS)
COLS = ["gameid", "league", "year", "split", "playoffs", "date", "patch", "side", "position",
        "playername", "teamname", "champion", "result", "kills", "deaths", "assists",
        "golddiffat15", "damageshare"]


def file_ids() -> dict[str, str]:
    import fetch_data
    ids = dict(fetch_data.CSV_FILE_IDS)
    if IDS_OVERRIDE.exists():                       # l'utilisateur peut ajouter/ecraser un ID (ex: 2026)
        ids.update({str(k): str(v) for k, v in json.load(open(IDS_OVERRIDE)).items()})
    return ids


def download_year(year: str, fid: str) -> Path | None:
    import gdown
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"oe_{year}.csv"
    url = f"https://drive.google.com/uc?id={fid}"
    try:
        gdown.download(url, str(out), quiet=True)
    except Exception as e:                           # quota Drive, reseau, etc. -> on saute cette annee
        print(f"[{year}] echec telechargement: {e}")
        return None
    if not out.exists() or out.stat().st_size < 10_000:
        print(f"[{year}] fichier trop petit/absent (quota Drive probable)")
        return None
    return out


def slim(df: pd.DataFrame, year: str) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df.get("date"), errors="coerce").dt.year.fillna(int(year)).astype(int)
    keep = [c for c in COLS if c in df.columns]
    return df[keep].copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", default=None, help="annees a rafraichir (defaut: 2 plus recentes connues)")
    ap.add_argument("--upload", action="store_true", help="gh release upload board-data board_input.parquet --clobber")
    args = ap.parse_args()

    ids = file_ids()
    years = args.years or sorted(ids, reverse=True)[:2]   # par defaut, les 2 annees les plus recentes
    fresh_frames = []
    for y in years:
        y = str(y)
        if y not in ids:
            print(f"[{y}] pas d'ID Drive connu (ajoutez-le dans data/oe_file_ids.json) — saute")
            continue
        p = download_year(y, ids[y])
        if p is None:
            continue
        try:
            fresh_frames.append(slim(pd.read_csv(p, low_memory=False), y))
            print(f"[{y}] ok")
        except Exception as e:
            print(f"[{y}] lecture echouee: {e}")

    if not fresh_frames:
        print("Aucune donnee fraiche recuperee — le snapshot reste inchange (best-effort).")
        return 1

    fresh = pd.concat(fresh_frames, ignore_index=True)
    fresh = fresh.dropna(subset=["gameid"])
    if SNAP.exists():
        base = pd.read_parquet(SNAP)
        base = base[~base.gameid.isin(set(fresh.gameid))]     # la version fraiche fait autorite sur ces games
        combined = pd.concat([base, fresh], ignore_index=True, sort=False)
    else:
        combined = fresh
    # ordre de colonnes stable
    combined = combined[[c for c in COLS if c in combined.columns]]
    combined.to_parquet(SNAP, index=False, compression="zstd")
    try:
        dmax = pd.to_datetime(combined["date"], errors="coerce").max()
        dthru = dmax.date() if pd.notna(dmax) else "?"
    except Exception:
        dthru = "?"
    print(f"snapshot: {SNAP.stat().st_size/1e6:.1f} MB | {len(combined):,} rows | "
          f"{combined.gameid.nunique():,} games | data_through {dthru}")

    if args.upload:
        try:
            subprocess.run(["gh", "release", "upload", "board-data", str(SNAP), "--clobber"], check=True)
            print("Release board-data mise a jour.")
        except Exception as e:
            print(f"upload gh echoue: {e}")
            return 2
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
