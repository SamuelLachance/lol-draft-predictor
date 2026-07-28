"""Recupere les donnees Oracle's Elixir fraiches depuis un MIROIR HuggingFace.

Pourquoi : le Google Drive officiel d'OE est quota-bloque globalement depuis les
IP datacenter (donc inutilisable en CI). HuggingFace, lui, se telecharge de
maniere fiable depuis n'importe ou (pas de quota "too many users"). C'est LA
source qui permet une mise a jour quotidienne automatique.

Miroir par defaut : Ultradistinto/oracle-lol-matches (CSV OE bruts par annee,
rafraichi periodiquement). On telecharge les annees recentes, on garde les
colonnes utiles au board, et on MERGE par gameid dans board_input.parquet
(on ne perd jamais l'historique).

    python ml/fetch_hf.py                # merge annee courante + precedente
    python ml/fetch_hf.py --years 2026   # annee(s) precise(s)
    python ml/fetch_hf.py --upload       # + gh release upload board-data
    python ml/fetch_hf.py --repo <id>    # autre miroir HF
"""
from __future__ import annotations
import argparse, re, subprocess, zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "board_input.parquet"
DEFAULT_REPO = "Ultradistinto/oracle-lol-matches"
COLS = ["gameid", "league", "year", "split", "playoffs", "date", "patch", "side", "position",
        "playername", "teamname", "champion", "result", "kills", "deaths", "assists",
        "golddiffat15", "damageshare"]
KEEP_POS = {"top", "jng", "mid", "bot", "sup", "team"}
TIER1 = {"LCK", "LPL", "LEC", "LTA", "LTA N", "LTA S", "LCS", "NA LCS", "LCP", "LJL",
         "VCS", "PCS", "CBLOL", "TCL", "LLA"}
TEXTCOLS = ["gameid", "league", "split", "date", "patch", "side", "position", "playername", "teamname", "champion"]
NUMCOLS = ["year", "playoffs", "result", "kills", "deaths", "assists", "golddiffat15", "damageshare"]


def _to_str_obj(s: pd.Series) -> pd.Series:
    # object dtype str/None (PAS le dtype "string" nullable : son pd.NA casse les tests
    # de verite de build_board comme all(pp["bp"]) / if c). Missing -> None.
    return s.map(lambda x: None if pd.isna(x) else str(x)).astype(object)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """dtypes homogenes -> ecriture parquet fiable + compatible build_board (object str/None)."""
    for c in TEXTCOLS:
        if c in df.columns:
            df[c] = _to_str_obj(df[c])
    for c in NUMCOLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _slim(df: pd.DataFrame, years: set[int]) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df.get("date"), errors="coerce").dt.year
    df = df[pd.to_numeric(df["year"], errors="coerce").isin(years)]
    df = df[df["position"].astype(str).str.lower().isin(KEEP_POS)]
    if "league" in df.columns:
        df = df[df["league"].astype(str).isin(TIER1)]
    df = df.copy()
    df["position"] = df["position"].astype(str).str.lower()
    df["side"] = df["side"].astype(str).str.lower()
    keep = [c for c in COLS if c in df.columns]
    out = df[keep].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    return _normalize(out)


def _read_any(path: str, years: set[int]) -> list[pd.DataFrame]:
    frames = []
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.lower().endswith(".csv") and any(str(y) in name for y in years):
                    with z.open(name) as fh:
                        frames.append(_slim(pd.read_csv(fh, low_memory=False), years))
    elif path.lower().endswith(".csv"):
        frames.append(_slim(pd.read_csv(path, low_memory=False), years))
    elif path.lower().endswith((".parquet", ".pq")):
        frames.append(_slim(pd.read_parquet(path), years))
    return frames


def fetch(repo: str, years: set[int]) -> pd.DataFrame | None:
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(repo, repo_type="dataset")
    # 1) fichiers par annee (csv/parquet) mentionnant une annee voulue
    per_year = [f for f in files if f.lower().endswith((".csv", ".parquet", ".pq"))
                and any(str(y) in f for y in years)]
    # 2) sinon, une archive zip contenant les CSV par annee
    targets = per_year or [f for f in files if f.lower().endswith(".zip")]
    frames = []
    for f in targets:
        try:
            p = hf_hub_download(repo, f, repo_type="dataset")
            frames.extend(_read_any(p, years))
        except Exception as e:
            print(f"  [{f}] echec: {e}")
    frames = [x for x in frames if x is not None and len(x)]
    return pd.concat(frames, ignore_index=True) if frames else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    now_year = pd.Timestamp.now().year
    years = set(args.years) if args.years else {now_year, now_year - 1}
    print(f"HF mirror {args.repo} | annees {sorted(years)}")

    fresh = fetch(args.repo, years)
    if fresh is None or not len(fresh):
        print("Aucune donnee fraiche recuperee (miroir vide/indispo) — snapshot inchange.")
        return 1
    fresh = fresh.dropna(subset=["gameid"])

    if not SNAP.exists():
        try:
            subprocess.run(["gh", "release", "download", "board-data", "--pattern", "board_input.parquet",
                            "-D", str(SNAP.parent), "--clobber"], check=True)
        except Exception as e:
            print(f"(pas de snapshot local, download Release echoue: {e})")
    if SNAP.exists():
        base = pd.read_parquet(SNAP)
        base = base[~base.gameid.isin(set(fresh.gameid))]        # les games frais font autorite
        combined = pd.concat([base, fresh], ignore_index=True, sort=False)
    else:
        combined = fresh
    combined = _normalize(combined[[c for c in COLS if c in combined.columns]])
    combined.to_parquet(SNAP, index=False, compression="zstd")
    dmax = pd.to_datetime(combined["date"], errors="coerce").max()
    print(f"snapshot: {SNAP.stat().st_size/1e6:.1f} MB | {combined.gameid.nunique():,} games | "
          f"+{fresh.gameid.nunique():,} frais | data_through {dmax.date() if pd.notna(dmax) else '?'}")

    if args.upload:
        try:
            subprocess.run(["gh", "release", "upload", "board-data", str(SNAP), "--clobber"], check=True)
            print("Release board-data mise a jour.")
        except Exception as e:
            print(f"upload gh echoue: {e}")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
