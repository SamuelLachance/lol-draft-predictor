"""Download Oracle's Elixir pro match CSVs for every available year (2014+)."""

from __future__ import annotations

import argparse
from pathlib import Path

import gdown
import pandas as pd
from tqdm import tqdm

# Google Drive file IDs published with Oracle's Elixir yearly dumps
# (same IDs used by the community OE pipeline).
CSV_FILE_IDS: dict[str, str] = {
    "2025": "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
    "2024": "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN",
    "2023": "1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2",
    "2022": "1EHmptHyzY8owv0BAcNKtkQpMwfkURwRy",
    "2021": "1fzwTTz77hcnYjOnO9ONeoPrkWCoOSecA",
    "2020": "1dlSIczXShnv1vIfGNvBjgk-thMKA5j7d",
    "2019": "11eKtScnZcpfZcD3w3UrD7nnpfLHvj9_t",
    "2018": "1GsNetJQOMx0QJ6_FN8M1kwGvU_GPPcPZ",
    "2017": "11fx3nNjSYB0X8vKxLAbYOrS2Bu6avm9A",
    "2016": "1muyfpaIqk8_0BFkgLCWXDGNgWSXoPBwG",
    "2015": "1qyckLuw0-hJM8XqFhlV9l1xAbr3H78T_",
    "2014": "12syQsRH2QnKrQZTQQ6G5zyVeTG2pAYvu",
}

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def download_year(year: str, force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"oe_{year}.csv"
    if out.exists() and out.stat().st_size > 10_000 and not force:
        return out

    file_id = CSV_FILE_IDS[year]
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(out), quiet=False)
    if not out.exists() or out.stat().st_size < 1000:
        raise RuntimeError(f"Download failed for {year}: {out}")
    return out


def load_and_normalize(path: Path, year: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df["source_year"] = int(year)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch all OE pro match CSVs")
    parser.add_argument("--years", nargs="*", default=sorted(CSV_FILE_IDS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    for year in tqdm(args.years, desc="Downloading OE years"):
        if year not in CSV_FILE_IDS:
            raise SystemExit(f"Unknown year {year}. Known: {sorted(CSV_FILE_IDS)}")
        path = download_year(year, force=args.force)
        frames.append(load_and_normalize(path, year))

    merged = pd.concat(frames, ignore_index=True, sort=False)
    out = ROOT / "data" / "processed" / "oe_all.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(f"Wrote {len(merged):,} rows -> {out}")
    if "league" in merged.columns:
        print("Leagues:", sorted(merged["league"].dropna().astype(str).unique())[:40], "...")
    if "gameid" in merged.columns:
        print("Unique games:", merged["gameid"].nunique())


if __name__ == "__main__":
    main()
