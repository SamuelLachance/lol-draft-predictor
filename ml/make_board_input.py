"""Extrait un snapshot slim (colonnes utiles au board) depuis oe_all.parquet.
Sert de source de donnees fiable pour le workflow (hebergee en Release GitHub,
car les fichiers Google Drive d'Oracle's Elixir sont quota-bloques en CI)."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT = Path(__file__).resolve().parents[1] / "data" / "board_input.parquet"
COLS = ["gameid","league","year","split","playoffs","date","patch","side","position",
        "playername","teamname","champion","result","kills","deaths","assists","golddiffat15","damageshare"]

def main():
    oe = pd.read_parquet(PROC / "oe_all.parquet")
    cols = [c for c in COLS if c in oe.columns]
    oe[cols].to_parquet(OUT, index=False, compression="zstd")
    print(f"board_input.parquet: {OUT.stat().st_size/1e6:.1f} MB, {len(oe)} rows, {len(cols)} cols")

if __name__ == "__main__":
    main()
