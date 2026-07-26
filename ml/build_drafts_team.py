"""Construit drafts_team.parquet = drafts.parquet + noms d'equipes (pour l'Elo).
Etape du pipeline quotidien : fetch_data -> build_drafts -> build_drafts_team -> build_board_data.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"

def main():
    oe = pd.read_parquet(PROC / "oe_all.parquet", columns=["gameid", "side", "position", "teamname"])
    tm = oe[oe.position.astype(str).str.lower() == "team"][["gameid", "side", "teamname"]].copy()
    tm["side"] = tm.side.astype(str).str.lower()
    blue = tm[tm.side == "blue"][["gameid", "teamname"]].rename(columns={"teamname": "blue_team"})
    red = tm[tm.side == "red"][["gameid", "teamname"]].rename(columns={"teamname": "red_team"})
    teams = blue.merge(red, on="gameid").drop_duplicates("gameid")
    d = pd.read_parquet(PROC / "drafts.parquet").merge(teams, on="gameid", how="left")
    d.to_parquet(PROC / "drafts_team.parquet", index=False)
    print(f"drafts_team: {len(d)} games | with team names: {d.blue_team.notna().mean():.3f}")

if __name__ == "__main__":
    main()
