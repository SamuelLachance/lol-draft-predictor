"""Turn Oracle's Elixir player rows into one draft sample per pro game."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ROLES = ["top", "jng", "mid", "bot", "sup"]


def _normalize_role(role: str) -> str | None:
    r = str(role).strip().lower()
    mapping = {
        "top": "top",
        "jng": "jng",
        "jungle": "jng",
        "jungler": "jng",
        "mid": "mid",
        "middle": "mid",
        "bot": "bot",
        "adc": "bot",
        "bottom": "bot",
        "sup": "sup",
        "support": "sup",
        "supp": "sup",
    }
    return mapping.get(r)


def build_draft_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    df = df.rename(columns=str.lower)

    required = ["gameid", "side", "position", "champion", "result"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Keep player rows only (drop team aggregate rows).
    players = df[df["position"].astype(str).str.lower().isin(
        ["top", "jng", "jungle", "jungler", "mid", "middle", "bot", "adc", "bottom", "sup", "support", "supp"]
    )].copy()

    players["role"] = players["position"].map(_normalize_role)
    players["side"] = players["side"].astype(str).str.lower().str.strip()
    players = players[players["side"].isin(["blue", "red"])]
    players = players.dropna(subset=["role", "champion", "gameid"])

    # One row per game/side/role — if duplicates, keep first.
    players = players.drop_duplicates(subset=["gameid", "side", "role"], keep="first")

    records: list[dict] = []
    for gameid, g in players.groupby("gameid", sort=False):
        if len(g) != 10:
            continue

        blue = g[g["side"] == "blue"]
        red = g[g["side"] == "red"]
        if len(blue) != 5 or len(red) != 5:
            continue

        blue_map = dict(zip(blue["role"], blue["champion"].astype(str)))
        red_map = dict(zip(red["role"], red["champion"].astype(str)))
        if any(r not in blue_map or r not in red_map for r in ROLES):
            continue

        blue_win = int(blue["result"].iloc[0])
        # OE uses 1/0 for win/loss on player rows.
        if blue_win not in (0, 1):
            continue

        meta = g.iloc[0]
        row = {
            "gameid": gameid,
            "blue_win": blue_win,
            "patch": meta.get("patch"),
            "league": meta.get("league"),
            "year": meta.get("year", meta.get("source_year")),
            "date": meta.get("date"),
            "playoffs": meta.get("playoffs"),
        }
        for role in ROLES:
            row[f"blue_{role}"] = blue_map[role]
            row[f"red_{role}"] = red_map[role]

        # Optional bans if present on player/team rows in source.
        for i in range(1, 6):
            bcol = f"ban{i}"
            if bcol in g.columns:
                # bans are usually on team rows; skip if not reliable
                pass

        records.append(row)

    drafts = pd.DataFrame.from_records(records)
    return drafts


def main() -> None:
    src = PROCESSED / "oe_all.parquet"
    if not src.exists():
        raise SystemExit("Run fetch_data.py first")

    df = pd.read_parquet(src)
    drafts = build_draft_table(df)
    out = PROCESSED / "drafts.parquet"
    drafts.to_parquet(out, index=False)

    print(f"Games with complete 5v5 drafts: {len(drafts):,}")
    print(f"Blue win rate: {drafts['blue_win'].mean():.3f}")
    if "league" in drafts.columns:
        print("Top leagues by games:")
        print(drafts["league"].value_counts().head(15).to_string())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
