"""Refit RAPIDE de la force d'equipe (Bradley-Terry a decalage de ligue) apres chaque game.

Meme modele et memes hyperparametres que build_board.py (ml/model_v2.py:BT), mais SANS les boucles
Elo/TrueSkill (3-4 min) : ~15 s. Le job "live" l'appelle apres avoir ajoute les games du jour au snapshot,
donc les ratings des equipes (bt, power, dev, decalages de ligue) integrent chaque game jouee.
Les poids du stack, le TrueSkill des rosters et le modele de draft restent ceux du dernier build complet.

    python ml/refresh_ratings.py            # lit data/board_input.parquet, met a jour web/public/data/{teams,regions,meta}.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as st, model_v2 as mv

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "board_input.parquet"
OUT = ROOT / "web" / "public" / "data"
ROLES = ["top", "jng", "mid", "bot", "sup"]


def main() -> int:
    oe = pd.read_parquet(SNAP, columns=["gameid", "side", "position", "teamname", "champion", "result", "league", "date"])
    oe["_date"] = pd.to_datetime(oe.date, errors="coerce")
    cm = st.canon_map(oe)
    oe["teamname"] = oe.teamname.map(lambda x: cm.get(x, x) if isinstance(x, str) else x)
    oe["position"] = oe.position.astype(str).str.lower(); oe["side"] = oe.side.astype(str).str.lower()
    # memes games que build_drafts_team : drafts 5v5 completes (10 lignes joueur avec champion)
    pl = oe[oe.position.isin(ROLES) & oe.champion.notna()]
    full = pl.groupby("gameid").size(); full = set(full[full == 10].index)
    tm = oe[(oe.position == "team") & oe.gameid.isin(full)]
    blue = tm[tm.side == "blue"].drop_duplicates("gameid").set_index("gameid")
    red = tm[tm.side == "red"].drop_duplicates("gameid").set_index("gameid")
    idx = blue.index.intersection(red.index)
    G = pd.DataFrame({"gameid": idx, "date": blue.loc[idx, "_date"].values, "league": blue.loc[idx, "league"].values,
                      "blue": blue.loc[idx, "teamname"].values, "red": red.loc[idx, "teamname"].values,
                      "y": blue.loc[idx, "result"].astype(float).values})
    G = G.dropna(subset=["date", "blue", "red", "y"]).sort_values("date").reset_index(drop=True)
    G["y"] = G.y.astype(int)
    home = st.home_groups(G)
    asof = G.date.max() + pd.Timedelta(days=1)
    rating, side, off = st.fit_bt(G, asof, home, **mv.BT)

    teams = json.load(open(OUT / "teams.json", encoding="utf-8"))
    for t in teams:
        r = rating(t["team"]); g_ = t["region"]
        t["bt"] = round(float(r), 4); t["power"] = round(st.to_elo(r))
        t["lg_off"] = round(st.to_elo(off.get(g_, 0.0)) - 1500); t["dev"] = round(st.to_elo(r) - st.to_elo(off.get(g_, 0.0)))
    teams.sort(key=lambda t: -t["power"])
    json.dump(teams, open(OUT / "teams.json", "w", encoding="utf-8"), ensure_ascii=False)
    regions = []
    for k in sorted(st.MAJOR, key=lambda k: -off.get(k, 0)):
        pw = [t["power"] for t in teams if t["region"] == k]
        regions.append({"region": k, "rating": round(st.to_elo(off.get(k, 0.0))), "mean_team_elo": round(float(np.mean(pw))) if pw else None, "n": len(pw)})
    json.dump(regions, open(OUT / "regions.json", "w", encoding="utf-8"), ensure_ascii=False)
    meta = json.load(open(OUT / "meta.json", encoding="utf-8"))
    meta["data_through"] = str(G.date.max().date()); meta["data_through_ts"] = str(G.date.max())
    meta["blend"]["bt_side"] = float(side); meta["refit_at"] = str(pd.Timestamp.utcnow())[:19]
    json.dump(meta, open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"refit BT: {len(G)} games jusqu'a {G.date.max()} | {len(teams)} equipes | top: {[(t['team'], t['power']) for t in teams[:5]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
