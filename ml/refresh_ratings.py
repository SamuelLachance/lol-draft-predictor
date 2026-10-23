"""Refit RAPIDE des signaux de force apres chaque game (job "live", ~30-40 s au lieu de ~25 min).

Refait EXACTEMENT comme build_board.py (memes fonctions, memes hyperparametres) :
  - bt  : Bradley-Terry d'equipe a decalage de ligue          -> power / bt / dev / decalages de ligue
  - pbt : Bradley-Terry au niveau joueur (somme du roster)   -> pbt de chaque equipe
  - fw  : forme ponderee (10 derniers resultats), h2h.json   -> confrontations directes sur 2 ans
Restent ceux du dernier build complet : poids du stack, TrueSkill des rosters, modele de draft, stats joueurs.

    python ml/refresh_ratings.py     # lit data/board_input.parquet, met a jour web/public/data/{teams,regions,meta,h2h}.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as st, model_v2 as mv, jsonsafe as jsafe

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "board_input.parquet"
OUT = ROOT / "web" / "public" / "data"
ROLES = ["top", "jng", "mid", "bot", "sup"]


def main() -> int:
    oe = pd.read_parquet(SNAP, columns=["gameid", "side", "position", "playername", "teamname", "champion", "result", "league", "date"])
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
    pgl = st.player_lists(oe[oe.gameid.isin(set(G.gameid))])
    G["bp"] = [pgl.get(g, {}).get("bp") for g in G.gameid]; G["rp"] = [pgl.get(g, {}).get("rp") for g in G.gameid]
    G["has_pl"] = [bool(a) and all(a) and bool(b) and all(b) for a, b in zip(G.bp, G.rp)]
    home = st.home_groups(G)
    asof = G.date.max() + pd.Timedelta(days=1)
    rating, side, off = st.fit_bt(G, asof, home, **mv.BT)
    pr, tdev, poff, pside = st.fit_pbt(G, asof, home, **mv.PBT)

    teams = json.load(open(OUT / "teams.json", encoding="utf-8"))
    names = {t["team"] for t in teams}
    for t in teams:
        r = rating(t["team"]); g_ = t["region"]
        t["bt"] = round(float(r), 4); t["power"] = round(st.to_elo(r))
        t["lg_off"] = round(st.to_elo(off.get(g_, 0.0)) - 1500); t["dev"] = round(st.to_elo(r) - st.to_elo(off.get(g_, 0.0)))
        rpl = [x["player"] for x in (t.get("roster") or [])]
        t["pbt"] = round(float(poff.get(g_, 0.0) + tdev.get(t["team"], 0.0) + sum(pr.get(p, 0.0) for p in rpl)), 4)
        mask = (G.blue == t["team"]) | (G.red == t["team"])
        res = [int(w) for w in G.y.where(G.blue == t["team"], 1 - G.y)[mask].tail(10)]
        t["fw"] = st.form_weighted(res)
        if res:
            t["form"] = f"{sum(res)}-{len(res) - sum(res)}"
    teams.sort(key=lambda t: -t["power"])
    jsafe.dump(teams, open(OUT / "teams.json", "w", encoding="utf-8"), ensure_ascii=False)
    regions = []
    for k in sorted(st.MAJOR, key=lambda k: -off.get(k, 0)):
        pw = [t["power"] for t in teams if t["region"] == k]
        regions.append({"region": k, "rating": round(st.to_elo(off.get(k, 0.0))), "mean_team_elo": round(float(np.mean(pw))) if pw else None, "n": len(pw)})
    jsafe.dump(regions, open(OUT / "regions.json", "w", encoding="utf-8"), ensure_ascii=False)
    # confrontations directes (2 ans)
    h2h = {}
    for r in G[G.date >= asof - pd.Timedelta(days=730)].itertuples():
        if r.blue in names and r.red in names:
            e = h2h.setdefault("|".join(sorted((r.blue, r.red))), {"n": 0, "w": {}}); e["n"] += 1
            w = r.blue if r.y else r.red; e["w"][w] = e["w"].get(w, 0) + 1
    jsafe.dump(h2h, open(OUT / "h2h.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    meta = json.load(open(OUT / "meta.json", encoding="utf-8"))
    meta["data_through"] = str(G.date.max().date()); meta["data_through_ts"] = str(G.date.max())
    meta["blend"]["bt_side"] = float(side); meta["blend"]["pbt_side"] = float(pside); meta["refit_at"] = str(pd.Timestamp.now(tz="UTC"))[:19]
    jsafe.dump(meta, open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"refit: {len(G)} games jusqu'a {G.date.max()} | {len(teams)} equipes | top: {[(t['team'], t['power']) for t in teams[:5]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
