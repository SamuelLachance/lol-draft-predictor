"""Etape 1 du modele v4 : variables causales (features.py) + BT/draft walk-forward -> data/processed/v4_feats.parquet

    python ml/exp_v4.py            # ~8-10 min
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as st, model_v2 as mv, draftmodel as dm, features as ft

ROOT = Path(__file__).resolve().parents[1]; PROC = ROOT / "data" / "processed"
ROLES = ft.ROLES
MAJOR_LEAGUES = {"LCK", "LPL", "LEC", "LTA", "LTA N", "LTA S", "LCS", "NA LCS", "LCP", "CBLOL"}
TIER1 = MAJOR_LEAGUES | st.INTL


def load():
    oe = pd.read_parquet(PROC / "oe_all.parquet", columns=["gameid", "side", "position", "playername", "teamname", "date"])
    oe["_date"] = pd.to_datetime(oe.date, errors="coerce")
    cm = st.canon_map(oe)
    pl = oe[oe.position.astype(str).str.lower().isin(ROLES)].copy()
    pl["position"] = pl.position.str.lower(); pl["side"] = pl.side.astype(str).str.lower()
    wide = pl.drop_duplicates(["gameid", "side", "position"]).pivot(index="gameid", columns=["side", "position"], values="playername")
    def cols(side): return [wide[(side, r)] if (side, r) in wide.columns else pd.Series(None, index=wide.index, dtype=object) for r in ROLES]
    bps, rps = list(zip(*cols("blue"))), list(zip(*cols("red")))
    pg = {g: {"bp": [x if isinstance(x, str) else None for x in b], "rp": [x if isinstance(x, str) else None for x in r]} for g, b, r in zip(wide.index, bps, rps)}
    dt = pd.read_parquet(PROC / "drafts_team.parquet").dropna(subset=["blue_team", "red_team", "blue_win"]).copy()
    dt["blue_team"] = dt.blue_team.map(lambda x: cm.get(x, x)); dt["red_team"] = dt.red_team.map(lambda x: cm.get(x, x))
    dt["blue_win"] = dt.blue_win.astype(int); dt["_date"] = pd.to_datetime(dt.date, errors="coerce")
    dt = dt[dt._date.notna()].sort_values("_date").reset_index(drop=True)
    return dt, pg


def main():
    dt, pg = load()
    attrs = ft.load_attrs(ROOT / "ml" / "champion_attrs.json")
    print("games", len(dt), flush=True)
    F = ft.build(dt, pg, attrs, TIER1, st.INTL)
    F = F.merge(dt[["gameid"] + [f"{s}_{r}" for s in ("blue", "red") for r in ROLES]], on="gameid")
    F.to_parquet(PROC / "v4_base.parquet", index=False); print("features", F.shape, flush=True)
    # BT + draft v2 en walk-forward mensuel depuis 2021
    G = F.rename(columns={"blue": "blue", "red": "red"})
    G["elo_logit"] = G["elo"]; G["ts_logit"] = G["ts"]
    home = st.home_groups(G)
    bt = np.full(len(G), np.nan); dl = np.full(len(G), np.nan)
    for m in pd.period_range("2021-01", G.date.max(), freq="M"):
        a, b = m.start_time, (m + 1).start_time
        te = np.where((G.date >= a) & (G.date < b))[0]
        if not len(te): continue
        rating, side, _ = st.fit_bt(G, a, home, **mv.BT)
        sub = G.iloc[te]
        bt[te] = np.array([rating(x) - rating(z) for x, z in zip(sub.blue, sub.red)]) + side
        dl[te] = dm.logit(mv.fit_draft(G, a), *dm.to_lists(sub))
        print(m, len(te), flush=True)
    F["bt"] = bt; F["draft"] = dl
    F["home_b"] = F.blue.map(lambda t: home.get(t, "?")); F["home_r"] = F.red.map(lambda t: home.get(t, "?"))
    F.to_parquet(PROC / "v4_feats.parquet", index=False); print("saved", F.shape)


if __name__ == "__main__":
    main()
