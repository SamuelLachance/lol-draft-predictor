"""Moteur v2 du board : Team Elo + region-calibration + TrueSkill joueurs/champions
+ predicteur blende (Elo region-calibre + TrueSkill roster). Exporte web/public/data/*.json
"""
from __future__ import annotations
import json, math, numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
import trueskill as ts
ROOT=Path(__file__).resolve().parents[1]; PROC=ROOT/"data"/"processed"; OUT=ROOT/"web"/"public"/"data"; OUT.mkdir(parents=True,exist_ok=True)
ROLES=["top","jng","mid","bot","sup"]
ENV=ts.TrueSkill(draw_probability=0.0); BETA=ENV.beta
def wl(a,b):  # trueskill winprob (a beats b), a,b = list of Rating
    mu=sum(r.mu for r in a)-sum(r.mu for r in b); s2=sum(r.sigma**2 for r in a)+sum(r.sigma**2 for r in b)+(len(a)+len(b))*BETA**2
    return 0.5*(1+math.erf(mu/math.sqrt(2*s2)))
def L(p): p=min(max(p,1e-4),1-1e-4); return math.log(p/(1-p))

REGION={}
for lg in ["LCK","LCKC","KeSPA","LCK CL"]: REGION[lg]="KR"
for lg in ["LPL","LDL","LPLOL","LPL CL"]: REGION[lg]="CN"
for lg in ["LEC","EM","LFL","LFL2","PRM","PRMP","NLC","NLC Aurora Open","UL","LVP SL","TCL","ESLOL","EBL","GLL","HLL","HM","NEXO","UKLC","LIT","AL","PGN","PRMP"]: REGION[lg]="EMEA"
for lg in ["LTA","LTA N","LTA S","LCS","NA LCS","NACL","CBLOL","CBLOLA","LAS","LLA","LRN","LRS","CD"]: REGION[lg]="AMERICAS"
for lg in ["LCP","PCS","VCS","LJL","LJLA","PCL","LCO","DDH","USP","UPL","OPL","VL"]: REGION[lg]="APAC"
# ligues TIER-1 internationales (Worlds/MSI) — PAS les tier-2 domestiques (LFL, EM, LVP SL...)
TIER1={"LCK","LPL","LEC","LTA","LTA N","LTA S","LCS","NA LCS","LCP","LJL","VCS","PCS","CBLOL","TCL","LLA"}

def main():
    oe=pd.read_parquet(PROC/"oe_all.parquet",columns=["gameid","side","position","playername","teamname","champion","result","league","date","kills","deaths","assists","golddiffat15","damageshare"])
    oe["_date"]=pd.to_datetime(oe.date,errors="coerce"); oe["year"]=oe._date.dt.year
    pl=oe[oe.position.astype(str).str.lower().isin(ROLES)].copy(); pl["position"]=pl.position.str.lower(); pl["side"]=pl.side.astype(str).str.lower()
    # joueurs par game
    pg={}
    for gid,g in pl.groupby("gameid",sort=False):
        b=g[g.side=="blue"].set_index("position"); r=g[g.side=="red"].set_index("position")
        if len(b)>=5 and len(r)>=5:
            pg[gid]={"bp":[b.loc[ro,"playername"] if ro in b.index else None for ro in ROLES],
                     "rp":[r.loc[ro,"playername"] if ro in r.index else None for ro in ROLES]}
    dt=pd.read_parquet(PROC/"drafts_team.parquet").dropna(subset=["blue_team","red_team","blue_win"]).copy()
    dt["blue_win"]=dt.blue_win.astype(int); dt["_date"]=pd.to_datetime(dt.date,errors="coerce"); dt=dt[dt._date.notna()].sort_values("_date").reset_index(drop=True)
    y=dt.blue_win.to_numpy(); n=len(dt); gid=dt.gameid.to_numpy()
    # home league & region par equipe
    lg=pd.concat([dt[["blue_team","league"]].rename(columns={"blue_team":"t"}),dt[["red_team","league"]].rename(columns={"red_team":"t"})])
    home=lg.groupby("t").league.agg(lambda s:s.mode().iloc[0] if len(s.mode()) else s.iloc[0]).to_dict()
    reg=lambda t: REGION.get(home.get(t,""),"OTHER")
    # passes causales
    R={};K=30; RG={}; KR=24
    PR=defaultdict(lambda:ENV.create_rating()); CR=defaultdict(lambda:ENV.create_rating())
    elo_logit=np.zeros(n); ts_logit=np.zeros(n)
    for i,row in enumerate(dt.itertuples(index=False)):
        tb,tr=str(row.blue_team),str(row.red_team); rb=R.get(tb,1500);rr=R.get(tr,1500)
        pe=1/(1+10**((rr-rb)/400)); elo_logit[i]=L(pe); R[tb]=rb+K*(y[i]-pe);R[tr]=rr+K*((1-y[i])-(1-pe))
        # region elo (majors), sur games inter-region
        ga,gb=reg(tb),reg(tr)
        if ga!=gb and ga!="OTHER" and gb!="OTHER":
            xa,xb=RG.get(ga,1500),RG.get(gb,1500); ea=1/(1+10**((xb-xa)/400)); RG[ga]=xa+KR*(y[i]-ea);RG[gb]=xb+KR*((1-y[i])-(1-ea))
        pp=pg.get(gid[i])
        if pp and all(pp["bp"]) and all(pp["rp"]):
            pb=[PR[p] for p in pp["bp"]];pr=[PR[p] for p in pp["rp"]]; ts_logit[i]=L(wl(pb,pr))
            npb,npr=ENV.rate([pb,pr],ranks=[0,1] if y[i] else [1,0])
            for p,x in zip(pp["bp"],npb):PR[p]=x
            for p,x in zip(pp["rp"],npr):PR[p]=x
        cb=[CR[getattr(row,f"blue_{r}")] for r in ROLES];cc=[CR[getattr(row,f"red_{r}")] for r in ROLES]
        ncb,ncc=ENV.rate([cb,cc],ranks=[0,1] if y[i] else [1,0])
        for r,x in zip(ROLES,ncb):CR[getattr(row,f"blue_{r}")]=x
        for r,x in zip(ROLES,ncc):CR[getattr(row,f"red_{r}")]=x
    # region-calibration de l'Elo d'equipe : effective = region_elo + (team - moyenne region)
    team_reg=defaultdict(list)
    for t,e in R.items():
        if home.get(t,"") in TIER1: team_reg[reg(t)].append(e)  # baseline région = équipes tier-1
    reg_mean_elo={k:np.mean(v) for k,v in team_reg.items()}
    reg_elo={k:RG.get(k,1500) for k in reg_mean_elo}
    def eff(t): return reg_elo.get(reg(t),1500)+(R[t]-reg_mean_elo.get(reg(t),1500)) if t in R else 1500
    # roster TS actuel par equipe (5 derniers joueurs par role)
    recent=pl[pl.year>=2025]
    roster={}
    for (team),g in recent.groupby("teamname"):
        rr={}
        for ro in ROLES:
            sub=g[g.position==ro].sort_values("_date")
            if len(sub): rr[ro]=sub.playername.iloc[-1]
        if len(rr)==5: roster[team]=rr
    def team_ts(t):
        if t not in roster: return None
        rs=[PR[roster[t][ro]] for ro in ROLES]; return (sum(r.mu for r in rs),math.sqrt(sum(r.sigma**2 for r in rs)))
    # blend logistique causal : elo_logit + ts_logit -> outcome
    from sklearn.linear_model import LogisticRegression
    cut=int(n*0.85); X=np.c_[elo_logit,ts_logit]; blend=LogisticRegression(C=1e6).fit(X[:cut],y[:cut])
    w0=float(blend.intercept_[0]); we,wt=[float(x) for x in blend.coef_[0]]
    # accuracy recente du blend (causal)
    pt=blend.predict_proba(X[cut:])[:,1]; acc=float(((pt>=.5).astype(int)==y[cut:]).mean())

    # ---- teams.json ----
    tr_rows=oe[(oe.position.astype(str).str.lower()=="team")&(oe.year>=2025)][["teamname","result","league"]].dropna(subset=["teamname","result"])
    rec=tr_rows.groupby("teamname").agg(wins=("result","sum"),games=("result","count"),league=("league",lambda s:s.mode().iloc[0] if len(s.mode()) else "")).reset_index()
    rec=rec[(rec.games>=8)&(rec.league.isin(TIER1))]
    teams=[]
    for r in rec.itertuples():
        tts=team_ts(r.teamname)
        teams.append({"team":r.teamname,"league":r.league,"region":reg(r.teamname),"power":round(eff(r.teamname)),
            "elo":round(R.get(r.teamname,1500)),"ts_mu":round(tts[0],1) if tts else None,"ts_sig":round(tts[1],1) if tts else None,
            "wins":int(r.wins),"losses":int(r.games-r.wins),"games":int(r.games),"wr":round(r.wins/r.games,3)})
    teams.sort(key=lambda t:-t["power"])
    json.dump(teams,open(OUT/"teams.json","w",encoding="utf-8"),ensure_ascii=False)

    # ---- players.json (TrueSkill) ----
    recent["kda"]=((recent.kills+recent.assists)/recent.deaths.replace(0,1)).clip(upper=20)
    g=recent.groupby(["playername","position"]).agg(team=("teamname",lambda s:s.mode().iloc[0] if len(s.mode()) else ""),league=("league",lambda s:s.mode().iloc[0] if len(s.mode()) else ""),games=("result","count"),wr=("result","mean"),gd15=("golddiffat15","mean"),kda=("kda","mean")).reset_index()
    g=g[g.games>=10]
    players=[]
    for r in g.itertuples():
        pr=PR.get(r.playername)
        if pr is None: continue
        players.append({"player":r.playername,"team":r.team,"league":r.league,"role":r.position,
            "ts":round(pr.mu-3*pr.sigma,1),"mu":round(pr.mu,1),"games":int(r.games),"wr":round(float(r.wr),3),
            "gd15":int(r.gd15) if r.gd15==r.gd15 else 0,"kda":round(float(r.kda),2) if r.kda==r.kda else 0})
    players.sort(key=lambda p:-p["ts"])
    json.dump(players,open(OUT/"players.json","w",encoding="utf-8"),ensure_ascii=False)

    # ---- champions.json (champion TrueSkill + meta) ----
    cc2=recent.groupby(["champion","position"]).agg(games=("result","count"),wr=("result","mean"),gd15=("golddiffat15","mean")).reset_index()
    cc2=cc2[cc2.games>=15]
    champs=[]
    for r in cc2.itertuples():
        cr=CR.get(r.champion)
        champs.append({"champion":r.champion,"role":r.position,"ts":round(cr.mu-3*cr.sigma,1) if cr else 25.0,
            "games":int(r.games),"wr":round(float(r.wr),3),"gd15":int(r.gd15) if r.gd15==r.gd15 else 0})
    champs.sort(key=lambda c:-c["ts"])
    json.dump(champs,open(OUT/"champions.json","w",encoding="utf-8"),ensure_ascii=False)

    # ---- regions.json ----
    regions=[{"region":k,"rating":round(v),"mean_team_elo":round(reg_mean_elo.get(k,1500))} for k,v in sorted(reg_elo.items(),key=lambda t:-t[1]) if k!="OTHER"]
    json.dump(regions,open(OUT/"regions.json","w",encoding="utf-8"),ensure_ascii=False)

    # ---- matches.json (blend causal) ----
    matches=[]
    for i in range(max(0,n-300),n):
        row=dt.iloc[i]; p=1/(1+math.exp(-(w0+we*elo_logit[i]+wt*ts_logit[i])))
        matches.append({"date":str(pd.to_datetime(row._date).date()),"blue":str(row.blue_team),"red":str(row.red_team),"pred":round(float(p),3),"blue_win":int(y[i])})
    json.dump(matches[::-1],open(OUT/"matches.json","w",encoding="utf-8"),ensure_ascii=False)

    json.dump({"updated":str(pd.Timestamp.now().date()),"n_teams":len(teams),"n_players":len(players),"n_champions":len(champs),
        "model":"Team Elo (region-cal) + player TrueSkill + draft","recent_acc":round(acc,3),
        "blend":{"w0":w0,"elo":we,"ts":wt},"note":"Player TrueSkill + region-calibrated Elo. Draft edge from upset analysis."},
        open(OUT/"meta.json","w",encoding="utf-8"),ensure_ascii=False)
    print(f"teams {len(teams)} | players {len(players)} | champs {len(champs)} | regions {len(regions)} | blend recent_acc {acc:.3f}")
    print("regions:",[(r['region'],r['rating']) for r in regions])
    print("top teams:",[(t['team'],t['power'],t['region']) for t in teams[:6]])

if __name__=="__main__": main()
