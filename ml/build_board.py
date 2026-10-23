"""Moteur v3 du board : Bradley-Terry a decalage de ligue + TrueSkill joueurs/champions + draft v2.
Exporte web/public/data/*.json et web/public/model/draft_v2.json
"""
from __future__ import annotations
import json, math, numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
import trueskill as ts
import sys; sys.path.insert(0,str(Path(__file__).resolve().parent))
import strength as st, draftmodel as dm, model_v2 as mv, features as ft, rapm, jsonsafe as jsafe
ROOT=Path(__file__).resolve().parents[1]; PROC=ROOT/"data"/"processed"; OUT=ROOT/"web"/"public"/"data"; OUT.mkdir(parents=True,exist_ok=True)
MODEL_OUT=ROOT/"web"/"public"/"model"; MODEL_OUT.mkdir(parents=True,exist_ok=True)
ROLES=["top","jng","mid","bot","sup"]
PICK=[f"blue_{r}" for r in ROLES]+[f"red_{r}" for r in ROLES]
ENV=ts.TrueSkill(draw_probability=0.0); BETA=ENV.beta
def wl(a,b):  # trueskill winprob (a beats b), a,b = list of Rating
    mu=sum(r.mu for r in a)-sum(r.mu for r in b); s2=sum(r.sigma**2 for r in a)+sum(r.sigma**2 for r in b)+(len(a)+len(b))*BETA**2
    return 0.5*(1+math.erf(mu/math.sqrt(2*s2)))
def L(p): p=min(max(p,1e-4),1-1e-4); return math.log(p/(1-p))

# ligues MAJEURES uniquement (celles qui envoient des equipes a Worlds/MSI) + evenements internationaux.
# Les divisions inferieures (TCL, LFL, EM, LDL, academies...) ne sont ni classees ni comparees aux majeures.
MAJOR_LEAGUES={"LCK","LPL","LEC","LTA","LTA N","LTA S","LCS","NA LCS","LCP","CBLOL"}
INTL=st.INTL
TIER1=MAJOR_LEAGUES|INTL

def destring(df):
    # convertit les colonnes dtype "string" (pd.NA) en object/None : les tests de verite
    # (all(...), if c) plantent sur pd.NA ("boolean value of NA is ambiguous").
    for c in df.columns:
        if str(df[c].dtype)=="string": df[c]=df[c].astype(object).where(df[c].notna(),None)
    return df

canon_map=st.canon_map

def mode_or(s,default=""):
    m=s.mode(); return str(m.iloc[0]) if len(m) else default

def main():
    oe=destring(pd.read_parquet(PROC/"oe_all.parquet",columns=["gameid","side","position","playername","teamname","champion","result","league","date","kills","deaths","assists","golddiffat15","damageshare"]))
    oe["_date"]=pd.to_datetime(oe.date,errors="coerce"); oe["year"]=oe._date.dt.year
    cm=canon_map(oe); oe["teamname"]=oe.teamname.map(lambda x:cm.get(x,x) if x else x)
    pl=oe[oe.position.astype(str).str.lower().isin(ROLES)].copy(); pl["position"]=pl.position.str.lower(); pl["side"]=pl.side.astype(str).str.lower()
    # joueurs par game (pivot vectorise)
    wide=pl.drop_duplicates(["gameid","side","position"]).pivot(index="gameid",columns=["side","position"],values="playername")
    def cols(side): return [wide[(side,r)] if (side,r) in wide.columns else pd.Series(None,index=wide.index,dtype=object) for r in ROLES]
    bps=list(zip(*cols("blue"))); rps=list(zip(*cols("red")))
    pg={gid:{"bp":[x if isinstance(x,str) else None for x in b],"rp":[x if isinstance(x,str) else None for x in r]} for gid,b,r in zip(wide.index,bps,rps)}
    dt=destring(pd.read_parquet(PROC/"drafts_team.parquet")).dropna(subset=["blue_team","red_team","blue_win"]).copy()
    dt["blue_team"]=dt.blue_team.map(lambda x:cm.get(x,x)); dt["red_team"]=dt.red_team.map(lambda x:cm.get(x,x))
    dt["blue_win"]=dt.blue_win.astype(int); dt["_date"]=pd.to_datetime(dt.date,errors="coerce"); dt=dt[dt._date.notna()].sort_values("_date").reset_index(drop=True)
    y=dt.blue_win.to_numpy(); n=len(dt); gid=dt.gameid.to_numpy()
    # passes causales : Elo brut (affichage + feature) et TrueSkill joueurs/champions
    R={};K=30
    PR=defaultdict(lambda:ENV.create_rating()); CR=defaultdict(lambda:ENV.create_rating())
    elo_logit=np.zeros(n); ts_logit=np.zeros(n)
    for i,row in enumerate(dt.itertuples(index=False)):
        tb,tr=str(row.blue_team),str(row.red_team); rb=R.get(tb,1500);rr=R.get(tr,1500)
        pe=1/(1+10**((rr-rb)/400)); elo_logit[i]=L(pe); R[tb]=rb+K*(y[i]-pe);R[tr]=rr+K*((1-y[i])-(1-pe))
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

    # ---- modele v4 : variables causales (forme, h2h, maitrise...) + BT equipe/joueur + draft, stack walk-forward ----
    attrs = ft.load_attrs(ROOT / "ml" / "champion_attrs.json")
    dt["playoffs"] = dt.playoffs.fillna(0) if "playoffs" in dt.columns else 0
    F = ft.build(dt, pg, attrs, TIER1, INTL)
    G = pd.DataFrame({"gameid": gid, "date": dt._date.values, "league": dt.league.values, "blue": dt.blue_team.astype(str).values,
        "red": dt.red_team.astype(str).values, "y": y, "elo_logit": elo_logit, "ts_logit": ts_logit})
    for c in PICK: G[c] = dt[c].astype(str).values
    for c in ["elo", "ts", "d_fw", "h2h", "comfort", "comfort_min"]: G[c] = F[c].values
    G["bp"] = [pg.get(x, {}).get("bp") for x in gid]; G["rp"] = [pg.get(x, {}).get("rp") for x in gid]
    G["has_pl"] = [bool(a_) and all(a_) and bool(b_) and all(b_) for a_, b_ in zip(G.bp, G.rp)]
    home = st.home_groups(G)
    asof = G.date.max() + pd.Timedelta(days=1)
    WF = mv.walk_forward(G, home); W, metrics = mv.stack(WF, home)
    rating, bt_side, off = st.fit_bt(G, asof, home, **mv.BT)
    pr_, tdev_, poff_, pbt_side = st.fit_pbt(G, asof, home, **mv.PBT)
    dmod = mv.fit_draft(G, asof)
    # RAPM champion : effet net (controle equipe + 10 joueurs + maitrise) -> "valeur cachee"
    WFs = WF.sort_values("date").reset_index(drop=True)
    bcR = WFs[[f"blue_{r}" for r in ROLES]].astype(str).values; rcR = WFs[[f"red_{r}" for r in ROLES]].astype(str).values
    bpR, rpR = WFs.bp.to_numpy(), WFs.rp.to_numpy(); hasR = WFs.has_pl.to_numpy()
    mbR, mrR = rapm.mastery(bpR, rpR, bcR, rcR, hasR)
    fit_idx = np.where(hasR & WFs.bt.notna().to_numpy() & WFs.league.isin(set(st.GROUP) | st.INTL).to_numpy() & (WFs.date >= "2021-06-01").to_numpy())[0]
    rv, rco = rapm.fit(WFs.bt.to_numpy(), WFs.y.to_numpy(), WFs.date.to_numpy(), bpR, rpR, bcR, rcR, mbR, mrR, fit_idx)
    rtab = rapm.champion_table(rv, rco)
    # effet ajuste evalue a la maitrise TYPIQUE du champion (comparable au winrate brut, qui melange tous les niveaux de maitrise)
    msum, mcnt = defaultdict(float), defaultdict(int)
    for i_ in np.where(hasR & (WFs.date >= "2025-01-01").to_numpy() & WFs.league.isin(TIER1).to_numpy())[0]:
        for k_ in range(5):
            msum[bcR[i_, k_]] += mbR[i_, k_]; mcnt[bcR[i_, k_]] += 1; msum[rcR[i_, k_]] += mrR[i_, k_]; mcnt[rcR[i_, k_]] += 1
    mean_m = {c_: msum[c_] / mcnt[c_] for c_ in mcnt}
    radj = {(r_.role, r_.champion): (r_.adj + (r_.learn or 0.0) * mean_m.get(r_.champion, 0.0), r_.learn) for r_ in rtab.itertuples()}
    grp = lambda t: home.get(t, "?")
    reg = lambda t: grp(t) if grp(t) in st.MAJOR else "OTHER"
    pw = lambda t: st.to_elo(rating(t))
    WF["p_pre"] = mv.apply(W["pre"], WF); WF["p_post"] = mv.apply(W["post"], WF)
    p_by_gid = dict(zip(WF.gameid, WF.p_pre))
    wq = W["post"]

    # export draft v2 (valeur de draft champion x role + matchups/synergies significatifs)
    vocab=dmod["vocab"]; coef=dmod["coef"]; dv={"c":{},"m":{},"s":{}}
    for k,j in vocab.items():
        v=float(coef[j])*dm.SCALE[k[0]]
        if k[0]=="c": dv["c"][k[2:]]=round(v,3)
        elif abs(v)>=0.04: dv[k[0]][k[2:]]=round(v,3)
    jsafe.dump({"asof":str(asof.date()),"b0":dmod["b0"],"weight":wq["draft"],**dv},open(MODEL_OUT/"draft_v2.json","w",encoding="utf-8"),ensure_ascii=False,separators=(",",":"))

    # ---- roster actuel par equipe = titulaire MODAL des 20 derniers games tier-1/internationaux ----
    recent=pl[(pl.year>=2025)&pl.league.isin(TIER1)]
    rt1r=recent.sort_values("_date")
    roster={}
    for team,g in rt1r.groupby("teamname"):
        rr={}
        for ro in ROLES:
            sub=g[g.position==ro]
            if len(sub): rr[ro]=mode_or(sub.tail(20).playername,str(sub.playername.iloc[-1]))
        if len(rr)==5: roster[team]=rr
    def team_ts(t):
        if t not in roster: return None
        rs=[PR[roster[t][ro]] for ro in ROLES]; return (sum(r.mu for r in rs),math.sqrt(sum(r.sigma**2 for r in rs)))

    # ---- matchs recents par equipe (probabilite PRE-draft, hors-echantillon) ----
    dt["p"]=[p_by_gid.get(g_,float("nan")) for g_ in gid]
    darr=dt._date.dt.strftime("%Y-%m-%d").to_numpy(); bt_=dt.blue_team.astype(str).to_numpy(); rt_=dt.red_team.astype(str).to_numpy()
    gteams={gid[i]:(bt_[i],rt_[i]) for i in range(n)}
    team_matches=defaultdict(list)
    for i in range(max(0,n-6000),n):
        p=dt.p.iloc[i]
        if p!=p: continue
        w=int(y[i])
        team_matches[bt_[i]].append({"date":darr[i],"opp":str(rt_[i]),"side":"blue","pred":round(float(p),3),"win":w})
        team_matches[rt_[i]].append({"date":darr[i],"opp":str(bt_[i]),"side":"red","pred":round(float(1-p),3),"win":1-w})
    team_matches={t:v[-12:][::-1] for t,v in team_matches.items()}

    # ---- teams.json : uniquement les equipes des ligues MAJEURES, actives ----
    trw=oe[(oe.position.astype(str).str.lower()=="team")&(oe.year>=2025)&oe.league.isin(TIER1)][["teamname","result","league","_date"]].dropna(subset=["teamname","result"])
    dom_lg=trw[trw.league.isin(MAJOR_LEAGUES)].groupby("teamname").league.agg(mode_or).to_dict()
    rec=trw.groupby("teamname").agg(wins=("result","sum"),games=("result","count"),last=("_date","max")).reset_index()
    rec=rec[(rec.games>=8)&rec.teamname.map(lambda t:reg(t)!="OTHER")&(rec["last"]>=asof-pd.Timedelta(days=240))]
    lastd=trw.groupby("teamname")._date.max().to_dict()
    teams=[]
    for r in rec.itertuples():
        t=r.teamname; tts=team_ts(t); g_=reg(t)
        rost=[{"role":ro,"player":roster[t][ro]} for ro in ROLES] if t in roster else []
        mm=team_matches.get(t,[])[:10]; f10=sum(m["win"] for m in mm)
        rpl = [roster[t][ro] for ro in ROLES] if t in roster else []
        pbt_t = poff_.get(g_, 0.0) + tdev_.get(t, 0.0) + sum(pr_.get(p_, 0.0) for p_ in rpl)
        res_t = [int(w_) for w_ in (dt.blue_win.where(dt.blue_team == t, 1 - dt.blue_win)[(dt.blue_team == t) | (dt.red_team == t)].tail(10))]
        teams.append({"team":t,"league":dom_lg.get(t,""),"region":g_,"power":round(pw(t)),"bt":round(float(rating(t)),4),"pbt":round(float(pbt_t),4),"fw":st.form_weighted(res_t),
            "elo":round(R.get(t,1500)),"ts_mu":round(tts[0],1) if tts else None,"ts_sig":round(tts[1],1) if tts else None,
            "wins":int(r.wins),"losses":int(r.games-r.wins),"games":int(r.games),"wr":round(r.wins/r.games,3),
            "lg_off":round(st.to_elo(off.get(g_,0.0))-1500),"dev":round(pw(t)-st.to_elo(off.get(g_,0.0))),
            "form":f"{f10}-{len(mm)-f10}","last":str(lastd.get(t,pd.NaT))[:10],
            "roster":rost,"matches":team_matches.get(t,[])})
    teams.sort(key=lambda t:-t["power"])
    jsafe.dump(teams,open(OUT/"teams.json","w",encoding="utf-8"),ensure_ascii=False)
    tnames={t["team"] for t in teams}
    h2h={}
    rec2=dt[dt._date>=asof-pd.Timedelta(days=730)]
    for r_ in rec2.itertuples():
        if r_.blue_team in tnames and r_.red_team in tnames:
            k_="|".join(sorted((r_.blue_team,r_.red_team))); e_=h2h.setdefault(k_,{"n":0,"w":{}}); e_["n"]+=1
            wn=r_.blue_team if r_.blue_win else r_.red_team; e_["w"][wn]=e_["w"].get(wn,0)+1
    jsafe.dump(h2h,open(OUT/"h2h.json","w",encoding="utf-8"),ensure_ascii=False,separators=(",",":"))

    # ---- players.json : PERF INDIVIDUELLE (signal primaire) + contexte d'equipe calibre par ligue ----
    rt1=recent.copy().sort_values("_date")
    rt1["team_kills"]=rt1.groupby(["gameid","side"])["kills"].transform("sum").clip(lower=1)
    rt1["kp"]=(rt1.kills+rt1.assists)/rt1.team_kills
    rt1["kda"]=((rt1.kills+rt1.assists)/rt1.deaths.replace(0,1)).clip(upper=20)
    rt1["gdate"]=rt1._date.dt.strftime("%Y-%m-%d")
    cur_team={}; pdet={}
    for (pn,pos),s in rt1.groupby(["playername","position"],sort=False):
        s=s.sort_values("_date"); last=s.tail(25)
        cur_team[(pn,pos)]=mode_or(last.teamname,str(last.teamname.iloc[-1]))
        cg=s.groupby("champion").agg(gp=("result","count"),cwr=("result","mean"),ckda=("kda","mean")).reset_index().sort_values("gp",ascending=False)
        champs=[{"champion":str(c.champion),"games":int(c.gp),"wr":round(float(c.cwr),3),"kda":round(float(c.ckda),2)} for c in cg.head(8).itertuples()]
        recg=[]
        for gg in s.tail(8).iloc[::-1].itertuples(index=False):
            br=gteams.get(gg.gameid,("","")); opp=br[1] if str(gg.side)=="blue" else br[0]
            recg.append({"date":str(gg.gdate),"champion":str(gg.champion),"win":int(gg.result),
                "k":int(gg.kills),"d":int(gg.deaths),"a":int(gg.assists),
                "gd15":int(gg.golddiffat15) if gg.golddiffat15==gg.golddiffat15 else 0,"opp":str(opp)})
        wins=int(s.result.sum()); pdet[(pn,pos)]={"champs":champs,"recent":recg,"wins":wins,"losses":int(len(s)-wins),"last":str(s._date.max())[:10]}
    g=rt1.groupby(["playername","position"]).agg(
        games=("result","count"),wr=("result","mean"),gd15=("golddiffat15","mean"),
        kda=("kda","mean"),dmg=("damageshare","mean"),kp=("kp","mean")).reset_index()
    g["team"]=[cur_team.get((pn,po),"") for pn,po in zip(g.playername,g.position)]
    g=g[g.team.isin(tnames)&(g.games>=20)].copy()          # joueurs des equipes majeures actives, echantillon suffisant
    for col in ["gd15","kda","dmg","kp","wr"]:
        g[col+"_z"]=g.groupby("position")[col].transform(lambda s:(s-s.mean())/(s.std() or 1)).fillna(0)
    Wt={"top":{"gd15_z":.28,"dmg_z":.18,"kda_z":.14,"kp_z":.15,"wr_z":.25},
       "jng":{"gd15_z":.16,"dmg_z":.08,"kda_z":.16,"kp_z":.30,"wr_z":.30},
       "mid":{"gd15_z":.24,"dmg_z":.22,"kda_z":.12,"kp_z":.17,"wr_z":.25},
       "bot":{"gd15_z":.20,"dmg_z":.28,"kda_z":.15,"kp_z":.12,"wr_z":.25},
       "sup":{"gd15_z":.0,"dmg_z":.0,"kda_z":.28,"kp_z":.42,"wr_z":.30}}
    KSH=25
    players=[]
    for r in g.itertuples():
        w=Wt.get(r.position,{}); perf_raw=sum(getattr(r,k)*v for k,v in w.items())
        shrink=r.games/(r.games+KSH); perf=perf_raw*shrink
        teampw=pw(r.team); rbase=st.to_elo(off.get(reg(r.team),0.0))
        teampw_eff=rbase+(teampw-rbase)*shrink
        c_team=(teampw_eff-1500)/40.0; c_ind=7.0*perf
        d=pdet.get((r.playername,r.position),{})
        players.append({"player":r.playername,"team":r.team,"league":dom_lg.get(r.team,""),"region":reg(r.team),"role":r.position,
            "rating":round(c_team+c_ind+50,1),"team_pw":round(teampw),"perf":round(float(perf),2),
            "c_team":round(float(c_team),1),"c_ind":round(float(c_ind),1),
            "games":int(r.games),"wins":d.get("wins",0),"losses":d.get("losses",0),"wr":round(float(r.wr),3),
            "gd15":int(r.gd15) if r.gd15==r.gd15 else 0,"dmg":round(float(r.dmg),3) if r.dmg==r.dmg else 0,
            "kda":round(float(r.kda),2) if r.kda==r.kda else 0,"kp":round(float(r.kp),2) if r.kp==r.kp else 0,
            "last":d.get("last",""),"champs":d.get("champs",[]),"recent":d.get("recent",[])})
    players.sort(key=lambda p:-p["rating"])
    jsafe.dump(players,open(OUT/"players.json","w",encoding="utf-8"),ensure_ascii=False)

    # ---- champions.json (TrueSkill + valeur de draft ajustee, tier-1 recent) ----
    cc2=recent.groupby(["champion","position"]).agg(games=("result","count"),wr=("result","mean"),gd15=("golddiffat15","mean")).reset_index()
    cc2=cc2[cc2.games>=10]
    champs=[]
    _w=[(radj[(r_.position,r_.champion)][0],r_.games) for r_ in cc2.itertuples() if (r_.position,r_.champion) in radj]
    adj_center=sum(a*g_ for a,g_ in _w)/max(1,sum(g_ for _,g_ in _w))      # recentrage : effet moyen pondere = 0 (comme le winrate brut moyen)
    for r in cc2.itertuples():
        cr=CR.get(r.champion)
        a_, l_ = radj.get((r.position, r.champion), (None, None))
        if a_ is not None: a_ = a_ - adj_center
        sg_ = lambda z: (1 / (1 + math.exp(-z)) - .5) * 100
        raw_ = ((float(r.wr) * r.games + 10 * .5) / (r.games + 10) - .5) * 100       # winrate brut shrinke (ce que voit la communaute)
        champs.append({"champion":r.champion,"role":r.position,"ts":round(cr.mu-3*cr.sigma,1) if cr else 25.0,
            "dv":dv["c"].get(f"{r.position}|{r.champion}",0.0),
            "adj":round(a_,3) if a_ is not None else None,"adj_pts":round(sg_(a_),1) if a_ is not None else None,
            "learn15":round(sg_(l_*2.08),1) if l_ is not None else None,"raw_pts":round(raw_,1),
            "hv":round(sg_(a_)-raw_,1) if a_ is not None else None,
            "games":int(r.games),"wr":round(float(r.wr),3),"gd15":int(r.gd15) if r.gd15==r.gd15 else 0})
    champs.sort(key=lambda c:-(c["adj"] if c["adj"] is not None else c["dv"]))
    jsafe.dump(champs,open(OUT/"champions.json","w",encoding="utf-8"),ensure_ascii=False)
    seen=sorted(set(pd.unique(dt[PICK].astype(str).values.ravel())))
    pool=[{"champion":c,"ts":round(CR[c].mu-3*CR[c].sigma,1)} for c in seen if c and c!="nan"]
    jsafe.dump(pool,open(OUT/"champ_pool.json","w",encoding="utf-8"),ensure_ascii=False)

    # ---- regions.json : force de ligue (decalage inter-ligues) ----
    regions=[]
    for k in sorted(st.MAJOR,key=lambda k:-off.get(k,0)):
        ts_=[t["power"] for t in teams if t["region"]==k]
        regions.append({"region":k,"rating":round(st.to_elo(off.get(k,0.0))),"mean_team_elo":round(float(np.mean(ts_))) if ts_ else None,"n":len(ts_)})
    jsafe.dump(regions,open(OUT/"regions.json","w",encoding="utf-8"),ensure_ascii=False)

    # ---- matches.json : 300 derniers matchs majeurs/internationaux, prob pre-draft hors-echantillon ----
    ok=dt.league.isin(TIER1)&dt.p.notna()
    last=dt[ok].tail(300)
    matches=[{"date":str(pd.to_datetime(r.date).date()),"blue":str(r.blue_team),"red":str(r.red_team),"pred":round(float(r.p),3),"blue_win":int(r.blue_win)} for r in last.itertuples()]
    jsafe.dump(matches[::-1],open(OUT/"matches.json","w",encoding="utf-8"),ensure_ascii=False)
    acc300=float(((last.p>=.5).astype(int)==last.blue_win).mean())

    data_through=str(dt._date.max().date())
    jsafe.dump({"updated":str(pd.Timestamp.now().date()),"data_through":data_through,
        "n_teams":len(teams),"n_players":len(players),"n_champions":len(champs),"n_games":int(n),
        "model":"v4 stack: team + player Bradley-Terry, roster TrueSkill, form, head-to-head, draft, player-champion mastery","recent_acc":round(acc300,3),
        "blend":{"pre":W["pre"],"post":W["post"],"bt_side":bt_side,"pbt_side":pbt_side},"metrics":metrics,"bt_params":mv.BT,"pbt_params":mv.PBT,
        "note":"Walk-forward validated (monthly refits, no leakage). Pro games are noisy: ~65% is near the ceiling for pre-game information."},
        open(OUT/"meta.json","w",encoding="utf-8"),ensure_ascii=False)
    WF[["gameid","bt","draft","p_pre","p_post"]].to_parquet(PROC/"wf_preds.parquet",index=False)
    print(f"teams {len(teams)} | players {len(players)} | champs {len(champs)} | regions {len(regions)} | acc300 {acc300:.3f} | data_through {data_through}")
    print("walk-forward:",json.dumps(metrics))
    print("league offsets:",[(r['region'],r['rating']) for r in regions])
    print("top teams:",[(t['team'],t['power'],t['region']) for t in teams[:12]])

if __name__=="__main__": main()
