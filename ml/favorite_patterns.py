"""Miroir de upset_patterns : qu'est-ce qui fait PERDRE le FAVORI (draft) ?
Sur les vrais matchs a favori (ecart Elo>=75), on compare la draft du FAVORI
quand il perd (upset) vs quand il tient. La difference = les erreurs/vulnerabilites
de draft du favori (a force d'equipe egale = c'est la meilleure equipe).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from collections import defaultdict
import csv as _csv
import train_v4 as v4
import interactions_kit as IK

PROC=v4.PROCESSED; ROLES=["top","jng","mid","bot","sup"]; PICK=[f"blue_{r}" for r in ROLES]+[f"red_{r}" for r in ROLES]
POKE={"poke","disengage","zone","siege","waveclear","anti_dive"}
ALLIN={"engage","catch","dive","wombo","full_early","early","snowball","lane_bully","speed_user","cc_lock"}
HYPER={"hypercarry","enchanter","scaling"}
NUM=["cc","mobility","waveclear","poke","scaling","tankiness","sustain","engage","disengage","burst","global","anti_dive","pick"]

def main():
    db=v4.load_db(); fam={}
    p=v4.SHANEI_CSV if hasattr(v4,'SHANEI_CSV') else (v4.ROOT.parent/"model"/"champion_families.csv")
    for r in _csv.DictReader(open(p,encoding='utf-8')): fam[str(r['champion']).lower()]=set(str(r['tags']).split('|'))
    d=pd.read_parquet(PROC/"drafts_team.parquet").dropna(subset=PICK+["blue_win","blue_team","red_team"]).copy()
    d["blue_win"]=d.blue_win.astype(int)
    for c in PICK: d[c]=d[c].astype(str)
    d["_date"]=pd.to_datetime(d.date,errors="coerce"); d=d[d._date.notna()].sort_values(["_date","gameid"]).reset_index(drop=True)
    y=d.blue_win.to_numpy(); n=len(d)
    R={};K=30;eb=np.zeros(n);er=np.zeros(n); cs=defaultdict(lambda:[0.,0.]); strg=np.zeros((n,10))
    for i,row in enumerate(d.itertuples(index=False)):
        tb,tr=str(row.blue_team),str(row.red_team);rb=R.get(tb,1500);rr=R.get(tr,1500)
        eb[i]=rb;er[i]=rr;exp=1/(1+10**((rr-rb)/400));R[tb]=rb+K*(y[i]-exp);R[tr]=rr+K*((1-y[i])-(1-exp))
        for j,c in enumerate([getattr(row,f"blue_{r}") for r in ROLES]+[getattr(row,f"red_{r}") for r in ROLES]):
            strg[i,j]=cs[c][0]/cs[c][1] if cs[c][1]>=5 else 0.5
        for c in [getattr(row,f"blue_{r}") for r in ROLES]: cs[c][0]+=y[i];cs[c][1]+=1
        for c in [getattr(row,f"red_{r}") for r in ROLES]: cs[c][0]+=1-y[i];cs[c][1]+=1
    fav_blue=eb>er; gap=np.abs(eb-er); fav_lost=np.where(fav_blue,1-y,y)  # 1 si le favori perd (upset)
    syn,cnt,idx,_=IK.matrices(sorted(pd.unique(d[PICK].values.ravel())))
    def side_champs(i,blue): return [d.iloc[i][f"{'blue' if blue else 'red'}_{r}"] for r in ROLES]
    def feats(champs):
        ts=[fam.get(c.lower(),set()) for c in champs]; ds=[db.get(c.lower(),{}) for c in champs]
        poke=sum(len(t&POKE) for t in ts);allin=sum(len(t&ALLIN) for t in ts);hyp=sum(len(t&HYPER) for t in ts);tot=poke+allin+hyp or 1
        f={"fam_poke":poke/tot,"fam_allin":allin/tot,"fam_hyper":hyp/tot,"purity":max(poke,allin,hyp)/tot}
        for a in NUM: f["at_"+a]=float(np.mean([x.get(a,0) for x in ds])) if ds else 0
        f["n_tank"]=sum(1 for x in ds if x.get("tankiness",0)>=2); f["n_ench"]=sum(1 for t in ts if "enchanter" in t)
        f["n_ad"]=sum(1 for x in ds if x.get("damage")=="ad");f["n_ap"]=sum(1 for x in ds if x.get("damage")=="ap"); f["dmg_imb"]=abs(f["n_ad"]-f["n_ap"])
        return f
    rows=[]
    for i in np.where(gap>=75)[0]:
        fv=side_champs(i,fav_blue[i]); ud=side_champs(i,not fav_blue[i])
        ff=feats(fv); fu=feats(ud); rec={"lost":int(fav_lost[i])}
        for k,v in ff.items(): rec["fav_"+k]=v
        for k in ff: rec["diff_"+k]=ff[k]-fu[k]      # favori - outsider
        fi=[0,1,2,3,4] if fav_blue[i] else [5,6,7,8,9]; ui=[5,6,7,8,9] if fav_blue[i] else [0,1,2,3,4]; s=strg[i]
        rec["fav_side_edge"]=(s[fi[0]]+s[fi[2]])-(s[ui[0]]+s[ui[2]]); rec["fav_str_edge"]=np.mean([s[k] for k in fi])-np.mean([s[k] for k in ui])
        bi=[idx[c] for c in fv];ri=[idx[c] for c in ud]
        rec["fav_syn"]=sum(syn[bi[a],bi[b]] for a in range(5) for b in range(a+1,5))
        rec["fav_syn_diff"]=rec["fav_syn"]-sum(syn[ri[a],ri[b]] for a in range(5) for b in range(a+1,5))
        rows.append(rec)
    df=pd.DataFrame(rows); Lz=df[df.lost==1];Wz=df[df.lost==0]
    print(f"Matchs a favori (ecart Elo>=75) : {len(df)}  | le favori PERD {df.lost.mean():.3f} du temps")
    cols=[c for c in df.columns if c!="lost"]; res=[]
    for c in cols:
        sd=df[c].std() or 1; res.append((c,Lz[c].mean(),Wz[c].mean(),(Lz[c].mean()-Wz[c].mean())/sd))
    res.sort(key=lambda t:-abs(t[3]))
    print("\nCe qui distingue la DRAFT du FAVORI quand il PERD (effet=(perd-tient)/sd), top 18 :")
    print(f"  {'feature':22s} {'perd':>8s} {'tient':>8s} {'effet':>7s}")
    for c,al,aw,e in res[:18]:
        print(f"  {c:22s} {al:8.3f} {aw:8.3f} {e:+7.3f}{'  <<' if abs(e)>0.08 else ''}")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    X=StandardScaler().fit_transform(df[cols].fillna(0).to_numpy()); yy=df.lost.to_numpy()
    pr=cross_val_predict(LogisticRegression(max_iter=2000,C=0.5),X,yy,cv=5,method="predict_proba")[:,1]
    print(f"\nModele draft-favori -> defaite : ROC-AUC (CV) = {roc_auc_score(yy,pr):.3f}")
    q=pd.qcut(pr,5,labels=False,duplicates="drop")
    print("taux de defaite du favori par quintile de 'draft a risque' :")
    for k in range(int(q.max())+1):
        sel=q==k; print(f"  Q{k+1}: {yy[sel].mean():.3f}  (n={sel.sum()})")

if __name__=="__main__": main()
