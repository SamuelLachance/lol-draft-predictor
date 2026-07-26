"""Idee utilisateur : miner les drafts des UPSETS.
Elo -> outsider. On compare la draft de l'outsider quand il GAGNE vs quand il PERD
(controle naturel : les deux sont l'equipe faible -> la difference = la draft).
On cherche des patterns recurrents : familles, attributs, archetypes, sidelanes, kit.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from collections import defaultdict
import train_v4 as v4
import interactions_kit as IK

PROC=v4.PROC if hasattr(v4,'PROC') else v4.PROCESSED
ROLES=["top","jng","mid","bot","sup"]; PICK=[f"blue_{r}" for r in ROLES]+[f"red_{r}" for r in ROLES]
POKE={"poke","disengage","zone","siege","waveclear","anti_dive"}
ALLIN={"engage","catch","dive","wombo","full_early","early","snowball","lane_bully","speed_user","cc_lock"}
HYPER={"hypercarry","enchanter","scaling"}
NUM=["cc","mobility","waveclear","poke","scaling","tankiness","sustain","engage","disengage","burst","global","anti_dive","pick"]

def main():
    db=v4.load_db()
    tags={c.lower():set(str(a.get('family','')).split())|set() for c,a in {}.items()}  # placeholder
    # tags depuis champion_families.csv
    import csv as _csv
    fam={}
    p=v4.SHANEI_CSV if hasattr(v4,'SHANEI_CSV') else (v4.ROOT.parent/"model"/"champion_families.csv")
    for r in _csv.DictReader(open(p,encoding='utf-8')):
        fam[str(r['champion']).lower()]=set(str(r['tags']).split('|'))
    d=pd.read_parquet(PROC/"drafts_team.parquet").dropna(subset=PICK+["blue_win","blue_team","red_team"]).copy()
    d["blue_win"]=d.blue_win.astype(int)
    for c in PICK: d[c]=d[c].astype(str)
    d["_date"]=pd.to_datetime(d.date,errors="coerce"); d=d[d._date.notna()].sort_values(["_date","gameid"]).reset_index(drop=True)
    y=d.blue_win.to_numpy(); n=len(d)
    # Elo causal + strength causale des champions (pour sidelane edge)
    R={};K=30;eb=np.zeros(n);er=np.zeros(n)
    cs=defaultdict(lambda:[0.,0.])  # champ causal winrate (role-agnostic)
    strengths=np.zeros((n,10))
    for i,row in enumerate(d.itertuples(index=False)):
        tb,tr=str(row.blue_team),str(row.red_team);rb=R.get(tb,1500);rr=R.get(tr,1500)
        eb[i]=rb;er[i]=rr;exp=1/(1+10**((rr-rb)/400))
        R[tb]=rb+K*(y[i]-exp);R[tr]=rr+K*((1-y[i])-(1-exp))
        for j,c in enumerate([getattr(row,f"blue_{r}") for r in ROLES]+[getattr(row,f"red_{r}") for r in ROLES]):
            strengths[i,j]=cs[c][0]/cs[c][1] if cs[c][1]>=5 else 0.5
        for c in [getattr(row,f"blue_{r}") for r in ROLES]: cs[c][0]+=y[i];cs[c][1]+=1
        for c in [getattr(row,f"red_{r}") for r in ROLES]: cs[c][0]+=1-y[i];cs[c][1]+=1

    fav_blue=eb>er; gap=np.abs(eb-er)
    ud_won=np.where(fav_blue,1-y,y)                    # 1 si l'outsider gagne
    # champions/roles de l'outsider et du favori
    syn,cnt,idx,_=IK.matrices(sorted(pd.unique(d[PICK].values.ravel())))
    def side_champs(i,blue): return [d.iloc[i][f"{'blue' if blue else 'red'}_{r}"] for r in ROLES]

    def feats(champs):
        ts=[fam.get(c.lower(),set()) for c in champs]; ds=[db.get(c.lower(),{}) for c in champs]
        poke=sum(len(t&POKE) for t in ts);allin=sum(len(t&ALLIN) for t in ts);hyp=sum(len(t&HYPER) for t in ts);tot=poke+allin+hyp or 1
        f={"fam_poke":poke/tot,"fam_allin":allin/tot,"fam_hyper":hyp/tot,"purity":max(poke,allin,hyp)/tot}
        for a in NUM: f["at_"+a]=float(np.mean([x.get(a,0) for x in ds])) if ds else 0
        f["n_tank"]=sum(1 for x in ds if x.get("tankiness",0)>=2)
        f["n_ench"]=sum(1 for t in ts if "enchanter" in t)
        f["n_ad"]=sum(1 for x in ds if x.get("damage")=="ad");f["n_ap"]=sum(1 for x in ds if x.get("damage")=="ap")
        f["dmg_imb"]=abs(f["n_ad"]-f["n_ap"]);f["n_ranged"]=sum(1 for x in ds if x.get("range_type")=="ranged")
        return f

    rows=[]
    real=(gap>=75)
    for i in np.where(real)[0]:
        ud=side_champs(i,not fav_blue[i]); fv=side_champs(i,fav_blue[i])
        fu=feats(ud); ff=feats(fv)
        rec={"won":int(ud_won[i]),"gap":gap[i]}
        for k,v in fu.items(): rec["ud_"+k]=v
        for k in fu: rec["diff_"+k]=fu[k]-ff[k]           # outsider - favori
        # sidelane edge (top+mid) via strength causale
        ud_idx=[0,1,2,3,4] if not fav_blue[i] else [5,6,7,8,9]
        fv_idx=[5,6,7,8,9] if not fav_blue[i] else [0,1,2,3,4]
        s=strengths[i]
        rec["side_edge"]=(s[ud_idx[0]]+s[ud_idx[2]])-(s[fv_idx[0]]+s[fv_idx[2]])   # top+mid
        rec["str_edge"]=np.mean([s[k] for k in ud_idx])-np.mean([s[k] for k in fv_idx])
        # kit synergy intra + counter net (outsider vs favori)
        bi=[idx[c] for c in ud];ri=[idx[c] for c in fv]
        rec["syn_ud"]=sum(syn[bi[a],bi[b]] for a in range(5) for b in range(a+1,5))
        rec["syn_diff"]=rec["syn_ud"]-sum(syn[ri[a],ri[b]] for a in range(5) for b in range(a+1,5))
        rec["counter_net"]=sum(cnt[bi[a],ri[b]] for a in range(5) for b in range(5))-sum(cnt[ri[a],bi[b]] for a in range(5) for b in range(5))
        rows.append(rec)
    df=pd.DataFrame(rows)
    W=df[df.won==1];L=df[df.won==0]
    base=df.won.mean()
    print(f"UPSETS analysables (ecart Elo>=75) : {len(df)}  | winrate outsider = {base:.3f}")
    print(f"  (outsider gagne n={len(W)}, perd n={len(L)})\n")

    # comparaison winners vs losers par feature (effet)
    cols=[c for c in df.columns if c not in ("won","gap")]
    res=[]
    for c in cols:
        a,b=W[c],L[c]; sd=df[c].std() or 1
        d_eff=(a.mean()-b.mean())/sd
        res.append((c,a.mean(),b.mean(),d_eff))
    res.sort(key=lambda t:-abs(t[3]))
    print("Feature de draft la plus liee a l'UPSET (effet = (win-loss)/sd), top 18 :")
    print(f"  {'feature':22s} {'win':>8s} {'loss':>8s} {'effet':>7s}")
    for c,aw,al,e in res[:18]:
        flag="  <<" if abs(e)>0.08 else ""
        print(f"  {c:22s} {aw:8.3f} {al:8.3f} {e:+7.3f}{flag}")

    # modele predictif : peut-on predire l'upset depuis la draft de l'outsider ?
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    X=StandardScaler().fit_transform(df[cols].fillna(0).to_numpy()); yy=df.won.to_numpy()
    p=cross_val_predict(LogisticRegression(max_iter=2000,C=0.5),X,yy,cv=5,method="predict_proba")[:,1]
    print(f"\nModele draft-outsider -> upset : ROC-AUC (CV) = {roc_auc_score(yy,p):.3f}  (0.5 = aucun pattern)")
    # quantile du meilleur predicteur composite
    q=pd.qcut(p,5,labels=False,duplicates="drop")
    print("winrate outsider par quintile de 'draft d'upset' predite :")
    for k in range(int(q.max())+1):
        sel=q==k; print(f"  Q{k+1}: {yy[sel].mean():.3f}  (n={sel.sum()})")

if __name__=="__main__": main()
