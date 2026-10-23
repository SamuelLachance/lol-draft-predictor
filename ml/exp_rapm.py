"""Approches "valeur cachee" testees en walk-forward (refit trimestriel), puis integrees au stack pour mesurer le gain.

 RAPM   : ridge logistique  y ~ bt + niveau des 10 joueurs + champion x role + champion x maitrise  (effet net du champion
          en controlant le joueur, l'equipe et la courbe d'apprentissage) -> radj_champ (effet champion+maitrise), radj_player
 EMB    : embeddings de champions (torch) : biais champion x role + synergies alliees <e_i,e_j> + matchups antisymetriques
          -> emb   (equivalent leger d'un GNN sur le graphe de composition)
    python ml/exp_rapm.py            # calcule data/processed/v4_rapm.parquet puis evalue le stack
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as st, exp_v4 as e4
warnings.filterwarnings("ignore")
PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
ROLES = ["top", "jng", "mid", "bot", "sup"]
torch.set_num_threads(8)


def prep():
    dt, pg = e4.load()
    F = pd.read_parquet(PROC / "v4_feats.parquet"); F["date"] = pd.to_datetime(F.date)
    F = F.merge(pd.read_parquet(PROC / "v4_pbt.parquet"), on="gameid")
    F = F[F.league.isin(set(st.GROUP) | st.INTL)].sort_values("date").reset_index(drop=True)
    F["bp"] = F.gameid.map(lambda g: pg.get(g, {}).get("bp")); F["rp"] = F.gameid.map(lambda g: pg.get(g, {}).get("rp"))
    F["has_pl"] = F.bp.map(lambda l: bool(l) and all(l)) & F.rp.map(lambda l: bool(l) and all(l))
    bc = F[[f"blue_{r}" for r in ROLES]].astype(str).values; rc = F[[f"red_{r}" for r in ROLES]].astype(str).values
    # maitrise causale : nombre de games JOUEES par ce joueur sur ce champion avant la game
    cnt = {}; mb = np.zeros((len(F), 5)); mr = np.zeros((len(F), 5))
    for i, (bp, rp, has) in enumerate(zip(F.bp, F.rp, F.has_pl)):
        if not has: continue
        for k in range(5):
            mb[i, k] = cnt.get((bp[k], bc[i, k]), 0); mr[i, k] = cnt.get((rp[k], rc[i, k]), 0)
        for k in range(5):
            cnt[(bp[k], bc[i, k])] = cnt.get((bp[k], bc[i, k]), 0) + 1; cnt[(rp[k], rc[i, k])] = cnt.get((rp[k], rc[i, k]), 0) + 1
    return F, bc, rc, np.log1p(mb), np.log1p(mr)


# ------------------------------------------------------------------ RAPM
def rapm_matrix(F, idx, bc, rc, mb, mr, vocab, grow, S):
    """colonnes : bt | joueurs | champion x role | champion x maitrise."""
    r, c, v = [], [], []
    def col(key, scale):
        j = vocab.get(key)
        if j is None:
            if not grow: return None
            j = vocab[key] = len(vocab) + 1
        return j, scale
    bt = F.bt.to_numpy()
    for row, i in enumerate(idx):
        r.append(row); c.append(0); v.append(bt[i] * S["bt"])
        for k, role in enumerate(ROLES):
            for sgn, pl, ch, m in ((1.0, F.bp.iat[i][k], bc[i, k], mb[i, k]), (-1.0, F.rp.iat[i][k], rc[i, k], mr[i, k])):
                for key, scale, val in ((("p", pl), S["p"], sgn), (("c", role, ch), S["c"], sgn), (("m", ch), S["m"], sgn * m)):
                    jj = col(key, scale)
                    if jj is None: continue
                    r.append(row); c.append(jj[0]); v.append(val * jj[1])
    return r, c, v


def fit_rapm(F, tr, bc, rc, mb, mr, S, lam):
    vocab = {}
    r, c, v = rapm_matrix(F, tr, bc, rc, mb, mr, vocab, True, S)
    X = sp.csr_matrix((v, (r, c)), shape=(len(tr), len(vocab) + 1))
    age = (F.date.iat[tr[-1]] - F.date.iloc[tr]).dt.total_seconds().to_numpy() / 86400
    lr = LogisticRegression(C=1.0 / lam, max_iter=300).fit(X, F.y.iloc[tr].to_numpy(), sample_weight=0.5 ** (age / 365))
    return vocab, lr.coef_[0]


def rapm_parts(F, te, bc, rc, mb, mr, vocab, coef, S):
    """contribution CHAMPION (champion x role + maitrise) et JOUEUR pour chaque game de te."""
    ch = np.zeros(len(te)); pl = np.zeros(len(te))
    for row, i in enumerate(te):
        for k, role in enumerate(ROLES):
            for sgn, p_, c_, m in ((1.0, F.bp.iat[i][k], bc[i, k], mb[i, k]), (-1.0, F.rp.iat[i][k], rc[i, k], mr[i, k])):
                j = vocab.get(("p", p_)); pl[row] += sgn * coef[j] * S["p"] if j else 0
                j = vocab.get(("c", role, c_)); ch[row] += sgn * coef[j] * S["c"] if j else 0
                j = vocab.get(("m", c_)); ch[row] += sgn * m * coef[j] * S["m"] if j else 0
    return ch, pl


# ------------------------------------------------------------------ embeddings (graphe de composition)
class EmbNet(torch.nn.Module):
    def __init__(self, nch, d=6):
        super().__init__()
        self.bias = torch.nn.Embedding(nch * 5, 1); self.e = torch.nn.Embedding(nch, d); self.u = torch.nn.Embedding(nch, d); self.v = torch.nn.Embedding(nch, d)
        for m in (self.bias, self.e, self.u, self.v): torch.nn.init.normal_(m.weight, 0, 0.02)
        self.w_bt = torch.nn.Parameter(torch.tensor(1.0)); self.b0 = torch.nn.Parameter(torch.tensor(0.1)); self.nch = nch
        self.rol = torch.arange(5)

    def contrib(self, B, R):                      # B,R : (n,5) ids champion
        bb = self.bias(B * 5 + self.rol).squeeze(-1).sum(1) - self.bias(R * 5 + self.rol).squeeze(-1).sum(1)
        def syn(X):
            e = self.e(X); s = e.sum(1); return 0.5 * ((s * s).sum(-1) - (e * e).sum((1, 2)))
        u_b, v_b, u_r, v_r = self.u(B), self.v(B), self.u(R), self.v(R)
        # matchup antisymetrique : sum_ij (u_bi . v_rj - u_rj . v_bi)
        mu = (u_b.sum(1) * v_r.sum(1)).sum(-1) - (u_r.sum(1) * v_b.sum(1)).sum(-1)
        return bb + syn(B) - syn(R) + mu

    def forward(self, B, R, bt):
        return self.w_bt * bt + self.b0 + self.contrib(B, R)


def fit_emb(F, tr, bc, rc, cid, epochs=12, d=6, wd=3e-3, seed=0):
    torch.manual_seed(seed)
    net = EmbNet(len(cid) + 1, d); opt = torch.optim.Adam(net.parameters(), lr=0.02)
    B = torch.tensor([[cid.get(c, 0) for c in row] for row in bc[tr]]); R = torch.tensor([[cid.get(c, 0) for c in row] for row in rc[tr]])
    bt = torch.tensor(F.bt.iloc[tr].to_numpy(), dtype=torch.float32); y = torch.tensor(F.y.iloc[tr].to_numpy(), dtype=torch.float32)
    age = (F.date.iat[tr[-1]] - F.date.iloc[tr]).dt.total_seconds().to_numpy() / 86400; w = torch.tensor(0.5 ** (age / 365), dtype=torch.float32)
    n = len(tr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, 512):
            b = perm[s:s + 512]; opt.zero_grad()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(B[b], R[b], bt[b]), y[b], reduction="none") * w[b]).mean()
            reg = wd * (net.e.weight.pow(2).sum() + net.u.weight.pow(2).sum() + net.v.weight.pow(2).sum() + 3 * net.bias.weight.pow(2).sum()) / n * 512
            (loss + reg).backward(); opt.step()
    return net


def main():
    t0 = time.time()
    F, bc, rc, mb, mr = prep(); print("prep", len(F), round(time.time() - t0), "s", flush=True)
    champs = sorted(set(bc.ravel()) | set(rc.ravel())); cid = {c: i + 1 for i, c in enumerate(champs)}
    S = {"bt": 3.0, "p": 0.8, "c": 1.0, "m": 0.6}; LAM = 6.0
    tier = (F.home_b.isin(st.MAJOR) & F.home_r.isin(st.MAJOR) & F.has_pl & F.bt.notna()).to_numpy()
    n = len(F); radj_c = np.full(n, np.nan); radj_p = np.full(n, np.nan); emb = np.full(n, np.nan)
    cuts = [pd.Timestamp(f"{y}-{m:02d}-01") for y in range(2023, 2027) for m in (1, 4, 7, 10)][1:] + [pd.Timestamp("2027-01-01")]
    last = None
    for a, b in zip(cuts[:-1], cuts[1:]):
        tr = np.where((F.date < a) & (F.date >= "2021-06-01") & F.has_pl.to_numpy() & F.bt.notna().to_numpy())[0]
        te = np.where((F.date >= a) & (F.date < b) & tier)[0]
        if not len(te) or not len(tr): continue
        vocab, coef = fit_rapm(F, tr, bc, rc, mb, mr, S, LAM); ch, pl = rapm_parts(F, te, bc, rc, mb, mr, vocab, coef, S)
        radj_c[te] = ch; radj_p[te] = pl
        net = fit_emb(F, tr, bc, rc, cid)
        with torch.no_grad():
            Bt = torch.tensor([[cid.get(c, 0) for c in row] for row in bc[te]]); Rt = torch.tensor([[cid.get(c, 0) for c in row] for row in rc[te]])
            emb[te] = net.contrib(Bt, Rt).numpy()
        print(a.date(), len(tr), len(te), f"{time.time() - t0:.0f}s", flush=True)
        last = (vocab, coef)
    out = pd.DataFrame({"gameid": F.gameid, "radj_champ": radj_c, "radj_player": radj_p, "emb": emb}); out.to_parquet(PROC / "v4_rapm.parquet", index=False)
    # --- valeur cachee : effet champion ajuste (dernier refit) vs presence/winrate bruts
    vocab, coef = last
    rows = []
    for key, j in vocab.items():
        if key[0] == "c": rows.append({"role": key[1], "champion": key[2], "adj": coef[j] * S["c"]})
    mast = {k[1]: coef[j] * S["m"] for k, j in vocab.items() if k[0] == "m"}
    pd.DataFrame(rows).assign(mastery=lambda d: d.champion.map(mast)).to_parquet(PROC / "rapm_champs.parquet", index=False)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
