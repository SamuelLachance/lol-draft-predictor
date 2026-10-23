/* Simulateur Swiss + quarts (regles OFFICIELLES Riot) execute dans le navigateur.
   Meme logique que ml/build_worlds.py : sert a recalculer IMMEDIATEMENT (sans attendre le job serveur) des qu'un resultat
   officiel tombe dans l'API lolesports : bilans, chances de qualification / titre, enjeux victoire-defaite.
   Regles : R1 tirage par tiers (connu) ; des R2 adversaire tire au hasard parmi le MEME bilan, jamais de revanche ;
   3 victoires = qualifie, 3 defaites = elimine ; Bo3 si une equipe a 2 victoires ou 2 defaites, sinon Bo1 ;
   quarts : les deux 3-0 contre deux des 3-2 (tirage), les 4 autres equipes tirees au hasard ; demies : tirage au hasard. */
(function (root) {
  const SIM = {};
  const K_LIVE = 0.10;
  const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]] } return a };
  const sig = z => 1 / (1 + Math.exp(-z));
  const erf = x => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * Math.exp(-x * x); return x >= 0 ? y : -y };
  const key = (a, b) => a < b ? a + '|' + b : b + '|' + a;

  SIM.seriesP = (p, bo) => { if (bo === 1) return p; const need = (bo >> 1) + 1, q = 1 - p; let s = 0; for (let k = 0; k < need; k++) { let c = 1; for (let i = 0; i < k; i++) c = c * (need - 1 + k - i) / (i + 1); s += c * Math.pow(p, need) * Math.pow(q, k) } return s };

  // --- modele v4 (identique a ml/build_worlds.py) : stack logistique ; R[team] = {bt, pbt, fw, ts_mu, ts_sig} ; H = h2h {"A|B":{n,w:{team:n}}}
  const h2hx = (a, b, H) => { const e = H && H[a < b ? a + '|' + b : b + '|' + a]; if (!e) return 0; return ((e.w[a] || 0) + 1) / (e.n + 2) - 0.5 };
  const tsLogit = (A, B) => {
    if (A.ts_mu == null || B.ts_mu == null) return 0;
    const BETA = 25 / 6, mu = A.ts_mu - B.ts_mu, s2 = A.ts_sig * A.ts_sig + B.ts_sig * B.ts_sig + 10 * BETA * BETA;
    let tp = 0.5 * (1 + erf(mu / Math.sqrt(2 * s2))); tp = Math.min(.999, Math.max(.001, tp)); return Math.log(tp / (1 - tp));
  };
  SIM.feats = (a, b, R, blend, H) => { const A = R[a], B = R[b];
    return { bt: A.bt - B.bt + blend.bt_side, pbt: (A.pbt || 0) - (B.pbt || 0) + (blend.pbt_side || 0), ts: tsLogit(A, B), h2h: h2hx(a, b, H),
      d_fw: (A.fw != null && B.fw != null) ? A.fw - B.fw : null } };
  SIM.zscore = (w, f) => { let z = w.w0; for (const c in w.fill) { const v = f[c]; z += w[c] * ((v == null || Number.isNaN(v)) ? w.fill[c] : v) } return z };
  // P(a bat b) en un game, symetrise sur le side (a bleu / a rouge)
  SIM.gameP = (a, b, R, blend, H, which, extra) => {
    const w = blend[which || 'pre'], fa = SIM.feats(a, b, R, blend, H), fb = SIM.feats(b, a, R, blend, H);
    if (extra) { Object.assign(fa, extra[0]); Object.assign(fb, extra[1]) }
    return 0.5 * (sig(SIM.zscore(w, fa)) + 1 - sig(SIM.zscore(w, fb)));
  };

  const boFor = (w1, l1, w2, l2) => (Math.max(w1, w2) === 2 || Math.max(l1, l2) === 2) ? 3 : 1;

  function drawPairs(grp, played) {
    for (let t = 0; t < 200; t++) {
      const g = shuffle(grp.slice()), ps = [];
      for (let i = 0; i + 1 < g.length; i += 2) ps.push([g[i], g[i + 1]]);
      if (!ps.some(p => played.has(key(p[0], p[1])))) return ps;
    }
    const rec = rest => {
      if (!rest.length) return [];
      const a = rest[0], others = rest.slice(1), c = shuffle(others.filter(b => !played.has(key(a, b))));
      for (const b of c) { const r = rec(others.filter(x => x !== b)); if (r) return [[a, b]].concat(r) }
      return null;
    };
    return rec(shuffle(grp.slice()));
  }

  // fixtures[round] = [{a,b,bo,win|null}] ; force = {a,b,win}
  SIM.simulate = (teams, fixtures, pf, n, force) => {
    const qual = {}, rec = {}, qlists = []; teams.forEach(t => { qual[t] = 0; rec[t] = {} });
    for (let s = 0; s < n; s++) {
      const W = {}, L = {}; teams.forEach(t => { W[t] = 0; L[t] = 0 }); const played = new Set();
      for (let rnd = 1; rnd <= 5; rnd++) {
        const done = new Set(), pairs = [];
        (fixtures[rnd] || []).forEach(f => {
          if (!(f.a in W) || !(f.b in W) || W[f.a] >= 3 || L[f.a] >= 3 || W[f.b] >= 3 || L[f.b] >= 3) return;
          pairs.push([f.a, f.b, f.bo, f.win]); done.add(f.a); done.add(f.b);
        });
        const groups = {};
        teams.forEach(t => { if (!done.has(t) && W[t] < 3 && L[t] < 3) (groups[W[t] + '-' + L[t]] = groups[W[t] + '-' + L[t]] || []).push(t) });
        Object.values(groups).forEach(grp => {
          let ps = drawPairs(grp, played);
          if (!ps) { const g = shuffle(grp.slice()); ps = []; for (let i = 0; i + 1 < g.length; i += 2) ps.push([g[i], g[i + 1]]) }
          ps.forEach(([a, b]) => pairs.push([a, b, boFor(W[a], L[a], W[b], L[b]), null]));
        });
        pairs.forEach(([a, b, bo, win]) => {
          if (win == null) win = (force && ((force.a === a && force.b === b) || (force.a === b && force.b === a))) ? force.win : (Math.random() < pf(a, b, bo) ? a : b);
          const lo = win === a ? b : a; W[win]++; L[lo]++; played.add(key(a, b));
        });
      }
      const q = []; teams.forEach(t => { const k = W[t] + '-' + L[t]; rec[t][k] = (rec[t][k] || 0) + 1; if (W[t] >= 3) { qual[t]++; q.push([t, L[t]]) } });
      qlists.push(q);
    }
    const out = {}; teams.forEach(t => { out[t] = { qual: qual[t] / n, rec: Object.fromEntries(Object.entries(rec[t]).map(([k, v]) => [k, v / n])) } });
    return { teams: out, qlists };
  };

  SIM.knockout = (qlists, pf5) => {
    const title = {}, semis = {}; let n = 0;
    qlists.forEach(q => {
      if (q.length !== 8) return; n++;
      const z = q.filter(x => x[1] === 0).map(x => x[0]), two = q.filter(x => x[1] === 2).map(x => x[0]), one = q.filter(x => x[1] === 1).map(x => x[0]);
      const play = (a, b) => Math.random() < pf5(a, b) ? a : b; let m;
      if (z.length === 2 && two.length === 3 && one.length === 3) {
        shuffle(two); m = [[z[0], two[0]], [z[1], two[1]]]; const rest = shuffle(one.concat([two[2]])); m.push([rest[0], rest[1]], [rest[2], rest[3]]);
      } else { const t = shuffle(q.map(x => x[0])); m = [[t[0], t[1]], [t[2], t[3]], [t[4], t[5]], [t[6], t[7]]] }
      const w4 = m.map(([a, b]) => play(a, b)); w4.forEach(t => semis[t] = (semis[t] || 0) + 1); shuffle(w4);
      const f = [play(w4[0], w4[1]), play(w4[2], w4[3])]; const c = play(f[0], f[1]); title[c] = (title[c] || 0) + 1;
    });
    const norm = o => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, v / (n || 1)]));
    return { title: norm(title), semis: norm(semis) };
  };

  // --- orchestration : calendrier site + resultats officiels lolesports -> etat recalcule ---
  // helpers = {findEvent(m, events), utc, sameTeam(a,b)} (common.js)
  SIM.live = async function (W, events, helpers, opts) {
    opts = opts || {};
    const swiss = W.swiss, blend = W.model.blend, R0 = W.ratings, H = W.h2h || {};
    if (!swiss || swiss.length !== 16 || !R0) return null;
    const R = {}; swiss.forEach(t => R[t] = Object.assign({}, R0[t]));
    const serverDone = new Set(W.done_ids || []);
    const nameOf = en => swiss.find(t => helpers.sameTeam(t, en)) || null;
    const ms = W.matches.filter(m => m.stage === 'Main Event' && /^Round \d/.test(m.round || '')).sort((a, b) => helpers.utc(a.date) - helpers.utc(b.date));
    const fixtures = {}, W_ = {}, L_ = {}; swiss.forEach(t => { W_[t] = 0; L_[t] = 0 });
    let extra = 0, known = 0, knownServer = 0; const upcoming = [], extraDone = [];
    ms.forEach(m => {
      const rnd = +m.round.match(/\d/)[0];
      let t1 = m.t1, t2 = m.t2, state = m.status === 'completed' ? 'completed' : 'unstarted', s1 = m.s1, s2 = m.s2;
      if (m.t1 && m.t2) knownServer++;
      const ev = helpers.findEvent(m, events || []);
      if (ev) {
        state = ev.state; const et = ev.match.teams;
        const n0 = nameOf(et[0].name), n1 = nameOf(et[1].name);
        if (n0 && n1) { t1 = n0; t2 = n1; s1 = et[0].result ? et[0].result.gameWins : null; s2 = et[1].result ? et[1].result.gameWins : null }
      }
      if (!t1 || !t2 || !(t1 in R) || !(t2 in R)) return;
      known++;
      const done = state === 'completed' && s1 != null && s2 != null && s1 !== s2;
      const win = done ? (s1 > s2 ? t1 : t2) : null;
      (fixtures[rnd] = fixtures[rnd] || []).push({ a: t1, b: t2, bo: m.bo, win });
      if (done) {
        const lo = win === t1 ? t2 : t1; W_[win]++; L_[lo]++;
        if (!serverDone.has(m.id)) { extra++; extraDone.push({ t1, t2, s1, s2 }) }
      } else upcoming.push({ m, t1, t2, rnd });
    });
    if (!extra && !(known > knownServer) && !opts.force) return null;       // le serveur est a jour
    // ajustement de rating pour les series terminees que le serveur n'a pas encore integrees
    extraDone.forEach(({ t1, t2, s1, s2 }) => {
      const p = SIM.gameP(t1, t2, R, blend, H), d = K_LIVE * (s1 * (1 - p) - s2 * p); R[t1].bt += d; R[t2].bt -= d;
    });
    const cache = new Map();
    const pf = (a, b, bo) => { const k = a + '>' + b + '#' + bo; let v = cache.get(k); if (v == null) { v = SIM.seriesP(SIM.gameP(a, b, R, blend, H), bo); cache.set(k, v) } return v };
    const N = opts.n || 3000, base = SIM.simulate(swiss, fixtures, pf, N);
    const ko = SIM.knockout(base.qlists, (a, b) => pf(a, b, 5));
    const standings = swiss.map(t => ({ team: t, w: W_[t], l: L_[t], qual: base.teams[t].qual, title: ko.title[t] || 0, semis: ko.semis[t] || 0, rec: base.teams[t].rec,
      power: (W.standings.find(s => s.team === t) || {}).power, region: (W.standings.find(s => s.team === t) || {}).region }));
    standings.sort((x, y) => (y.w - y.l) - (x.w - x.l) || y.w - x.w || y.qual - x.qual);
    // enjeux : victoire / defaite des prochains matchs (en tranches pour ne pas figer l'interface)
    const stakes = {}, next = upcoming.slice(0, opts.maxStakes || 10);
    for (const { m, t1, t2 } of next) {
      await new Promise(r => setTimeout(r, 0));
      const aw = SIM.simulate(swiss, fixtures, pf, opts.nCond || 1000, { a: t1, b: t2, win: t1 }), bw = SIM.simulate(swiss, fixtures, pf, opts.nCond || 1000, { a: t1, b: t2, win: t2 });
      stakes[m.slug] = { t1, t2, records: { a: W_[t1] + '-' + L_[t1], b: W_[t2] + '-' + L_[t2] }, qual_now: { a: base.teams[t1].qual, b: base.teams[t2].qual },
        decisive: W_[t1] === 2 || W_[t2] === 2 || L_[t1] === 2 || L_[t2] === 2,
        if: { a_win: { a: aw.teams[t1].qual, b: aw.teams[t2].qual }, b_win: { a: bw.teams[t1].qual, b: bw.teams[t2].qual } } };
    }
    const pred = {};   // proba de serie recalculee (ratings ajustes) pour les matchs a venir
    upcoming.forEach(({ m, t1, t2 }) => { const g = SIM.gameP(t1, t2, R, blend, H); pred[m.slug] = { t1, t2, game: g, series: SIM.seriesP(g, m.bo) } });
    return { standings, stakes, pred, extra, n: N, at: Date.now() };
  };

  if (typeof module !== 'undefined') module.exports = SIM; else root.SIM = SIM;
})(typeof window !== 'undefined' ? window : globalThis);
