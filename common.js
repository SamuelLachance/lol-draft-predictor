/* Socle commun : theme, langue FR/EN, icones champions (DDragon), donnees live (lolesports) */
(function () {
  const LP = window.LP = {};
  const base = document.body.dataset.base || '';
  LP.base = base;
  LP.$ = s => document.querySelector(s);
  LP.el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e };
  LP.esc = s => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
  LP.pct = (x, d = 0) => x == null ? '–' : (x * 100).toFixed(d) + '%';
  const store = { get(k) { try { return localStorage.getItem(k) } catch (e) { return null } }, set(k, v) { try { localStorage.setItem(k, v) } catch (e) { } } };

  // ---------- langue ----------
  let lang = store.get('lp_lang') || ((navigator.language || 'en').toLowerCase().startsWith('fr') ? 'fr' : 'en');
  LP.lang = () => lang;
  LP.t = (fr, en) => lang === 'fr' ? fr : en;
  LP.tx = o => o == null ? '' : (typeof o === 'string' ? o : (o[lang] || o.fr || o.en || ''));
  LP.setLang = l => { lang = l; store.set('lp_lang', l); document.documentElement.lang = l; document.dispatchEvent(new Event('lp:lang')) };
  document.documentElement.lang = lang;

  // ---------- theme ----------
  const th = store.get('lp_theme'); if (th) document.documentElement.dataset.t = th;
  LP.toggleTheme = () => { const r = document.documentElement; r.dataset.t = r.dataset.t === 'light' ? '' : 'light'; store.set('lp_theme', r.dataset.t || '') };

  // ---------- dates ----------
  LP.utc = s => s ? Date.parse(String(s).replace(' ', 'T') + (/[zZ]|[+-]\d\d:?\d\d$/.test(s) ? '' : 'Z')) : NaN;
  LP.fmtTime = ts => new Date(ts).toLocaleTimeString(lang === 'fr' ? 'fr-CA' : 'en-US', { hour: '2-digit', minute: '2-digit' });
  LP.fmtDay = ts => new Date(ts).toLocaleDateString(lang === 'fr' ? 'fr-CA' : 'en-US', { weekday: 'short', day: 'numeric', month: 'short' });
  LP.ago = ts => { const s = Math.max(0, (Date.now() - ts) / 1000); if (s < 90) return LP.t('à l\'instant', 'just now'); if (s < 3600) return Math.round(s / 60) + ' min'; return Math.round(s / 3600) + ' h' };

  // ---------- donnees ----------
  LP.json = async (f, bust) => {
    const r = await fetch(base + 'data/' + f + '.json' + (bust ? '?t=' + Date.now() : ''), { cache: bust ? 'no-store' : 'default' });
    if (!r.ok) throw new Error(f + ' ' + r.status);
    const txt = await r.text();
    try { return JSON.parse(txt) } catch (e) { return JSON.parse(txt.replace(/:NaN\b/g, ':null').replace(/,NaN\b/g, ',null').replace(/\[NaN\b/g, '[null').replace(/-?Infinity\b/g, 'null')) }   // filet de securite : NaN invalide en JSON
  };

  // ---------- icones champions ----------
  const DD = 'https://ddragon.leagueoflegends.com';
  const ALIAS = { 'Wukong': 'MonkeyKing', 'Renata Glasc': 'Renata', 'Nunu & Willump': 'Nunu', 'Dr. Mundo': 'DrMundo', "Bel'Veth": 'Belveth', "Cho'Gath": 'Chogath', "Kai'Sa": 'Kaisa', "Kha'Zix": 'Khazix', "Kog'Maw": 'KogMaw', "Rek'Sai": 'RekSai', "Vel'Koz": 'Velkoz', "K'Sante": 'KSante', 'LeBlanc': 'Leblanc', "Fiddlesticks": 'Fiddlesticks' };
  let ddv = null, nameToId = null, idToName = {};
  const ddReady = (async () => {
    try {
      const v = (await (await fetch(DD + '/api/versions.json')).json())[0]; ddv = v;
      const d = (await (await fetch(DD + '/cdn/' + v + '/data/en_US/champion.json')).json()).data;
      nameToId = {}; Object.values(d).forEach(c => { nameToId[c.name] = c.id; idToName[c.id] = c.name });
    } catch (e) { nameToId = {} }
  })();
  const idOf = n => (nameToId && nameToId[n]) || ALIAS[n] || String(n).replace(/['. &]/g, '');
  LP.champName = id => idToName[id] || id;
  LP.champ = (name, o = {}) => {
    if (!name) return '<span class="muted">–</span>';
    return `<span class="champ${o.lg ? ' lg' : ''}${o.ban ? ' ban' : ''}"><img data-c="${LP.esc(name)}" alt="">${o.noname ? '' : `<span>${LP.esc(name)}</span>`}</span>`;
  };
  LP.hydrate = async root => {
    await ddReady; if (!ddv) return;
    (root || document).querySelectorAll('img[data-c]').forEach(i => {
      if (i.dataset.done) return; i.dataset.done = 1;
      i.onerror = () => { i.style.visibility = 'hidden' };
      i.src = DD + '/cdn/' + ddv + '/img/champion/' + idOf(i.dataset.c) + '.png';
    });
  };

  // ---------- API lolesports (officielle, CORS ouvert) : calendrier + games en direct ----------
  const ES = 'https://esports-api.lolesports.com/persisted/gw/', KEY = '0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z', WORLDS = '98767975604431411';
  const esGet = async (p) => { const r = await fetch(ES + p, { headers: { 'x-api-key': KEY } }); if (!r.ok) throw new Error('es ' + r.status); return r.json() };
  const nk = s => String(s || '').replace(/ø/gi, 'o').normalize('NFKD').toLowerCase().replace(/[^a-z0-9]/g, '');
  LP.nk = nk;
  // memes equipes ? egalite ou inclusion (>=3 car.) : 'Gen.G' ~ 'Gen.G Esports', 'LØS' ~ 'LOS' ; 'Team Liquid' !~ 'Team Secret Whales'
  LP.sameTeam = (a, b) => { const x = nk(a), y = nk(b); return !!x && !!y && (x === y || (Math.min(x.length, y.length) >= 3 && (x.includes(y) || y.includes(x)))) };
  LP.esSchedule = async () => {
    const d = await esGet('getSchedule?hl=en-US&leagueId=' + WORLDS);
    return d.data.schedule.events.filter(e => e.startTime >= '2026-10-01' && e.match);
  };
  // associe un match du site (date UTC + equipes) a un evenement lolesports
  LP.findEvent = (m, events) => {
    const ts = LP.utc(m.date); let best = null, bs = 1e18;
    const mine = [nk(m.t1), nk(m.t2)].filter(Boolean);
    events.forEach(e => {
      const dt = Math.abs(Date.parse(e.startTime) - ts); if (dt > 3 * 3600e3) return;
      const names = e.match.teams.flatMap(t => [nk(t.name), nk(t.code)]);
      const hit = mine.filter(n => names.some(x => x && LP.sameTeam(n, x))).length;
      const sc = dt - hit * 4 * 3600e3; if (sc < bs) { bs = sc; best = e }
    });
    return best;
  };
  LP.esDetails = async id => (await esGet('getEventDetails?hl=en-US&id=' + id)).data.event;
  LP.liveWindow = async (gameId) => {
    for (let back = 60; back <= 120; back += 10) {
      const st = new Date(Math.floor((Date.now() - back * 1000) / 10000) * 10000).toISOString();
      const r = await fetch('https://feed.lolesports.com/livestats/v1/window/' + gameId + '?startingTime=' + st);
      if (r.ok) { const d = await r.json(); if (d.frames && d.frames.length) return d }
    }
    return null;
  };

  // flux livestats rejouable : fenetre a un instant PASSE (multiple de 10 s, sinon 400). null = pas encore commence / indisponible
  LP.windowAt = async (gid, ms) => {
    const st = new Date(Math.floor(ms / 10000) * 10000).toISOString();
    try { const r = await fetch('https://feed.lolesports.com/livestats/v1/window/' + gid + '?startingTime=' + st); if (!r.ok) return null;
      const tx = await r.text(); if (!tx) return null; const d = JSON.parse(tx); return d.frames && d.frames.length ? d : null } catch (e) { return null }
  };
  // debut du game (+-10 s) : le flux ne repond qu'apres le debut -> recherche dichotomique (meme methode que l'entrainement du modele live)
  LP.findGameStart = async (gid, loMs, hiMs) => {
    const q = ms => LP.windowAt(gid, ms); let a = Math.floor(loMs / 10000) * 10000, b = Math.floor(hiMs / 10000) * 10000;
    if (!(await q(b))) return null; if (await q(a)) return a;
    while (b - a > 10000) { const m = a + Math.round((b - a) / 2 / 10000) * 10000; if (m <= a || m >= b) break; if (await q(m)) b = m; else a = m }
    return b;
  };

  // ---------- squelette commun ----------
  LP.langBtn = () => { const b = LP.el('button', 'lang', lang.toUpperCase()); b.title = 'FR / EN'; b.onclick = () => { LP.setLang(lang === 'fr' ? 'en' : 'fr'); b.textContent = lang.toUpperCase() }; return b };
  LP.themeBtn = () => { const b = LP.el('button', 'tog', '◐'); b.onclick = LP.toggleTheme; return b };
})();

/* ---------- tableaux triables : clic sur un en-tete (1er clic : numerique decroissant / texte croissant ; 2e clic : inverse) ---------- */
(function () {
  const states = new Map();      // cle de tableau -> {col, dir} : conserve le tri quand la page se re-rend
  const MISSING = /^(–|-|—|\s*)$/;
  function parse(cell) {
    if (cell.dataset.sort !== undefined && cell.dataset.sort !== '') { const n = Number(cell.dataset.sort); return Number.isNaN(n) ? { t: cell.dataset.sort } : { n } }
    const x = cell.textContent.replace(/\u00a0/g, ' ').trim();
    if (MISSING.test(x)) return null;
    if (x === '✔') return { n: 1.01 }; if (x === '✘') return { n: -1 };
    if (/^\d{4}-\d{2}-\d{2}/.test(x)) return { t: x };
    let m = x.match(/^(\d+)-(\d+)$/); if (m) { const w = +m[1], l = +m[2]; return { n: (w + l ? w / (w + l) : 0) + w * 1e-6 } }
    m = x.match(/^([+−-]?\d[\d,]*(?:\.\d+)?)\s*(%|[A-Za-zé]{1,6})?(?:\s|$)/);
    if (m) return { n: parseFloat(m[1].replace(/,/g, '').replace('−', '-')) };
    return { t: x.toLowerCase() };
  }
  function apply(table, col, dir) {
    const tb = table.tBodies[0]; if (!tb) return;
    const rows = [...tb.rows];
    const th = table.tHead.rows[0].cells[col];
    let vals = rows.map(r => parse(r.cells[col] || document.createElement('td')));
    const numeric = vals.filter(v => v && v.n !== undefined).length >= 0.6 * Math.max(1, vals.filter(Boolean).length);
    const idx = rows.map((r, i) => i);
    idx.sort((a, b) => {
      const va = vals[a], vb = vals[b];
      if (!va && !vb) return +rows[a].dataset.i0 - +rows[b].dataset.i0;
      if (!va) return 1; if (!vb) return -1;                       // valeurs manquantes toujours en bas
      let c;
      if (th.classList.contains('rk')) c = +rows[a].dataset.i0 - +rows[b].dataset.i0;
      else if (numeric) c = (va.n !== undefined ? va.n : -Infinity) - (vb.n !== undefined ? vb.n : -Infinity);
      else c = String(va.t !== undefined ? va.t : va.n).localeCompare(String(vb.t !== undefined ? vb.t : vb.n), undefined, { numeric: true });
      if (c === 0) return +rows[a].dataset.i0 - +rows[b].dataset.i0;
      return dir === 'asc' ? c : -c;
    });
    idx.forEach(i => tb.append(rows[i]));
    [...tb.rows].forEach((r, i) => { const k = r.querySelector('td.rk'); if (k) k.textContent = i + 1 });
    [...table.tHead.rows[0].cells].forEach((h, i) => { const a = h.querySelector('.arr'); if (a) a.textContent = i === col ? (dir === 'asc' ? '▲' : '▼') : '' });
    table._numeric = numeric;
  }
  const keyOf = t => ((t.closest('.panel,.mt,.sheet,.pbody') || document.body).querySelector('h3,h2,.sec')?.textContent || '') + '|' + [...t.tHead.rows[0].cells].map(c => c.textContent.trim()).join('|');
  function init(table) {
    if (table.dataset.sortable || !table.tHead || !table.tBodies[0] || !table.tHead.rows[0]) return;
    table.dataset.sortable = '1';
    [...table.tBodies[0].rows].forEach((r, i) => r.dataset.i0 = i);
    const key = keyOf(table);
    [...table.tHead.rows[0].cells].forEach((th, i) => {
      th.classList.add('sortable'); th.title = LP.t('Cliquer pour trier', 'Click to sort');
      const a = document.createElement('span'); a.className = 'arr'; th.append(a);
      th.addEventListener('click', e => {
        if (e.target.closest('[data-team],[data-player]')) return;
        const prev = states.get(key); let dir;
        if (prev && prev.col === i) dir = prev.dir === 'desc' ? 'asc' : 'desc';
        else { const vs = [...table.tBodies[0].rows].map(r => parse(r.cells[i] || document.createElement('td'))).filter(Boolean); dir = vs.filter(v => v.n !== undefined).length >= 0.6 * Math.max(1, vs.length) && !th.classList.contains('rk') ? 'desc' : 'asc' }
        states.set(key, { col: i, dir }); apply(table, i, dir);
      });
    });
    const st = states.get(key); if (st) apply(table, st.col, st.dir);
  }
  const scan = root => (root.matches && root.matches('table') ? [root] : []).concat([...(root.querySelectorAll ? root.querySelectorAll('table') : [])]).forEach(init);
  const start = () => {
    scan(document.body);
    new MutationObserver(ms => ms.forEach(m => m.addedNodes.forEach(n => { if (n.nodeType === 1) scan(n) }))).observe(document.body, { childList: true, subtree: true });
  };
  if (document.body) start(); else document.addEventListener('DOMContentLoaded', start);
})();
