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
    return r.json();
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
  const nk = s => String(s || '').normalize('NFKD').toLowerCase().replace(/[^a-z0-9]/g, '');
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
      const hit = mine.filter(n => names.some(x => x && (x.startsWith(n.slice(0, 4)) || n.startsWith(x.slice(0, 4))))).length;
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

  // ---------- squelette commun ----------
  LP.langBtn = () => { const b = LP.el('button', 'lang', lang.toUpperCase()); b.title = 'FR / EN'; b.onclick = () => { LP.setLang(lang === 'fr' ? 'en' : 'fr'); b.textContent = lang.toUpperCase() }; return b };
  LP.themeBtn = () => { const b = LP.el('button', 'tog', '◐'); b.onclick = LP.toggleTheme; return b };
})();
