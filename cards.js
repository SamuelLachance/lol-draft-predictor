/* Fiches equipe / joueur partagees (accueil + pages de match) : tout nom portant data-team / data-player ouvre la fiche. */
(function () {
  const { el, esc, pct, t, tx } = LP;
  const LPC = window.LPC = {};
  const ROLES = ['top', 'jng', 'mid', 'bot', 'sup'], RN = { top: 'Top', jng: 'Jungle', mid: 'Mid', bot: 'Bot', sup: 'Support' };
  const S = { teams: [], players: [], regions: [], worlds: null, tidx: {}, pidx: {}, pname: {}, regidx: {}, roleStats: {}, stack: [], base: LP.base || '' };
  const ea = s => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/"/g, '&quot;');
  const sgn = (v, d = 0) => (v >= 0 ? '+' : '') + v.toFixed(d);

  // ---------- liens a inserer dans les pages ----------
  LPC.team = (name, o = {}) => name ? `<span class="lnk${o.b === false ? '' : ' tb'}" data-team="${ea(name)}">${esc(o.label || name)}</span>` : '–';
  LPC.player = (name, role, o = {}) => name ? `<span class="lnk" data-player="${ea(name)}" data-role="${role || ''}"${o.team ? ` data-pteam="${ea(o.team)}"` : ''}>${esc(name)}</span>` : '–';

  LPC.init = d => {
    Object.assign(S, d);
    S.tidx = Object.fromEntries((S.teams || []).map(x => [x.team, x]));
    S.pidx = {}; S.pname = {}; (S.players || []).forEach(p => { S.pidx[p.player + '|' + p.role] = p; if (!S.pname[p.player] || S.pname[p.player].games < p.games) S.pname[p.player] = p });
    S.regidx = Object.fromEntries((S.regions || []).map(r => [r.region, r]));
    ROLES.forEach(ro => {
      const ps = (S.players || []).filter(p => p.role === ro), col = f => ps.map(f).filter(v => v != null).sort((a, b) => a - b);
      S.roleStats[ro] = { gd15: col(p => p.gd15), kp: col(p => p.kp), kda: col(p => p.kda), dmg: col(p => p.dmg || 0) };
    });
    if (!LPC._on) {
      LPC._on = true;
      document.body.addEventListener('click', e => {
        const pl = e.target.closest('[data-player]'), tm = e.target.closest('[data-team]');
        if (pl) { e.preventDefault(); e.stopPropagation(); LPC.open({ type: 'player', name: pl.dataset.player, role: pl.dataset.role, team: pl.dataset.pteam }) }
        else if (tm) { e.preventDefault(); e.stopPropagation(); LPC.open({ type: 'team', name: tm.dataset.team }) }
      }, true);
      document.addEventListener('keydown', e => { if (e.key === 'Escape') LPC.close() });
      document.addEventListener('lp:lang', () => { const m = document.getElementById('modal'); if (m && m.style.display === 'flex' && S.stack.length) LPC.open(S.stack[S.stack.length - 1], false) });
    }
  };
  LPC.update = d => { Object.assign(S, d) };

  // ---------- modale ----------
  LPC.close = () => { const m = document.getElementById('modal'); if (m) m.style.display = 'none'; S.stack = [] };
  LPC.open = (ref, push) => {
    if (push !== false) S.stack.push(ref);
    let m = document.getElementById('modal');
    if (!m) { m = el('div', 'modal-bg'); m.id = 'modal'; document.body.append(m); m.addEventListener('click', e => { if (e.target === m) LPC.close() }) }
    m.innerHTML = ''; const sheet = el('div', 'sheet'); sheet.style.maxWidth = '720px';
    sheet.append(ref.type === 'player' ? playerCard(ref) : teamCard(ref.name));
    m.append(sheet); m.style.display = 'flex'; m.scrollTop = 0; LP.hydrate(m);
  };
  const back = () => { const b = el('button', 'mx', '‹'); b.title = 'Back'; b.onclick = () => { S.stack.pop(); const prev = S.stack[S.stack.length - 1]; prev ? LPC.open(prev, false) : LPC.close() }; return b };
  const closeX = () => { const b = el('button', 'mx', '✕'); b.style.marginLeft = 'auto'; b.onclick = LPC.close; return b };

  // ---------- briques ----------
  const pctile = (role, stat, v) => { const a = (S.roleStats[role] || {})[stat]; if (!a || !a.length || v == null) return .5; let lo = 0; for (const x of a) if (x < v) lo++; return lo / a.length };
  const color = p => p >= .6 ? 'var(--fav)' : p >= .35 ? 'var(--warn)' : 'var(--bad)';
  const seg = (lbl, val, frac, c) => { const r = el('div', 'brow'); r.innerHTML = `<span class="lbl">${lbl}</span><span class="val" style="color:${c}">${val}</span>`; const bar = el('div', 'bar'), sp = el('span'); sp.style.width = Math.max(2, Math.min(100, frac * 100)) + '%'; sp.style.background = c; bar.append(sp); r.append(bar); return r };
  const tiles = pairs => { const w = el('div', 'mini'); pairs.forEach(([k, v]) => { const d = el('div', 't'); d.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`; w.append(d) }); return w };
  const table = (head, rows) => {
    const mt = el('div', 'mt'), s = el('div', 'scroll'), tb = el('table');
    tb.innerHTML = '<thead><tr>' + head.map(h => `<th class="${h[1] || ''}">${h[0]}</th>`).join('') + '</tr></thead>';
    const b = el('tbody'); rows.forEach(r => { const tr = el('tr'); tr.innerHTML = r; b.append(tr) }); tb.append(b); s.append(tb); mt.append(s); return mt;
  };
  const sec = txt => el('div', 'sec', txt);
  const W = () => S.worlds;
  const matchLink = m => `<a href="${S.base}match/#${m.slug}">${t('page du match', 'match page')} →</a>`;

  // ---------- fiche joueur ----------
  function playerCard(ref) {
    const { name, team } = ref; let role = ref.role;
    const p = (role && S.pidx[name + '|' + role]) || S.pname[name], wrap = el('div');
    role = (p && p.role) || role;
    // parties du Worlds 2026 (depuis les pages de match)
    const wg = [], Wd = W();
    if (Wd) Wd.matches.forEach(m => (m.games || []).forEach(g => {
      ['blue', 'red'].forEach(side => (g[side + '_players'] || []).forEach((pn, i) => {
        if (pn === name) wg.push({ champion: g[side + '_picks'][i], role: ROLES[i], win: (g.blue_win === 1) === (side === 'blue'), team: g[side], opp: g[side === 'blue' ? 'red' : 'blue'], slug: m.slug, n: g.n, side });
      }));
    }));
    if (!p) {
      const h = el('div', 'mhead'); if (S.stack.length > 1) h.append(back()); h.append(el('h2', null, esc(name)), role ? el('span', 'tag', (RN[role] || role)) : '', closeX());
      const b = el('div', 'mbody');
      if (team) b.append(el('div', 'muted', `${t('Équipe', 'Team')} : ${LPC.team(team)}`));
      b.append(el('div', 'muted', t('Joueur hors des ligues majeures suivies (pas de rating détaillé).', 'Player outside the tracked major leagues (no detailed rating).')));
      if (wg.length) { b.append(sec('Worlds 2026')); b.append(worldsGames(wg)) }
      wrap.append(h, b); return wrap;
    }
    const h = el('div', 'mhead'); if (S.stack.length > 1) h.append(back());
    h.append(el('h2', null, esc(p.player)), el('span', 'tag', RN[role] || role));
    h.append(Object.assign(el('span'), { innerHTML: LPC.team(p.team, { b: false }) })); h.append(el('span', 'tag', p.league), closeX());
    const b = el('div', 'mbody');
    b.append(el('div', null, `<div style="color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px">Rating</div><div class="rtg" style="font-size:30px">${p.rating.toFixed(1)}</div>`));
    const rank = (S.players || []).filter(x => x.role === role).findIndex(x => x.player === p.player) + 1;
    b.append(el('div', 'muted', `<span style="font-size:12px">#${rank} ${t('au poste', 'in role')} ${RN[role]} · ${(S.players || []).filter(x => x.role === role).length} ${t('joueurs notés', 'rated players')} · ${p.games} games · ${p.wins}-${p.losses}</span>`));
    b.append(sec(t('Construction du rating', 'How the rating is built')));
    const mag = Math.max(Math.abs(p.c_ind), Math.abs(p.c_team), 8);
    b.append(seg(t('Individuel', 'Individual'), sgn(p.c_ind, 1), Math.abs(p.c_ind) / mag, p.c_ind >= 0 ? 'var(--gold)' : 'var(--bad)'));
    b.append(seg(t('Contexte équipe', 'Team context'), sgn(p.c_team, 1), Math.abs(p.c_team) / mag, p.c_team >= 0 ? 'var(--accent)' : 'var(--bad)'));
    b.append(tiles([['Winrate', pct(p.wr)], [t('Bilan', 'Record'), `${p.wins}-${p.losses}`], ['KP', pct(p.kp)], ['KDA', p.kda], ['GD@15', sgn(p.gd15)], [t('Part dégâts', 'Dmg share'), pct(p.dmg)]]));
    b.append(sec(t('vs pairs du rôle (percentile)', 'vs role peers (percentile)')));
    [['GD@15', 'gd15', p.gd15], [t('Part de dégâts', 'Dmg share'), 'dmg', p.dmg], ['KP', 'kp', p.kp], ['KDA', 'kda', p.kda]].forEach(([lbl, s, v]) => {
      const pc = pctile(role, s, v), disp = s === 'gd15' ? sgn(v) : (s === 'dmg' || s === 'kp') ? pct(v) : v; b.append(seg(lbl, disp + ` · ${Math.round(pc * 100)}e`, pc, color(pc)));
    });
    if (p.champs && p.champs.length) { b.append(sec(t('Pool de champions', 'Champion pool'))); b.append(table([['Champion'], ['G', 'num'], ['WR', 'num'], ['KDA', 'num']], p.champs.map(c => `<td>${LP.champ(c.champion)}</td><td class="num">${c.games}</td><td class="num">${pct(c.wr)}</td><td class="num">${c.kda}</td>`))) }
    if (wg.length) { b.append(sec('Worlds 2026')); b.append(worldsGames(wg)) }
    if (p.recent && p.recent.length) { b.append(sec(t('Dernières games', 'Recent games'))); b.append(table([['Date'], ['Champion'], ['R'], ['K/D/A', 'num'], ['GD@15', 'num'], ['vs']], p.recent.map(g => `<td class="muted">${g.date}</td><td>${LP.champ(g.champion)}</td><td class="${g.win ? 'ok' : 'no'}">${g.win ? 'W' : 'L'}</td><td class="num">${g.k}/${g.d}/${g.a}</td><td class="num">${sgn(g.gd15)}</td><td>${LPC.team(g.opp, { b: false })}</td>`))) }
    wrap.append(h, b); return wrap;
  }
  function worldsGames(wg) {
    const w = wg.filter(x => x.win).length;
    const box = el('div'); box.append(el('div', 'muted', `<span style="font-size:12px">${wg.length} games · ${w}W-${wg.length - w}L</span>`));
    box.append(table([['Champion'], [t('Rés.', 'Res')], ['vs'], [t('Match', 'Match')]], wg.map(g => `<td>${LP.champ(g.champion)}</td><td class="${g.win ? 'ok' : 'no'}">${g.win ? 'W' : 'L'}</td><td>${LPC.team(g.opp, { b: false })}</td><td><a href="${S.base}match/#${g.slug}">G${g.n}</a></td>`)));
    return box;
  }

  // ---------- fiche equipe ----------
  function teamCard(name) {
    const x = S.tidx[name], Wd = W(), pr = Wd && Wd.profiles && Wd.profiles[name], wrap = el('div');
    const h = el('div', 'mhead'); if (S.stack.length > 1) h.append(back());
    h.append(el('h2', null, esc(name)));
    if (x) h.append(el('span', 'tag', x.league), el('span', 'tag', x.region), el('span', 'muted', `${x.wins}-${x.losses} · ${pct(x.wr)}`));
    h.append(closeX());
    const b = el('div', 'mbody');
    if (!x && !pr) { b.append(el('div', 'muted', t('Équipe hors des ligues majeures suivies : pas de fiche détaillée.', 'Team outside the tracked major leagues: no detailed card.'))); wrap.append(h, b); return wrap }
    // --- Worlds 2026 : statut, prochain match, resultats
    if (Wd) {
      const st = (Wd.standings || []).find(s => s.team === name);
      const mine = Wd.matches.filter(m => m.t1 === name || m.t2 === name);
      const nxt = mine.find(m => m.status !== 'completed' && m.t1 && m.t2);
      if (st || mine.length) {
        const bx = el('div'); bx.style.cssText = 'background:var(--sunk);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:8px';
        let html = `<div class="sec" style="margin-top:0">Worlds 2026</div>`;
        if (st) html += `<div class="kv"><div><div class="k">Swiss</div><div class="v">${st.w}-${st.l}</div></div><div><div class="k">${t('Qualif. top 8', 'Top-8 odds')}</div><div class="v">${st.w >= 3 ? '✔' : st.l >= 3 ? '✘' : pct(st.qual)}</div></div><div><div class="k">${t('Titre', 'Title')}</div><div class="v">${pct(st.title, 1)}</div></div></div>`;
        if (nxt) {
          const opp = nxt.t1 === name ? nxt.t2 : nxt.t1, p = nxt.pred ? (nxt.t1 === name ? nxt.pred.series : 1 - nxt.pred.series) : null;
          html += `<div style="margin-top:8px;font-size:13px">${t('Prochain match', 'Next match')} : <b>vs ${LPC.team(opp, { b: false })}</b> · ${LP.fmtDay(LP.utc(nxt.date))} ${LP.fmtTime(LP.utc(nxt.date))} · Bo${nxt.bo}${p != null ? ` · <b>${pct(p, 1)}</b> ${t('de victoire', 'win prob.')}` : ''} · ${matchLink(nxt)}</div>`;
        }
        const done = mine.filter(m => m.status === 'completed');
        if (done.length) html += `<div style="margin-top:6px;font-size:12px" class="muted">${t('Résultats', 'Results')} : ` + done.map(m => { const me1 = m.t1 === name, s1 = me1 ? m.s1 : m.s2, s2 = me1 ? m.s2 : m.s1; return `<a href="${S.base}match/#${m.slug}" class="${s1 > s2 ? 'ok' : 'no'}">${s1 > s2 ? 'W' : 'L'} ${s1}-${s2} vs ${esc(me1 ? m.t2 : m.t1)}</a>` }).join(' · ') + '</div>';
        bx.innerHTML = html; b.append(bx);
      }
    }
    if (x) {
      b.append(tiles([['Power', x.power], [t('Niveau joueurs', 'Player level'), x.pbt != null ? Math.round(x.pbt * 173.7) : '–'], ['Roster TS', x.ts_mu != null ? x.ts_mu : '–'], ['L10', x.form], [t('Forme pond.', 'Weighted form'), x.fw != null ? Math.round(x.fw * 100) : '–'], [t('Ligue', 'League'), sgn(x.lg_off)]]));
      b.append(sec(t('Power = niveau de la ligue + écart de l\'équipe', 'Power = league level + team deviation')));
      b.append(el('div', 'muted', `<span style="font-size:11px">${x.region} (${sgn(x.lg_off)}) ${x.dev >= 0 ? '+' : '−'} ${Math.abs(x.dev)} = <b style="color:var(--gold)">${x.power}</b> · ${t('dernier match', 'last game')} ${x.last}</span>`));
    }
    if (pr) {
      if (pr.bullets && pr.bullets.length) { b.append(sec(t('Identité de jeu', 'Playstyle'))); b.append(el('ul', 'bul', pr.bullets.map(u => `<li>${esc(tx(u.t))}</li>`).join(''))) }
      const f1 = v => v == null ? '–' : v.toFixed(1), f2 = v => v == null ? '–' : v.toFixed(2);
      b.append(tiles([[t('Durée moy.', 'Avg length'), f1(pr.len) + ' min'], ['Kills/min', f2(pr.kpm)], [t('Dragons/game', 'Dragons/game'), f2(pr.dragons)], ['Barons/game', f2(pr.barons)],
        [t('Win% bleu', 'Blue win%'), pr.blue_g ? pct(pr.blue_w / pr.blue_g) : '–'], [t('Win% rouge', 'Red win%'), pr.red_g ? pct(pr.red_w / pr.red_g) : '–']]));
    }
    // --- roster
    const roster = (x && x.roster) || [];
    if (roster.length) {
      b.append(sec(t('Roster', 'Roster')));
      b.append(table([['Role'], [t('Joueur', 'Player')], ['Rating', 'num'], ['KP', 'num'], ['KDA', 'num'], ['GD@15', 'num']], roster.map(m => { const pp = S.pidx[m.player + '|' + m.role];
        return `<td><span class="tag">${m.role.toUpperCase()}</span></td><td><b>${LPC.player(m.player, m.role, { team: name })}</b></td><td class="rtg num">${pp ? pp.rating.toFixed(1) : '–'}</td><td class="num">${pp ? pct(pp.kp) : '–'}</td><td class="num">${pp ? pp.kda : '–'}</td><td class="num">${pp ? sgn(pp.gd15) : '–'}</td>` })));
    }
    // --- pools de champions par role (profil Worlds)
    if (pr && pr.roles) {
      b.append(sec(t('Champions habituels par rôle (depuis mars, pondéré récent)', 'Usual champions by role (since March, recency-weighted)')));
      b.append(table([['Role'], [t('Joueur', 'Player')], [t('Top 3 champions', 'Top 3 champions')]], ROLES.map(ro => {
        const pl = (roster.find(m => m.role === ro) || {}).player;
        return `<td><span class="tag">${RN[ro]}</span></td><td>${pl ? LPC.player(pl, ro, { team: name }) : '–'}</td><td style="white-space:normal">${pr.roles[ro].pool.slice(0, 3).map(c => `<span style="display:inline-flex;margin-right:10px;gap:5px;align-items:center">${LP.champ(c.champion)}<span class="muted">${pct(c.w)}${c.games >= 3 ? ' · ' + pct(c.wins / c.games) : ''}</span></span>`).join('')}</td>` })));
      if (pr.bans_p1 && pr.bans_p1.length) b.append(el('div', 'muted', `<span style="font-size:12px">${t('Bans habituels (phase 1)', 'Usual bans (phase 1)')} : ${pr.bans_p1.slice(0, 5).map(c => `${LP.champ(c.champion, { ban: 1 })} <span class="muted">${pct(c.rate)}</span>`).join(' · ')}</span>`));
    }
    if (x && x.matches && x.matches.length) {
      b.append(sec(t('Derniers matchs (pick du modèle vs résultat)', 'Recent matches (model pick vs result)')));
      b.append(table([['Date'], [t('Match', 'Match')], [t('Modèle', 'Model')], [t('Rés.', 'Res')], [t('OK', 'Hit')]], x.matches.map(m => { const ok = (m.pred >= .5) === (m.win === 1);
        return `<td class="muted">${m.date}</td><td>${LPC.team(m.opp, { b: false })} <span class="muted" style="font-size:10px">(${m.side})</span></td><td><span class="chip fav">${pct(m.pred)}</span></td><td class="${m.win ? 'ok' : 'no'}">${m.win ? 'W' : 'L'}</td><td class="${ok ? 'ok' : 'no'}">${ok ? '✓' : '✗'}</td>` })));
    }
    wrap.append(h, b); return wrap;
  }
})();
