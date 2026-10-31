const $ = (id) => document.getElementById(id);
const state = { dataset: 'synthetic', meetings: [], selectedMeeting: null, jobTimer: null,
  graph: { initialized: false, scope: 'global', meetingId: null, network: null, data: null },
  agent: { initialized: false, sending: false, history: [] } };

function renderIcons() {
  if (window.lucide && typeof window.lucide.createIcons === 'function') window.lucide.createIcons();
}
function applyUi(theme, persist = true) {
  const next = theme === 'fixed' ? 'fixed' : 'console';
  document.documentElement.dataset.ui = next;
  document.documentElement.classList.toggle('no-lucide', !window.lucide);
  document.querySelectorAll('[data-ui-choice]').forEach(b =>
    b.setAttribute('aria-pressed', String(b.dataset.uiChoice === next)));
  if (persist) { try { localStorage.setItem('sm-ui', next); } catch (e) {} }
  renderIcons();
  if (state.graph.data && document.getElementById('view-graph')?.classList.contains('active')) {
    renderGraph(state.graph.data);
  }
}
const METHOD_LABELS = {
  bm25: 'BM25', dense: 'Dense · custom', feature_aware: 'Feature-aware',
  french_me5: 'French ME5', multilingual_e5: 'Multilingual E5',
  colbert: 'ColBERT · BGE-M3', graphrag: 'GraphRAG · hybride'
};
const BENCH_ORDER = ['bm25', 'dense', 'feature_aware', 'french_me5', 'multilingual_e5', 'colbert', 'graphrag'];

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function pretty(value) { return JSON.stringify(value, null, 2); }
function toast(message, error=false) {
  const el = $('toast'); el.textContent = message; el.className = `toast show${error ? ' error' : ''}`;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => el.className = 'toast', 3300);
}
async function api(url, options={}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === `view-${name}`));
  document.querySelectorAll('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.view === name));
  const labels = {search:'Recherche sémantique',manager:'Gestion des réunions',datasets:'Gestion des datasets',benchmarks:'Évaluation des modèles',graph:'Graphe des discussions',agent:'Agent sémantique'};
  $('breadcrumbCurrent').textContent = labels[name] || name;
  if (name === 'manager') loadMeetings();
  if (name === 'datasets') renderDatasetCards();
  if (name === 'graph') initGraph();
  if (name === 'agent') initAgent();
}
function fillDatasetSelect(datasets) {
  $('datasetSelect').innerHTML = datasets.map(d => `<option value="${escapeHtml(d.slug)}">${escapeHtml(d.name)}</option>`).join('');
  $('datasetSelect').value = state.dataset;
}
async function refreshState() {
  try {
    const data = await api('/api/state');
    fillDatasetSelect(data.datasets);
    $('deviceLabel').textContent = data.models.cuda_available ? data.models.gpu_name : 'CPU mode · CUDA indisponible';
    $('footerModel').textContent = `CUSTOM ENCODER · ${data.models.custom_model.dimension}D · ${data.models.device.toUpperCase()}`;
    renderDatasetCards(data.datasets);
    await loadMeetings();
    updateScopeSelect();
  } catch (err) { toast(err.message, true); }
}
async function loadMeetings() {
  try {
    const data = await api(`/api/meetings?dataset=${encodeURIComponent(state.dataset)}`);
    state.meetings = data.meetings || [];
    $('meetingCount').textContent = state.meetings.length;
    $('managerStats').textContent = `${state.meetings.length} RÉUNIONS`;
    updateScopeSelect(); renderMeetingList($('meetingFilter').value || '');
  } catch (err) { toast(err.message, true); }
}
function updateScopeSelect() {
  const select = $('scopeSelect');
  const prev = select.value;
  select.innerHTML = '<option value="">Toutes les réunions</option>' + state.meetings.map(m =>
    `<option value="${escapeHtml(m.meeting_id)}">${escapeHtml(m.title || m.meeting_id)}</option>`).join('');
  if (state.meetings.some(m => m.meeting_id === prev)) select.value = prev;
}
function renderMeetingList(filter='') {
  const query = filter.toLocaleLowerCase();
  const rows = state.meetings.filter(m => `${m.title} ${m.project||''} ${m.meeting_id} ${m.style||''}`.toLocaleLowerCase().includes(query));
  $('meetingList').innerHTML = rows.map(m => `<button class="meeting-item ${state.selectedMeeting === m.meeting_id ? 'active' : ''}" data-mid="${escapeHtml(m.meeting_id)}">
    <div class="meeting-title">${escapeHtml(m.title || m.meeting_id)}</div>
    <div class="meeting-sub"><span><i class="style-dot ${m.style === 'messy' ? 'messy' : ''}"></i>${escapeHtml(m.meeting_date || 'Date inconnue')}</span><span>${Number(m.message_count||0)} msg</span></div>
  </button>`).join('') || '<div class="empty-state"><p>Aucune réunion trouvée.</p></div>';
  document.querySelectorAll('.meeting-item').forEach(b => b.addEventListener('click', () => openMeeting(b.dataset.mid)));
}
async function openMeeting(mid) {
  try {
    const data = await api(`/api/meeting/${encodeURIComponent(mid)}?dataset=${encodeURIComponent(state.dataset)}`);
    state.selectedMeeting = mid; renderMeetingList($('meetingFilter').value || '');
    const m = data.meeting;
    $('chatEmpty').classList.add('hidden'); $('chatContent').classList.remove('hidden');
    $('chatHeader').innerHTML = `<div><div class="chat-title">${escapeHtml(m.title || m.meeting_id)}</div><div class="chat-meta">${escapeHtml(m.meeting_date || '')} · ${escapeHtml(m.style || 'custom')} · ${data.messages.length} MESSAGES</div></div><div class="chat-summary">${escapeHtml(m.summary || '')}</div>`;
    $('chatMessages').innerHTML = data.messages.map((row, i) => {
      const jsonId = `chat-json-${i}`;
      return `<div class="chat-row">
        <div class="chat-avatar">${escapeHtml((row.username||'?').slice(0,2).toUpperCase())}</div>
        <div class="chat-bubble"><div class="chat-bubble-head"><b>${escapeHtml(row.username)}</b><time>${escapeHtml(row.timestamp)}</time></div><p>${escapeHtml(row.message)}</p>
          <div class="chat-id">${escapeHtml(row.chunk_id)} <button class="chat-json-toggle" data-json="${jsonId}"><i data-lucide="braces" aria-hidden="true"></i>JSON ↗</button></div>
          <pre class="json-preview" id="${jsonId}">${escapeHtml(pretty(row.source_json))}</pre>
        </div>
      </div>`;
    }).join('');
    document.querySelectorAll('.chat-json-toggle').forEach(b => b.addEventListener('click', () => $(b.dataset.json).classList.toggle('open')));
    renderIcons();
  } catch (err) { toast(err.message, true); }
}
function resultCard(row) {
  const id = `json-${Math.random().toString(36).slice(2)}`;
  return `<article class="result-card"><div class="result-top"><span class="rank-chip">#${row.rank}</span><span class="score">score ${Number(row.score||0).toFixed(4)}</span></div>
    <div class="source-meta"><span class="speaker">${escapeHtml(row.username || '')}</span><span>${escapeHtml(row.timestamp || '')}</span><span>${escapeHtml(row.meeting_title || row.meeting_id || '')}</span></div>
    <p class="result-message">${escapeHtml(row.message || '')}</p>
    <div class="result-bottom"><span>${escapeHtml(row.chunk_id || '')}</span><button class="json-toggle" data-json="${id}"><i data-lucide="braces" aria-hidden="true"></i>VIEW SOURCE JSON ↗</button></div>
    <pre class="json-preview" id="${id}">${escapeHtml(pretty(row.source_json || {}))}</pre></article>`;
}
function bindJsonToggles() {
  document.querySelectorAll('.json-toggle').forEach(b => b.addEventListener('click', () => $(b.dataset.json).classList.toggle('open')));
}
function renderSearch(data) {
  $('searchResults').className = 'results-grid';
  if (data.results_by_method) {
    $('searchResults').innerHTML = Object.entries(data.results_by_method).map(([method, payload]) => {
      if (!payload || payload.available === false) return `<section class="method-column"><div class="method-column-head"><b>${METHOD_LABELS[method] || method}</b><span>UNAVAILABLE</span></div><div class="method-error">${escapeHtml(payload?.error || 'Modèle indisponible.')}</div></section>`;
      return `<section class="method-column"><div class="method-column-head"><b>${METHOD_LABELS[method] || method}</b><span>${payload.candidate_count || 0} CANDIDATES</span></div>${(payload.results || []).map(resultCard).join('') || '<p class="method-error">Aucun résultat.</p>'}</section>`;
    }).join('');
    $('resultCount').textContent = '5 méthodes'; $('searchMeta').textContent = 'Comparaison directe · même corpus';
  } else {
    const rows = data.results || [];
    $('searchResults').innerHTML = rows.map(resultCard).join('') || '<div class="empty-state"><h3>Aucun résultat</h3><p>Essayez une autre formulation ou changez la portée.</p></div>';
    $('resultCount').textContent = rows.length; $('searchMeta').textContent = `${data.candidate_count || 0} chunks candidats · ${data.explanation || data.method}`;
  }
  bindJsonToggles();
  renderIcons();
}
async function runSearch() {
  const query = $('queryInput').value.trim();
  if (!query) { toast('Écrivez une requête avant de lancer la recherche.', true); $('queryInput').focus(); return; }
  const btn = $('searchButton'); btn.disabled = true; btn.innerHTML = 'Searching <span>◌</span>';
  $('searchResults').className = 'results-grid'; $('searchResults').innerHTML = '<div class="empty-state"><div class="progress-spinner"></div><h3>Interrogation de la mémoire…</h3><p>Encodage de la requête et classement des chunks.</p></div>';
  try {
    const data = await api('/api/search', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset:state.dataset,query,method:$('methodSelect').value,top_k:Number($('topKSelect').value),meeting_id:$('scopeSelect').value || null})});
    renderSearch(data);
  } catch (err) { $('searchResults').innerHTML = `<div class="method-error">${escapeHtml(err.message)}</div>`; toast(err.message,true); }
  finally { btn.disabled = false; btn.innerHTML = 'Search <span>↗</span><i class="ui-lucide" data-lucide="arrow-up-right" aria-hidden="true"></i>'; renderIcons(); }
}
function renderDatasetCards(datasets=null) {
  const draw = (items) => {
    $('datasetCards').innerHTML = items.map(d => `<article class="dataset-card ${d.slug === state.dataset ? 'active' : ''}">
      <div class="dataset-card-head"><span class="format-tag ${d.builtin?'green-tag':''}">${d.builtin?'BUILT-IN':'CUSTOM MEMORY'}</span><span class="switcher-dot"></span></div>
      <h3>${escapeHtml(d.name)}</h3><p>${d.style_counts ? (d.style_counts.transcribed ? `${d.style_counts.transcribed} transcribed` : `${d.style_counts.formelle} formelles · ${d.style_counts.messy} messy`) : 'Dataset importé'}</p>
      <div class="dataset-metrics"><div><b>${d.meeting_count}</b><span>MEETINGS</span></div><div><b>${d.message_count.toLocaleString('fr-FR')}</b><span>CHUNKS</span></div><div><b>${d.has_queries?'YES':'NO'}</b><span>LABELS</span></div></div>
      <button data-activate="${escapeHtml(d.slug)}">${d.slug === state.dataset ? 'ACTIVE MEMORY' : 'ACTIVATE MEMORY'}</button></article>`).join('');
    document.querySelectorAll('[data-activate]').forEach(b => b.addEventListener('click', () => activateDataset(b.dataset.activate)));
    renderIcons();
  };
  if (datasets) draw(datasets); else api('/api/state').then(x => draw(x.datasets)).catch(e => toast(e.message,true));
}
async function activateDataset(slug) {
  state.dataset = slug; $('datasetSelect').value = slug; state.selectedMeeting = null;
  state.graph.data = null;
  await loadMeetings(); renderDatasetCards(); toast('Mémoire activée.');
  fillAgentMeetingSelect();
  if (document.getElementById('view-graph')?.classList.contains('active')) initGraph();
}
const METRIC_FORMAT = {
  'Recall@1': v => `${(100 * v).toFixed(1)}%`,
  'Recall@5': v => `${(100 * v).toFixed(1)}%`,
  'Recall@10': v => `${(100 * v).toFixed(1)}%`,
  'MRR@10': v => v.toFixed(3),
  'nDCG@5': v => v.toFixed(3),
};
function maxesFor(rows, keys) {
  const maxes = {};
  keys.forEach(k => { maxes[k] = Math.max(...rows.map(([, m]) => Number(m[k] ?? -Infinity))); });
  return maxes;
}
function metricTds(metrics, keys, maxes, rowCount) {
  return keys.map(k => {
    const value = Number(metrics[k] ?? 0);
    const isMax = rowCount > 1 && maxes[k] > 0 && value === maxes[k];
    return `<td${isMax ? ' class="is-max" title="Meilleur score de la colonne"' : ''}>${METRIC_FORMAT[k](value)}</td>`;
  }).join('');
}
$('meetingsFile').addEventListener('change', () => $('meetingsFilename').textContent = $('meetingsFile').files[0]?.name || 'Aucun fichier sélectionné');
$('importForm').addEventListener('submit', async event => {
  event.preventDefault();
  const file = $('meetingsFile').files[0]; if (!file) return toast('Sélectionnez le JSON des réunions.', true);
  const body = new FormData(); body.append('meetings_file',file); body.append('dataset_name',$('datasetName').value);
  const qf = $('queriesFile').files[0]; if (qf) body.append('queries_file',qf);
  const btn = $('importButton'); btn.disabled = true; btn.textContent = 'Building immutable memory…';
  $('importStatus').className = 'status-box'; $('importStatus').textContent = 'Parsing JSON · encoding messages · extracting features · writing SQLite…';
  try {
    const result = await api('/api/import',{method:'POST',body});
    $('importStatus').className = 'status-box'; $('importStatus').innerHTML = `<b>${escapeHtml(result.name)}</b><br>${result.meeting_count} réunions · ${result.message_count} chunks · ${result.query_count} queries<br>${(result.notes||[]).map(escapeHtml).join('<br>')}`;
    await refreshState(); await activateDataset(result.slug); toast('Dataset importé et mémoire construite.');
  } catch (err) { $('importStatus').className = 'status-box error'; $('importStatus').textContent = err.message; toast(err.message,true); }
  finally { btn.disabled = false; btn.innerHTML = 'Build memory <span>↗</span><i class="ui-lucide" data-lucide="arrow-up-right" aria-hidden="true"></i>'; renderIcons(); }
});
function renderBenchmarkResult(job) {
  $('benchmarkProgress').classList.add('hidden');
  const result = job.result || {};
  if (result.status === 'NO_LABELLED_QUERIES') {
    $('benchmarkResults').innerHTML = `<div class="method-error">${escapeHtml(result.message)}</div>`; return;
  }
  const methods = result.methods || {};
  const rows = Object.entries(methods).filter(([_,v])=>v.metrics)
    .sort((a, b) => (BENCH_ORDER.indexOf(a[0]) + 1 || 99) - (BENCH_ORDER.indexOf(b[0]) + 1 || 99))
    .map(([key,v])=>[key,v.metrics]);
  const keys = ['Recall@1', 'Recall@5', 'Recall@10', 'MRR@10', 'nDCG@5'];
  const maxes = maxesFor(rows, keys);
  $('benchmarkResults').innerHTML = `<div class="panel baseline-panel"><div class="panel-topline"><span class="tiny-label">${escapeHtml(result.dataset)} · ${escapeHtml(result.split)} SPLIT</span><span class="format-tag green-tag">${result.query_count} QUESTIONS</span></div><table class="benchmark-table"><thead><tr><th>RETRIEVAL METHOD</th><th>RECALL@1</th><th>RECALL@5</th><th>RECALL@10</th><th>MRR@10</th><th>NDCG@5</th></tr></thead><tbody>${rows.map(([key,m])=>`<tr><td><span class="method-label"><i class="method-indicator"></i>${METHOD_LABELS[key]||key}</span></td>${metricTds(m, keys, maxes, rows.length)}</tr>`).join('')}</tbody></table><div class="benchmark-note">${rows.length} méthodes scorées. Les requêtes multi-réponses comptent la fraction des chunks cibles retrouvés. Les méthodes indisponibles ou échouées sont listées ci-dessous; les scores ne sont pas inventés.</div>${Object.entries(methods).filter(([_,v])=>!v.metrics).map(([key,v])=>`<div class="method-error">${escapeHtml(METHOD_LABELS[key]||key)} : ${escapeHtml((v.errors||[]).join(' · ') || v.status || 'indisponible')}</div>`).join('')}</div>`;
  renderIcons();
}
async function runBenchmark() {
  const btn = $('runBenchmark'); btn.disabled = true;
  $('benchmarkProgress').classList.remove('hidden'); $('progressLabel').textContent = `Lancement sur ${state.dataset}…`;
  $('benchmarkResults').innerHTML = '';
  try {
    const data = await api('/api/benchmark',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset:state.dataset,split:$('benchmarkSplit').value,methods:['bm25','dense','feature_aware','french_me5','multilingual_e5','colbert','graphrag']})});
    pollJob(data.job_id);
  } catch (err) { $('benchmarkProgress').classList.add('hidden'); btn.disabled=false; toast(err.message,true); }
}
function pollJob(jobId) {
  clearInterval(state.jobTimer);
  state.jobTimer = setInterval(async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      $('progressLabel').textContent = job.status === 'running' ? 'Encodage et classement des requêtes…' : job.status === 'queued' ? 'En attente…' : job.status;
      if (job.status === 'completed') { clearInterval(state.jobTimer); renderBenchmarkResult(job); $('runBenchmark').disabled=false; toast('Benchmark terminé.'); }
      if (job.status === 'failed') { clearInterval(state.jobTimer); $('benchmarkProgress').classList.add('hidden'); $('benchmarkResults').innerHTML = `<div class="method-error">${escapeHtml(job.error)}</div>`; $('runBenchmark').disabled=false; toast(job.error,true); }
    } catch (err) { clearInterval(state.jobTimer); $('runBenchmark').disabled=false; toast(err.message,true); }
  }, 1100);
}

const GRAPH_GROUP_COLORS = {
  project: '#8b6ff0', meeting: '#5eead4', topic: '#6c7bff',
  speaker: '#f4b183', chunk: '#c9d1d9', act: '#8b949e'
};
const GRAPH_GROUP_SIZES = { project: 24, meeting: 17, topic: 13, speaker: 15, chunk: 9, act: 11 };
const REL_COLORS = {
  supports: '#22c55e', contradicts: '#f59e0b', supersedes: '#8b6ff0',
  creates_action: '#5eead4', answers: '#6c7bff', unrelated: '#555', references: '#f4b183'
};
function graphEdgeColor(edge) {
  const rel = (edge.tags || []).find(t => REL_COLORS[t]);
  return rel ? REL_COLORS[rel] : '#2f3a46';
}
function fillGraphMeetingSelect() {
  const select = $('graphMeetingSelect');
  const prev = state.graph.meetingId;
  select.innerHTML = state.meetings.map(m =>
    `<option value="${escapeHtml(m.meeting_id)}">${escapeHtml(m.title || m.meeting_id)}</option>`).join('');
  if (state.meetings.some(m => m.meeting_id === prev)) select.value = prev;
  else if (state.meetings.length) state.graph.meetingId = state.meetings[0].meeting_id;
}
async function initGraph() {
  if (!state.graph.initialized) {
    state.graph.initialized = true;
    document.querySelectorAll('[data-graph-scope]').forEach(b =>
      b.addEventListener('click', () => setGraphScope(b.dataset.graphScope)));
    $('graphMeetingSelect').addEventListener('change', e => {
      state.graph.meetingId = e.target.value;
      if (state.graph.scope === 'meeting') loadGraph('meeting', e.target.value);
    });
    window.addEventListener('resize', () => state.graph.network?.redraw());
  }
  if (!state.meetings.length) await loadMeetings();
  fillGraphMeetingSelect();
  await loadGraph(state.graph.scope, state.graph.scope === 'meeting' ? state.graph.meetingId : null);
}
async function setGraphScope(scope) {
  state.graph.scope = scope === 'meeting' ? 'meeting' : 'global';
  document.querySelectorAll('[data-graph-scope]').forEach(b =>
    b.setAttribute('aria-pressed', String(b.dataset.graphScope === state.graph.scope)));
  const select = $('graphMeetingSelect');
  select.disabled = state.graph.scope === 'global';
  if (state.graph.scope === 'meeting') {
    if (!state.meetings.length) await loadMeetings();
    fillGraphMeetingSelect();
    await loadGraph('meeting', state.graph.meetingId);
  } else {
    await loadGraph('global');
  }
}
async function loadGraph(scope, meetingId) {
  const canvas = $('graphCanvas');
  canvas.innerHTML = '<div class="graph-loading">Construction du graphe…</div>';
  try {
    const dataset = encodeURIComponent(state.dataset);
    const url = scope === 'meeting'
      ? `/api/graph/${encodeURIComponent(meetingId)}?dataset=${dataset}`
      : `/api/graph?dataset=${dataset}`;
    const data = await api(url);
    state.graph.data = data;
    state.graph.scope = data.scope;
    $('graphMeetingSelect').disabled = data.scope !== 'meeting';
    $('graphStats').textContent = `${data.stats.nodes} NŒUDS · ${data.stats.edges} LIENS`;
    $('graphLegend').innerHTML = data.legend.map(l =>
      `<span class="graph-legend-item"><i style="background:${GRAPH_GROUP_COLORS[l.group] || '#888'}"></i>${escapeHtml(l.label)}</span>`).join('');
    renderGraph(data);
  } catch (err) {
    canvas.innerHTML = `<div class="method-error">${escapeHtml(err.message)}</div>`;
    toast(err.message, true);
  }
}
function renderGraph(data) {
  const canvas = $('graphCanvas');
  if (!window.vis || !window.vis.Network) {
    canvas.innerHTML = '<div class="method-error">vis-network indisponible (CDN bloqué).</div>';
    return;
  }
  canvas.innerHTML = '';
  const consoleUi = document.documentElement.dataset.ui === 'console';
  const nodes = new vis.DataSet(data.nodes.map(n => ({
    id: n.id, label: n.label.length > 32 ? `${n.label.slice(0, 30)}…` : n.label, group: n.group,
    x: n.x, y: n.y,
    size: GRAPH_GROUP_SIZES[n.group] || 10,
    title: n.title || n.label,
    color: { background: GRAPH_GROUP_COLORS[n.group] || '#888', border: '#0b0b0b', highlight: { background: '#fff', border: '#fff' } },
    font: { color: consoleUi ? '#a3a3a3' : '#eaf1fb', size: 11, face: 'JetBrains Mono', strokeWidth: 3, strokeColor: '#000' },
    tags: n.tags || [], raw: n
  })));
  const edges = new vis.DataSet(data.edges.map((e, i) => ({
    id: `e${i}`, from: e.from, to: e.to,
    label: e.label || '',
    title: (e.tags || []).join(' · '),
    color: { color: graphEdgeColor(e), highlight: '#fff', opacity: 0.75 },
    width: e.width || 1,
    dashes: (e.tags || []).includes('precedes'),
    arrows: (e.tags || []).some(t => REL_COLORS[t]) ? { to: { enabled: true, scaleFactor: 0.35 } } : undefined,
    font: { color: consoleUi ? '#6b6b6b' : '#8998ad', size: 9, face: 'JetBrains Mono', strokeWidth: 0 },
    smooth: { type: 'continuous', roundness: 0.3 }
  })));
  const isGlobal = data.scope === 'global';
  const options = {
    autoResize: true,
    physics: isGlobal ? { enabled: false } : {
      enabled: true,
      stabilization: { enabled: true, iterations: 220, fit: true },
      barnesHut: { gravitationalConstant: -18000, springLength: 130, springConstant: 0.04, damping: 0.2, avoidOverlap: 0.2 }
    },
    interaction: { hover: true, tooltipDelay: 140, navigationButtons: false, keyboard: { enabled: true } },
    nodes: { shape: 'dot', borderWidth: 1, chosen: true },
    edges: { selectionWidth: 2 },
    layout: isGlobal ? { improvedLayout: false } : { improvedLayout: true }
  };
  state.graph.network = new vis.Network(canvas, { nodes, edges }, options);
  state.graph.network.on('click', params => {
    if (params.nodes.length) inspectGraphNode(params.nodes[0]);
  });
  if (isGlobal) {
    setTimeout(() => state.graph.network.fit({ animation: false }), 60);
  } else {
    state.graph.network.once('stabilizationIterationsDone', () => state.graph.network.fit({ animation: false }));
  }
}
function inspectGraphNode(id) {
  const node = state.graph.data?.nodes.find(n => n.id === id);
  if (!node) return;
  const tags = (node.tags || []).map(t => `<span class="graph-tag">${escapeHtml(t)}</span>`).join('');
  let html = `<div class="tiny-label">${escapeHtml(node.group)}</div><h3>${escapeHtml(node.label)}</h3><div class="graph-tags">${tags}</div>`;
  if (node.speaker || node.time) {
    html += `<div class="graph-meta">${node.speaker ? `<div><span>INTERVENANT</span><b>${escapeHtml(node.speaker)}</b></div>` : ''}${node.time ? `<div><span>HORODATAGE</span><b>${escapeHtml(node.time)}</b></div>` : ''}${node.act ? `<div><span>ACTE</span><b>${escapeHtml(node.act)}</b></div>` : ''}${node.topic ? `<div><span>THÈME</span><b>${escapeHtml(node.topic)}</b></div>` : ''}</div>`;
  }
  if (node.text) html += `<p class="graph-text">${escapeHtml(node.text)}</p>`;
  if (node.group === 'meeting' && node.meeting_id) {
    html += `<button class="primary-button full-button graph-open" data-open-meeting="${escapeHtml(node.meeting_id)}">Ouvrir le graphe de la réunion</button>`;
  }
  $('graphInspector').innerHTML = html;
  const open = document.querySelector('.graph-open');
  if (open) open.addEventListener('click', () => {
    state.graph.meetingId = open.dataset.openMeeting;
    $('graphMeetingSelect').value = state.graph.meetingId;
    setGraphScope('meeting');
  });
}

function agentFormat(text) {
  return escapeHtml(text || '')
    .replace(/\[([A-Za-z0-9:_\-]+)\]/g, '<span class="agent-cite">[$1]</span>')
    .replace(/\n/g, '<br>');
}
function agentMessage(role, html) {
  const wrap = document.createElement('div');
  wrap.className = `agent-message ${role}`;
  wrap.innerHTML = `<div class="agent-bubble">${html}</div>`;
  const box = $('agentMessages');
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
  return wrap;
}
function agentTraceStep(step) {
  const box = $('agentTrace');
  const placeholder = box.querySelector('.graph-hint');
  if (placeholder) placeholder.remove();
  const id = `agent-step-${step.step}`;
  let el = document.getElementById(id);
  if (!el) {
    el = document.createElement('div');
    el.id = id;
    box.appendChild(el);
  }
  el.className = `agent-step ${step.status || ''}`;
  el.innerHTML = `<div class="agent-step-head"><i></i><b>${escapeHtml(step.label || step.step)}</b><span>${escapeHtml(step.status || '')}</span></div>${step.detail ? `<details><summary>détail</summary><pre>${escapeHtml(JSON.stringify(step.detail, null, 2))}</pre></details>` : ''}`;
  box.scrollTop = box.scrollHeight;
}
function agentEvidenceHtml(evidence) {
  const cards = evidence.map((e, i) => {
    const jsonId = `agent-json-${i}-${Math.random().toString(36).slice(2)}`;
    return `<article class="agent-evidence-card">
      <div class="agent-evidence-head"><code>${escapeHtml(e.chunk_id)}</code><span>${escapeHtml(e.username)} · ${escapeHtml(e.meeting_date)} ${escapeHtml(e.timestamp)}</span><span class="agent-tag">${escapeHtml(e.act || '')}${e.status ? ' / ' + escapeHtml(e.status) : ''}</span></div>
      <p>${escapeHtml(e.message)}</p>
      <button type="button" class="agent-json-toggle" data-agent-json="${jsonId}">JSON ↗</button>
      <pre class="agent-json" id="${jsonId}">${escapeHtml(JSON.stringify(e.source_json, null, 2))}</pre>
    </article>`;
  }).join('');
  return `<details class="agent-evidence"><summary>Preuves (${evidence.length})</summary><div class="agent-evidence-list">${cards}</div></details>`;
}
function renderAgentAction(action) {
  const el = $('agentAction');
  if (!action) { el.innerHTML = ''; return; }
  let html = '<div class="tiny-label">ACTION PROPOSÉE</div>';
  if (action.sections) {
    const labels = {contexte:'Contexte', faits:'Faits', chronologie:'Chronologie', decisions:'Décisions', points_ouverts:'Points ouverts', responsables:'Responsables', risques:'Risques', references:'Références'};
    html += '<div class="agent-dossier">';
    for (const name of action.section_order || []) {
      const rows = action.sections[name] || [];
      if (!rows.length) continue;
      html += `<section><h4>${labels[name] || name} <span>${rows.length}</span></h4>`;
      html += rows.map(row => typeof row === 'string'
        ? `<p class="agent-dossier-ref"><code>${escapeHtml(row)}</code></p>`
        : `<p>${escapeHtml(row.statement)} <span class="agent-cite">[${escapeHtml(row.chunk_id)}]</span></p>`).join('');
      html += '</section>';
    }
    if ((action.missing_information || []).length) {
      html += `<p class="agent-missing">Information manquante : ${action.missing_information.map(escapeHtml).join(', ')}</p>`;
    }
    html += '</div>';
  } else if (action.intent === 'draft_email') {
    html += `<div class="agent-email"><div><span>DESTINATAIRE</span><b>${escapeHtml(action.recipient?.name || '—')}</b></div><div><span>EMAIL</span><b>${escapeHtml(action.recipient?.email || 'non disponible')}</b></div><div><span>STATUT</span><b>${escapeHtml(action.recipient?.resolution_status || '')}</b></div><div><span>OBJET</span><b>${escapeHtml(action.subject || '')}</b></div><pre class="agent-json">${escapeHtml(action.body || '')}</pre><p class="agent-missing">Brouillon uniquement · confirmation requise · non exécuté.</p></div>`;
  } else if (action.intent === 'clarify') {
    html += `<p>${escapeHtml(action.question || 'Précision nécessaire.')}</p>`;
  } else if (action.timeline) {
    html += `<div class="agent-timeline">${action.timeline.map(row => `<div><span>${escapeHtml(row.date)} ${escapeHtml(row.timestamp)}</span><p>${escapeHtml(row.statement)} <span class="agent-cite">[${escapeHtml(row.chunk_id)}]</span></p></div>`).join('')}</div>`;
  } else {
    html += `<pre class="agent-json">${escapeHtml(JSON.stringify(action, null, 2))}</pre>`;
  }
  el.innerHTML = html;
}
function renderAgentResult(payload) {
  const llm = payload.llm || {};
  const meta = `${payload.intent || 'answer'} · ${llm.status || 'no-llm'}${llm.elapsed_ms ? ' · ' + llm.elapsed_ms + ' ms' : ''}`;
  const cite = (payload.citations || []).map(c => `<span class="agent-cite">[${escapeHtml(c)}]</span>`).join(' ');
  const html = `<div class="agent-answer-meta">${escapeHtml(meta)}</div>
    <div class="agent-answer-text">${agentFormat(payload.answer)}</div>
    ${cite ? `<div class="agent-cites"><span>Citations</span>${cite}</div>` : ''}
    ${payload.evidence?.length ? agentEvidenceHtml(payload.evidence) : ''}`;
  const wrap = agentMessage('assistant', html);
  wrap.querySelectorAll('.agent-json-toggle').forEach(b => b.addEventListener('click', () => $(b.dataset.agentJson).classList.toggle('open')));
  renderAgentAction(payload.action);
  $('agentTraceTime').textContent = `${payload.trace_ms || 0} ms`;
  $('agentMessages').scrollTop = $('agentMessages').scrollHeight;
}
async function askAgent(event) {
  event.preventDefault();
  if (state.agent.sending) return;
  const query = $('agentQuery').value.trim();
  if (!query) return;
  state.agent.sending = true;
  $('agentSend').disabled = true;
  agentMessage('user', `<div class="agent-answer-text">${escapeHtml(query)}</div>`);
  $('agentQuery').value = '';
  $('agentTrace').innerHTML = '';
  $('agentAction').innerHTML = '';
  $('agentTraceTime').textContent = '…';
  try {
    const response = await fetch('/api/agent', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, dataset: state.dataset, meeting_id: $('agentMeeting').value || null,
                            top_k: Number($('agentTopK').value), use_llm: true,
                            history: state.agent.history.slice(-6)})});
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      let index;
      while ((index = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, index).trim();
        buffer = buffer.slice(index + 1);
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.type === 'step') agentTraceStep(event);
        else if (event.type === 'result') {
          renderAgentResult(event.payload);
          state.agent.history.push({role: 'user', content: query});
          state.agent.history.push({role: 'assistant', content: event.payload.answer || ''});
        }
        else if (event.type === 'error') agentMessage('assistant', `<div class="method-error">${escapeHtml(event.error)}</div>`);
      }
    }
  } catch (err) {
    agentMessage('assistant', `<div class="method-error">${escapeHtml(err.message)}</div>`);
    toast(err.message, true);
  } finally {
    state.agent.sending = false;
    $('agentSend').disabled = false;
    $('agentQuery').focus();
  }
}
function fillAgentMeetingSelect() {
  const select = $('agentMeeting');
  if (!select) return;
  const prev = select.value;
  select.innerHTML = '<option value="">Toutes les réunions</option>' + state.meetings.map(m =>
    `<option value="${escapeHtml(m.meeting_id)}">${escapeHtml(m.title || m.meeting_id)}</option>`).join('');
  if (state.meetings.some(m => m.meeting_id === prev)) select.value = prev;
}
async function initAgent() {
  if (!state.agent.initialized) {
    state.agent.initialized = true;
    $('agentForm').addEventListener('submit', askAgent);
    $('agentQuery').addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('agentForm').requestSubmit(); }
    });
    document.querySelectorAll('[data-agent-query]').forEach(b =>
      b.addEventListener('click', () => { $('agentQuery').value = b.dataset.agentQuery; $('agentForm').requestSubmit(); }));
  }
  if (!state.meetings.length) await loadMeetings();
  fillAgentMeetingSelect();
  try {
    const health = await api('/api/agent/health');
    $('agentModel').textContent = health.model || 'OLLAMA';
    $('agentStatus').textContent = health.status === 'ready'
      ? `LLM READY · ${health.model}` : `LLM ${String(health.status).toUpperCase()}`;
    $('agentStatus').title = health.error || health.models?.join(', ') || '';
  } catch (err) {
    $('agentStatus').textContent = 'LLM UNAVAILABLE';
    $('agentStatus').title = err.message;
  }
}

document.querySelectorAll('.nav-item').forEach(b => b.addEventListener('click', () => showView(b.dataset.view)));
$('searchButton').addEventListener('click', runSearch);
$('queryInput').addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });
document.querySelectorAll('[data-query]').forEach(b => b.addEventListener('click', () => { $('queryInput').value = b.dataset.query; runSearch(); }));
$('meetingFilter').addEventListener('input', e => renderMeetingList(e.target.value));
$('datasetSelect').addEventListener('change', e => activateDataset(e.target.value));
$('refreshState').addEventListener('click', () => { refreshState(); toast('État actualisé.'); });
$('runBenchmark').addEventListener('click', runBenchmark);
document.querySelectorAll('[data-ui-choice]').forEach(b => b.addEventListener('click', () => applyUi(b.dataset.uiChoice)));
applyUi(document.documentElement.dataset.ui, false);
refreshState();
