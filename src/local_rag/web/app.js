const $ = selector => document.querySelector(selector);
const status = $('#status');
const operationBar = $('#operation-bar');
const result = $('#result');
const answer = $('#answer');
const citations = $('#citations');
const documents = $('#documents');
const queryButton = $('#query-button');
const clearButton = $('#clear-documents');
const selectionCount = $('#selection-count');
const fileInput = $('#file');
const auditPanel = $('#citation-audit');
const auditList = $('#audits');
let indexedDocuments = [];
let selectedIds = new Set();
let queuedFiles = [];
let currentCitations = [];
let usedCitations = new Set();
let currentAnswer = '';
let querying = false;
let uploading = false;
let managing = false;
let uploadLimit = null;
let jevMode = 'off';
let queryController = null;
let readerDocument = null;
let readerPage = 1;
let readerController = null;
let readerObjectUrl = null;
let readerZoom = 0;

function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'icon'); svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', '#i-' + name); svg.append(use); return svg;
}
async function request(url, options = {}) {
  let response;
  try { response = await fetch(url, options); }
  catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('Cannot reach Context Foundry. Start the app and try again.');
  }
  let data;
  try { data = await response.json(); }
  catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('The app returned an unreadable response. Try again.');
  }
  if (!response.ok) {
    if (response.status === 503) throw new Error('The model could not finish this request. Check that your configured model service is running, then try again.');
    throw new Error(typeof data.detail === 'string' ? data.detail : 'This request could not be completed. Check your input and try again.');
  }
  return data;
}
function setStatus(message, tone = 'ready') { status.textContent = message; operationBar.dataset.tone = tone; }
function uploadStatus(message, tone = 'ready') { $('#upload-status').textContent = message; $('#upload-status').dataset.tone = tone; }
function scheduleProgress(stages, update) {
  return stages.map(([delay, ...message]) => setTimeout(() => update(...message), delay));
}
function clearProgress(timers) { timers.forEach(clearTimeout); }
function queryProgress(title, note) { $('#loading-title').textContent = title; $('#loading-note').textContent = note; }
function updateSelectionState() {
  const count = selectedIds.size;
  const busy = querying || uploading || managing;
  selectionCount.textContent = count ? `Searching ${count} selected document${count === 1 ? '' : 's'}` : indexedDocuments.length ? 'Select a document in your library to ask about it.' : 'Add a document to get started.';
  queryButton.disabled = busy || count === 0 || !$('#question').value.trim();
  clearButton.disabled = busy || indexedDocuments.length === 0;
  $('#select-all').disabled = busy || !indexedDocuments.length || count === indexedDocuments.length;
  $('#upload-button').disabled = busy || !queuedFiles.length;
  fileInput.disabled = busy;
  for (const node of documents.querySelectorAll('input, button')) node.disabled = busy;
  for (const node of document.querySelectorAll('.suggestion')) node.disabled = busy;
  $('#empty-action').disabled = busy;
  if (!currentAnswer) {
    const ready = indexedDocuments.length > 0;
    $('#empty-title').textContent = ready ? 'What do you want to understand?' : 'Start with a document.';
    $('#empty-note').textContent = ready ? 'Ask about a detail, connect ideas across documents, or use one of the question ideas above.' : 'Add something you want to understand: a report, a paper, or your own notes.';
    $('#empty-action').replaceChildren(document.createTextNode(ready ? 'Write a question' : 'Choose documents'), icon('arrow'));
  }
}
function renderDocuments() {
  documents.replaceChildren(); documents.setAttribute('aria-busy', 'false');
  $('#library-count').textContent = indexedDocuments.length;
  if (!indexedDocuments.length) {
    const empty = document.createElement('p'); empty.className = 'empty-library';
    empty.textContent = 'No documents yet. Add your first file above.'; documents.append(empty);
  }
  for (const item of indexedDocuments) {
    const row = document.createElement('div'); row.className = 'document';
    const checkbox = document.createElement('input'); checkbox.type = 'checkbox';
    checkbox.className = 'document-select'; checkbox.id = 'document-' + item.document_id;
    checkbox.value = item.document_id; checkbox.checked = selectedIds.has(item.document_id);
    checkbox.addEventListener('change', () => {
      const hadSelection = selectedIds.size > 0;
      checkbox.checked ? selectedIds.add(item.document_id) : selectedIds.delete(item.document_id);
      updateSelectionState();
      if (checkbox.checked && !hadSelection && matchMedia('(max-width: 760px)').matches) setLibraryCollapsed(true);
    });
    const info = document.createElement('div');
    const label = document.createElement('label'); label.htmlFor = checkbox.id; label.textContent = item.filename;
    const meta = document.createElement('small');
    meta.textContent = /\.pdf$/i.test(item.filename) ? `${item.pages} page${item.pages === 1 ? '' : 's'} · Ready to ask` : 'Text document · Ready to ask';
    label.append(meta); info.append(label);
    if (/\.pdf$/i.test(item.filename) && item.original_available) {
      const view = document.createElement('button'); view.className = 'document-preview'; view.type = 'button'; view.textContent = 'View PDF';
      view.setAttribute('aria-label', 'View ' + item.filename); view.addEventListener('click', () => openReader(item, 1)); info.append(view);
    }
    const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'icon-button';
    remove.setAttribute('aria-label', 'Remove ' + item.filename); remove.title = 'Remove document'; remove.append(icon('close'));
    remove.addEventListener('click', () => removeDocuments(item));
    row.append(checkbox, info, remove); documents.append(row);
  }
  updateSelectionState();
}
async function refreshDocuments(selectNew = false) {
  const previousIds = new Set(indexedDocuments.map(item => item.document_id));
  const items = await request('/api/documents');
  selectedIds = new Set(items.filter(item => selectedIds.has(item.document_id) || (selectNew && !previousIds.has(item.document_id))).map(item => item.document_id));
  indexedDocuments = items; renderDocuments();
  if (selectNew && items.length && matchMedia('(max-width: 760px)').matches) setLibraryCollapsed(true);
}
function setLibraryCollapsed(collapsed) {
  $('.library').dataset.collapsed = collapsed;
  $('#toggle-library').setAttribute('aria-expanded', String(!collapsed));
  $('#toggle-library').textContent = collapsed ? 'Manage documents' : 'Hide documents';
}
$('#toggle-library').addEventListener('click', () => setLibraryCollapsed($('.library').dataset.collapsed !== 'true'));
async function refreshProviderState() {
  try {
    const health = await request('/health');
    jevMode = health.jev_mode || 'off';
    uploadLimit = Number(health.max_upload_mb) || null;
    $('#format-note').textContent = 'PDF, Markdown, or text' + (uploadLimit ? ` · ${uploadLimit} MB per file` : '');
    $('#provider-label').textContent = 'Workspace connected'; $('#provider-state').dataset.ready = 'true';
    let privacy = health.local_models === 'true' ? 'Files stay in this workspace. Your models run on this computer.' : 'Files stay in this workspace. Your configured model providers receive questions and document text.';
    const auditState = $('#audit-state'); auditState.dataset.mode = jevMode;
    if (jevMode === 'observe' || jevMode === 'repair') {
      auditState.hidden = false;
      $('#audit-state-label').textContent = jevMode === 'repair' ? 'Jev repair on' : 'Jev checks on';
      auditState.title = jevMode === 'repair' ? 'TypeSafe checks claim support and may repair weak citations.' : 'TypeSafe checks claim support against cited excerpts.';
      privacy += jevMode === 'repair' ? ' Jev sends generated claims and relevant excerpt text to TypeSafe and may test alternative retrieved excerpts.' : ' Jev sends generated claims and cited excerpt text to TypeSafe.';
    } else auditState.hidden = true;
    $('#privacy-state').textContent = privacy;
  } catch { $('#provider-label').textContent = 'Workspace unavailable'; }
}
function renderQueue() {
  const queue = $('#queued-files'); queue.replaceChildren(); queue.hidden = !queuedFiles.length;
  queuedFiles.forEach(file => { const item = document.createElement('p'); item.textContent = file.name; queue.append(item); });
}
function queueFiles(files) {
  if (querying || uploading || managing) return;
  queuedFiles = []; const errors = [];
  for (const file of files) {
    if (!/\.(pdf|md|txt)$/i.test(file.name)) errors.push(`${file.name}: choose a PDF, Markdown, or text file.`);
    else if (!file.size) errors.push(`${file.name} is empty. Choose a file with text.`);
    else if (uploadLimit && file.size > uploadLimit * 1024 * 1024) errors.push(`${file.name} is too large. Choose a file under ${uploadLimit} MB.`);
    else queuedFiles.push(file);
  }
  renderQueue(); uploadStatus(errors.join(' '), errors.length ? 'error' : 'ready'); updateSelectionState();
}
fileInput.addEventListener('change', () => queueFiles(fileInput.files));
const dropzone = $('#dropzone');
dropzone.addEventListener('dragover', event => { event.preventDefault(); if (!querying && !uploading && !managing) dropzone.dataset.drag = 'true'; });
dropzone.addEventListener('dragleave', () => { dropzone.dataset.drag = 'false'; });
dropzone.addEventListener('drop', event => { event.preventDefault(); dropzone.dataset.drag = 'false'; queueFiles(event.dataTransfer.files); });
// Keep a file dropped outside the upload area from navigating away.
window.addEventListener('dragover', event => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); });
window.addEventListener('drop', event => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); });
$('#upload-form').addEventListener('submit', async event => {
  event.preventDefault(); if (!queuedFiles.length || querying || uploading || managing) return;
  uploading = true; updateSelectionState(); let added = 0; const failed = []; const messages = [];
  try {
    for (let index = 0; index < queuedFiles.length; index++) {
      const file = queuedFiles[index]; const form = new FormData(); form.append('file', file);
      const position = queuedFiles.length > 1 ? ` (${index + 1} of ${queuedFiles.length})` : '';
      uploadStatus(`Reading ${file.name}${position}. Extracting text and document structure.`);
      const uploadProgress = scheduleProgress([
        [9000, `Checking page text and OCR for ${file.name}${position}. Large or scanned PDFs take longer.`],
        [24000, `Building searchable sections for ${file.name}${position}. Local embeddings may take a little time.`],
      ], uploadStatus);
      try {
        const data = await request('/api/documents', { method: 'POST', body: form });
        selectedIds.add(data.document_id); added++;
      } catch (error) { failed.push(file); messages.push(`${file.name}: ${error.message}`); }
      finally { clearProgress(uploadProgress); }
    }
    queuedFiles = failed; fileInput.value = ''; renderQueue();
    if (added) { clearAnswer(); await refreshDocuments(); }
    const summary = added ? `${added} document${added === 1 ? '' : 's'} added and selected. ` : '';
    uploadStatus(summary + messages.join(' '), messages.length ? 'error' : 'ready');
    if (added && !messages.length) {
      setStatus('Your documents are ready. Write a question or choose an idea above.');
      if (matchMedia('(max-width: 760px)').matches) setLibraryCollapsed(true);
    }
  } catch (error) { uploadStatus(error.message, 'error'); }
  finally { uploading = false; updateSelectionState(); }
});
$('#question').addEventListener('input', updateSelectionState);
$('#question').addEventListener('keydown', event => { if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); if (!queryButton.disabled) $('#query-form').requestSubmit(); } });
if (/Mac|iPhone|iPad/.test(navigator.platform)) $('#keyboard-hint').textContent = '⌘ + Enter to ask';
document.querySelectorAll('.suggestion').forEach(button => button.addEventListener('click', () => { $('#question').value = button.dataset.question; updateSelectionState(); $('#question').focus(); }));
$('#empty-action').addEventListener('click', () => {
  if (indexedDocuments.length) $('#question').focus();
  else { setLibraryCollapsed(false); fileInput.click(); }
});
$('#select-all').addEventListener('click', () => { selectedIds = new Set(indexedDocuments.map(item => item.document_id)); renderDocuments(); });

function clearAnswer() {
  currentAnswer = ''; currentCitations = []; usedCitations.clear(); answer.replaceChildren(); citations.replaceChildren();
  clearAuditDecorations();
  result.hidden = true; $('#empty-answer').hidden = false; $('#copy-answer').hidden = true;
  $('#source-empty').hidden = false; $('#source-tabs').hidden = true; citations.hidden = true;
  $('#source-count').textContent = 'No passages yet'; auditPanel.hidden = true;
  $('#source-empty h3').textContent = 'The page behind the answer.';
  $('#source-empty p').textContent = 'Click a citation in your answer to read its passage here, then check the original PDF page.';
}
function normalizeHeadings(container, minimumLevel) {
  for (const heading of container.querySelectorAll('h1, h2, h3, h4, h5, h6')) {
    const level = Math.min(6, Number(heading.tagName.slice(1)) + minimumLevel - 1);
    const replacement = document.createElement('h' + level);
    replacement.append(...heading.childNodes); heading.replaceWith(replacement);
  }
}
function linkCitations() {
  usedCitations = new Set();
  const walker = document.createTreeWalker(answer, NodeFilter.SHOW_TEXT); const nodes = [];
  while (walker.nextNode()) { if (!walker.currentNode.parentElement.closest('a, code, pre, button')) nodes.push(walker.currentNode); }
  for (const node of nodes) {
    const fragment = document.createDocumentFragment(); let position = 0; let found = false;
    for (const match of node.textContent.matchAll(/\[(\d+)\]/g)) {
      const id = Number(match[1]); if (!currentCitations[id - 1]) continue;
      found = true; usedCitations.add(id); fragment.append(document.createTextNode(node.textContent.slice(position, match.index)));
      const button = document.createElement('button'); button.type = 'button'; button.className = 'citation-ref'; button.textContent = id;
      button.dataset.citation = id; button.setAttribute('aria-pressed', 'false'); button.setAttribute('aria-controls', 'citations');
      const item = currentCitations[id - 1]; button.setAttribute('aria-label', `Read passage ${id} from ${item.source}, page ${item.page}`);
      button.addEventListener('click', () => showSource(id, true)); fragment.append(button); position = match.index + match[0].length;
    }
    if (found) { fragment.append(document.createTextNode(node.textContent.slice(position))); node.replaceWith(fragment); }
  }
}
function showSource(id, focus = false) {
  const item = currentCitations[id - 1]; if (!item) return;
  document.querySelectorAll('.source-tab, .citation-ref').forEach(node => node.setAttribute('aria-pressed', String(Number(node.dataset.citation) === id)));
  citations.replaceChildren(); citations.hidden = false;
  const type = document.createElement('span'); type.className = 'source-type'; type.textContent = usedCitations.has(id) ? 'Cited in your answer' : 'Related passage';
  const name = document.createElement('h3'); name.className = 'source-name'; name.textContent = item.source;
  const page = document.createElement('p'); page.className = 'source-page'; page.textContent = /\.pdf$/i.test(item.source) ? `Passage ${id} · Page ${item.page}` : `Passage ${id} · Text document`;
  const text = document.createElement('div'); text.className = 'evidence-text'; text.innerHTML = item.text_html;
  normalizeHeadings(text, 4);
  citations.append(type, name, page, text);
  const doc = indexedDocuments.find(document => document.document_id === item.document_id);
  if (doc && doc.original_available) {
    const actions = document.createElement('div'); actions.className = 'source-actions';
    if (/\.pdf$/i.test(item.source)) {
      const view = document.createElement('button'); view.type = 'button'; view.className = 'button button-secondary'; view.append(icon('book'), document.createTextNode('View PDF page ' + item.page));
      view.addEventListener('click', () => openReader(doc, item.page)); actions.append(view);
    } else {
      const download = document.createElement('a'); download.className = 'button button-secondary'; download.href = `/api/documents/${encodeURIComponent(doc.document_id)}/original`; download.download = doc.filename; download.textContent = 'Download original'; actions.append(download);
    }
    citations.append(actions);
  } else {
    const note = document.createElement('p'); note.className = 'source-unavailable'; note.textContent = 'Add this document again to view the original file.'; citations.append(note);
  }
  if (focus) { citations.tabIndex = -1; citations.focus({ preventScroll: true }); citations.scrollIntoView({ block: 'nearest', behavior: 'auto' }); }
}
function renderSources() {
  const tabs = $('#source-tabs'); tabs.replaceChildren(); tabs.hidden = !currentCitations.length;
  $('#source-empty').hidden = !!currentCitations.length;
  const found = currentCitations.length; const cited = usedCitations.size;
  $('#source-count').textContent = found ? (cited ? `${cited} cited \u00b7 ${found} found` : `${found} found`) : 'No passages found';
  currentCitations.forEach((item, index) => {
    const tab = document.createElement('button'); tab.type = 'button'; tab.className = 'source-tab'; tab.textContent = index + 1; tab.dataset.citation = index + 1;
    tab.setAttribute('aria-label', `Read passage ${index + 1}: ${item.source}, page ${item.page}`); tab.setAttribute('aria-pressed', 'false'); tab.setAttribute('aria-controls', 'citations');
    tab.addEventListener('click', () => showSource(index + 1)); tabs.append(tab);
  });
  if (currentCitations.length) showSource([...usedCitations][0] || 1);
  else {
    citations.hidden = true;
    $('#source-empty h3').textContent = 'No matching passages.';
    $('#source-empty p').textContent = 'Try a more specific question or add a document that covers this topic.';
  }
}
function auditWasRepaired(audit) {
  return !!audit.original && audit.original.citations.join(',') !== audit.citations.join(',');
}
function auditTone(audit) { return auditWasRepaired(audit) ? 'repaired' : audit.suggested_action; }
function auditLabel(audit) {
  return { keep: 'Likely supported', withhold: 'Evidence concern', review: 'Needs review', unavailable: 'Check unavailable', repaired: 'Citation repaired' }[auditTone(audit)];
}
function clearAuditDecorations() {
  answer.querySelectorAll('.claim-status').forEach(node => node.remove());
  answer.querySelectorAll('.audited-claim').forEach(node => node.classList.remove('audited-claim'));
}
function normalizedText(value) { return value.replace(/\s+/g, ' ').trim(); }
function decorateAuditedClaims(audits) {
  clearAuditDecorations(); const usedBlocks = new Set();
  const blocks = [...answer.querySelectorAll('p, li, td, blockquote')];
  for (const audit of audits) {
    const expected = normalizedText(audit.text);
    const block = blocks.find(node => {
      if (usedBlocks.has(node)) return false;
      const copy = node.cloneNode(true); copy.querySelectorAll('button').forEach(button => button.remove());
      return normalizedText(copy.textContent).includes(expected);
    });
    if (!block) continue;
    usedBlocks.add(block); block.classList.add('audited-claim');
    const badge = document.createElement('button'); badge.type = 'button'; badge.className = 'claim-status';
    badge.dataset.action = auditTone(audit); badge.textContent = auditLabel(audit);
    badge.setAttribute('aria-label', `${auditLabel(audit)} for claim ${audit.claim_id}. Open answer support check.`);
    badge.addEventListener('click', () => {
      auditPanel.open = true;
      $(`#claim-audit-${audit.claim_id}`)?.scrollIntoView({
        block: 'nearest',
        behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
      });
    });
    block.append(document.createTextNode(' '), badge);
  }
}
function appendAuditMetric(value, label, tone) {
  const metric = document.createElement('div'); metric.className = 'audit-metric'; metric.dataset.tone = tone;
  const number = document.createElement('strong'); number.textContent = value;
  const description = document.createElement('span'); description.textContent = label;
  metric.append(number, description); $('#audit-metrics').append(metric);
}
function appendProbabilityRows(container, assessment) {
  const labels = { supported: 'Support', contradicted: 'Contradiction', insufficient: 'Insufficient evidence' };
  const rows = document.createElement('div'); rows.className = 'audit-probabilities';
  for (const [relation, label] of Object.entries(labels)) {
    const value = assessment.probabilities[relation]; const row = document.createElement('div'); row.className = 'probability-row'; row.dataset.relation = relation;
    const name = document.createElement('label'); name.textContent = label;
    const progress = document.createElement('progress'); progress.max = 1; progress.value = value; progress.setAttribute('aria-label', `${label} ${(value * 100).toFixed(1)} percent`);
    const percentage = document.createElement('output'); percentage.textContent = `${(value * 100).toFixed(1)}%`;
    row.append(name, progress, percentage); rows.append(row);
  }
  container.append(rows);
}
function renderAudits(data) {
  auditList.replaceChildren(); clearAuditDecorations(); $('#audit-metrics').replaceChildren();
  auditPanel.hidden = !data.audit_mode || data.audit_mode === 'off';
  if (auditPanel.hidden) return;
  const audits = data.audits || []; const counts = { keep: 0, attention: 0, repaired: 0, unavailable: 0 };
  for (const audit of audits) {
    const tone = auditTone(audit);
    if (tone === 'repaired') counts.repaired++;
    else if (tone === 'keep') counts.keep++;
    else if (tone === 'unavailable') counts.unavailable++;
    else counts.attention++;
  }
  appendAuditMetric(counts.keep, 'Likely supported', 'keep');
  appendAuditMetric(counts.attention, 'Needs attention', 'withhold');
  appendAuditMetric(counts.repaired, 'Citations repaired', 'repaired');
  appendAuditMetric(counts.unavailable, 'Check unavailable', 'unavailable');
  const summaryParts = [];
  if (counts.keep) summaryParts.push(`${counts.keep} supported`);
  if (counts.repaired) summaryParts.push(`${counts.repaired} repaired`);
  if (counts.attention) summaryParts.push(`${counts.attention} needs attention`);
  if (counts.unavailable) summaryParts.push(`${counts.unavailable} unavailable`);
  $('#audit-title').textContent = data.audit_mode === 'repair' ? 'Answer support check · repair on' : 'Answer support check';
  $('#audit-summary').textContent = summaryParts.join(' · ') || 'No generated claims to check.';
  $('#audit-note').textContent = data.audit_mode === 'repair' ? 'Jev compares every generated claim with its cited excerpt. Repair mode may replace a weak citation with a stronger retrieved passage; it never rewrites the claim.' : 'Jev compares every generated claim with its cited excerpt. These checks measure citation support, not general factual accuracy or answer completeness.';
  $('#audit-privacy-note').textContent = data.audit_mode === 'repair' ? 'For this answer, TypeSafe received generated claims, cited excerpt text, and any alternative retrieved excerpts tested for repair. The original files were not uploaded.' : 'For this answer, TypeSafe received generated claims and their cited excerpt text. The original files were not uploaded.';
  auditPanel.open = counts.attention > 0 || counts.repaired > 0 || counts.unavailable > 0;
  if (!audits.length) { const empty = document.createElement('p'); empty.className = 'audit-unavailable'; empty.textContent = 'The answer contained no generated claims to check.'; auditList.append(empty); return; }
  for (const audit of audits) {
    const tone = auditTone(audit); const item = document.createElement('article'); item.className = 'claim-audit'; item.id = `claim-audit-${audit.claim_id}`;
    const head = document.createElement('div'); head.className = 'audit-claim-head';
    const number = document.createElement('p'); number.className = 'audit-claim-number'; number.textContent = `Claim ${audit.claim_id}`;
    const action = document.createElement('span'); action.className = 'audit-action'; action.dataset.action = tone; action.textContent = auditLabel(audit); head.append(number, action);
    const claim = document.createElement('p'); claim.className = 'audit-claim-text'; claim.textContent = audit.text;
    const meta = document.createElement('p'); meta.className = 'audit-citations';
    const citationsText = audit.citations.map(id => `[${id}]`).join(' '); const model = audit.model || audit.requested_model;
    meta.textContent = `Citation${audit.citations.length === 1 ? '' : 's'} ${citationsText}${model ? ` · ${model}` : ''}${Number.isFinite(audit.elapsed_seconds) ? ` · ${audit.elapsed_seconds.toFixed(2)}s` : ''}`;
    item.append(head, claim, meta);
    if (audit.assessment) appendProbabilityRows(item, audit.assessment);
    else { const unavailable = document.createElement('p'); unavailable.className = 'audit-unavailable'; unavailable.textContent = 'The support check could not be completed. Review the cited passage directly.'; item.append(unavailable); }
    if (audit.original) {
      const original = audit.original.citations.map(id => `[${id}]`).join(' '); const attempts = (audit.repair_attempts || []).length;
      const repair = document.createElement('div'); repair.className = 'repair-callout';
      const route = document.createElement('p'); route.className = 'repair-route';
      const note = document.createElement('p');
      if (auditWasRepaired(audit)) {
        route.textContent = `${original} original → ${citationsText} replacement`;
        note.textContent = `A stronger retrieved passage passed the support threshold after ${attempts} alternative check${attempts === 1 ? '' : 's'}.`;
      } else {
        route.textContent = `${original} retained`;
        note.textContent = `No alternative passage passed the support threshold after ${attempts} check${attempts === 1 ? '' : 's'}.`;
      }
      repair.append(route, note); item.append(repair);
    }
    const details = document.createElement('details'); details.className = 'audit-technical';
    const technicalSummary = document.createElement('summary'); technicalSummary.textContent = 'Technical audit record';
    const trace = document.createElement('pre'); trace.textContent = JSON.stringify(audit, null, 2); details.append(technicalSummary, trace); item.append(details); auditList.append(item);
  }
  decorateAuditedClaims(audits);
}
$('#query-form').addEventListener('submit', async event => {
  event.preventDefault(); const question = $('#question').value.trim();
  if (querying || uploading || managing || !selectedIds.size || !question) return;
  const querySources = indexedDocuments.filter(item => selectedIds.has(item.document_id)).map(item => item.filename);
  querying = true; queryController = new AbortController(); updateSelectionState();
  $('#loading-state').hidden = false; $('#empty-answer').hidden = true;
  queryProgress('Searching your selected documents', 'Looking for passages that match your question.');
  $('.answer-card').setAttribute('aria-busy', 'true'); setStatus('Looking through your selected documents…', 'working');
  $('#answer-title').textContent = currentAnswer ? 'Previous answer' : 'Your answer';
  const started = Date.now(); $('#elapsed').textContent = '0 seconds';
  const timer = setInterval(() => { $('#elapsed').textContent = `${Math.floor((Date.now() - started) / 1000)} seconds`; }, 1000);
  const progressTimers = scheduleProgress([
    [6000, 'Reading the strongest passages', 'Comparing the retrieved text with each part of your question.'],
    [18000, 'Writing a source-backed answer', 'The local model is organizing the answer and its citation numbers.'],
    [35000, jevMode === 'off' ? 'Finishing your answer' : 'Checking answer support', jevMode === 'off' ? 'The local model is still working. Long documents and larger models can take more time.' : 'The local answer and TypeSafe citation check may take a little longer.'],
  ], queryProgress);
  try {
    const data = await request('/api/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: queryController.signal, body: JSON.stringify({ question, document_ids: [...selectedIds] }) });
    currentAnswer = data.answer; currentCitations = data.citations;
    answer.innerHTML = data.answer_html; normalizeHeadings(answer, 3); linkCitations(); renderSources(); renderAudits(data);
    $('#answer-scope').textContent = 'Asked about: ' + querySources.join(', ');
    $('#question-echo').textContent = question; result.hidden = false; $('#copy-answer').hidden = false;
    $('#answer-note').textContent = currentCitations.length ? 'Follow a numbered citation to check the passage behind a claim.' : 'Try a more specific question, select another document, or add a source that covers this topic.';
    setStatus(currentCitations.length ? 'Answer ready. Click a citation to check the source.' : 'No relevant passages found. Try another question or document.');
    if (matchMedia('(max-width: 760px)').matches) result.scrollIntoView({ block: 'start', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  } catch (error) {
    setStatus(error.name === 'AbortError' ? 'Request stopped. You can edit your question and try again.' : error.message, error.name === 'AbortError' ? 'ready' : 'error');
  } finally {
    clearInterval(timer); clearProgress(progressTimers); querying = false; queryController = null; $('#loading-state').hidden = true;
    $('#empty-answer').hidden = !!currentAnswer; $('.answer-card').setAttribute('aria-busy', 'false'); updateSelectionState();
    $('#answer-title').textContent = 'Your answer';
  }
});
$('#cancel-query').addEventListener('click', () => queryController?.abort());
$('#copy-answer').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(currentAnswer); setStatus('Answer copied, including citation numbers.'); }
  catch { setStatus('The browser could not copy the answer. Select the answer text and copy it.', 'error'); }
});
async function confirmRemoval(item) {
  const dialog = $('#remove-dialog'); dialog.returnValue = 'cancel';
  $('#remove-title').textContent = item ? 'Remove document?' : 'Remove all documents?';
  $('#remove-note').textContent = item ? `${item.filename} and its stored original will be removed from this workspace. Your file outside this app is unaffected.` : 'All documents and their stored originals will be removed from this workspace. Your files outside this app are unaffected.';
  $('#confirm-remove').textContent = item ? 'Remove document' : 'Remove all documents';
  return new Promise(resolve => { dialog.addEventListener('close', () => resolve(dialog.returnValue === 'remove'), { once: true }); dialog.showModal(); });
}
async function removeDocuments(item = null) {
  if (querying || uploading || managing || !indexedDocuments.length) return;
  managing = true; updateSelectionState();
  try {
    if (!await confirmRemoval(item)) return;
    await request(item ? `/api/documents/${encodeURIComponent(item.document_id)}` : '/api/documents', { method: 'DELETE' });
    clearAnswer(); await refreshDocuments(); setStatus(item ? `${item.filename} removed.` : 'All documents removed. Add a document to start again.');
    if (!indexedDocuments.length) setLibraryCollapsed(false);
  } catch (error) { setStatus(error.message, 'error'); }
  finally { managing = false; updateSelectionState(); }
}
clearButton.addEventListener('click', () => removeDocuments());
async function loadReaderPage(page) {
  if (!readerDocument) return;
  readerPage = Math.min(readerDocument.pages, Math.max(1, Math.trunc(Number(page)) || 1));
  readerController?.abort(); const controller = new AbortController(); readerController = controller;
  $('#reader-page').value = readerPage; $('#previous-page').disabled = readerPage === 1; $('#next-page').disabled = readerPage === readerDocument.pages;
  const image = $('#reader-image'); image.hidden = true; image.alt = `Original page ${readerPage} of ${readerDocument.filename}`;
  $('#reader-status').textContent = `Loading page ${readerPage}…`; $('#reader-status').hidden = false;
  try {
    const response = await fetch(`/api/documents/${encodeURIComponent(readerDocument.document_id)}/pages/${readerPage}`, { signal: controller.signal });
    if (!response.ok) { let data = {}; try { data = await response.json(); } catch {} throw new Error(data.detail || 'This page could not be loaded. Try downloading the original PDF.'); }
    const blob = await response.blob(); if (controller.signal.aborted) return;
    if (readerObjectUrl) URL.revokeObjectURL(readerObjectUrl);
    readerObjectUrl = URL.createObjectURL(blob);
    image.onload = () => { image.hidden = false; $('#reader-status').hidden = true; };
    image.onerror = () => { $('#reader-status').textContent = 'This page could not be displayed. Download the original PDF to open it in your reader.'; };
    image.src = readerObjectUrl; $('.reader-scroll').scrollTop = 0;
  } catch (error) { if (error.name !== 'AbortError') $('#reader-status').textContent = error.message; }
}
function openReader(item, page) {
  setReaderZoom(0);
  readerDocument = item; $('#reader-title').textContent = item.filename; $('#reader-total').textContent = `of ${item.pages}`;
  $('#reader-page').max = item.pages; $('#download-original').href = `/api/documents/${encodeURIComponent(item.document_id)}/original`;
  $('#download-original').download = item.filename; $('#pdf-reader').showModal(); loadReaderPage(page);
}
$('#close-reader').addEventListener('click', () => $('#pdf-reader').close());
$('#pdf-reader').addEventListener('close', () => { readerController?.abort(); if (readerObjectUrl) URL.revokeObjectURL(readerObjectUrl); readerObjectUrl = null; $('#reader-image').removeAttribute('src'); readerDocument = null; });
$('#previous-page').addEventListener('click', () => loadReaderPage(readerPage - 1));
$('#next-page').addEventListener('click', () => loadReaderPage(readerPage + 1));
$('#reader-page').addEventListener('change', event => loadReaderPage(event.target.value));
function setReaderZoom(value) {
  readerZoom = Math.max(0, Math.min(3, value));
  $('#reader-image').dataset.zoom = readerZoom;
  $('#zoom-label').textContent = ['Fit width', '150%', '200%', '300%'][readerZoom];
  $('#zoom-out').disabled = readerZoom === 0; $('#zoom-in').disabled = readerZoom === 3;
}
$('#zoom-out').addEventListener('click', () => setReaderZoom(readerZoom - 1));
$('#zoom-in').addEventListener('click', () => setReaderZoom(readerZoom + 1));
Promise.allSettled([refreshDocuments(true), refreshProviderState()]).then(results => {
  if (results[0].status === 'rejected') {
    documents.setAttribute('aria-busy', 'false'); documents.replaceChildren();
    const note = document.createElement('p'); note.className = 'empty-library';
    note.textContent = 'Your library could not be loaded. Refresh the page to try again.'; documents.append(note); setStatus(results[0].reason.message, 'error');
  }
});
