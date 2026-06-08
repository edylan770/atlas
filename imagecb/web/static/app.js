/** Imagecb web UI — vanilla JS, no build step required. */

const EXAMPLES = [
  "Screenshots of dashboards from Q3_Review.pptx",
  "Charts showing revenue growth",
  "Only images modified this month",
  "Logos on white backgrounds",
];

const state = {
  sessionId: null,
  searchEventId: null,
  messages: [],
  loading: false,
};

function userIdHeader() {
  const uid = localStorage.getItem("imagecb.userId");
  return uid ? { "X-User-Id": uid } : {};
}

async function recordInteraction(imageId, interactionType, rank) {
  if (!state.searchEventId) return;
  try {
    await api("/api/telemetry/interaction", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...userIdHeader() },
      body: JSON.stringify({
        search_event_id: state.searchEventId,
        image_id: imageId,
        interaction_type: interactionType,
        rank: rank ?? null,
      }),
    });
  } catch {
    /* non-fatal */
  }
}

const $ = (id) => document.getElementById(id);

function escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}

/** Minimal markdown: **bold**, bullets, paragraphs. */
function renderMarkdown(text) {
  const lines = text.split("\n");
  let html = "";
  let inList = false;
  for (const line of lines) {
    const t = line.trim();
    if (!t) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      continue;
    }
    if (t.startsWith("• ") || t.startsWith("- ")) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += `<li>${formatInline(t.slice(2))}</li>`;
    } else {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += `<p>${formatInline(t)}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html;
}

function formatInline(s) {
  return escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
      if (Array.isArray(detail)) {
        detail = detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
      }
    } catch {
      /* ignore */
    }
    throw new Error(String(detail));
  }
  return res.json();
}

async function refreshStatus() {
  try {
    const s = await api("/api/status");
    $("indexed-badge").textContent = `${s.indexed_count} indexed`;
    $("footer-count").textContent = s.indexed_count;
  } catch {
    /* server starting */
  }
}

function showError(msg) {
  const el = $("error-banner");
  if (!msg) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.textContent = msg;
  el.classList.remove("hidden");
}

function renderMessages() {
  const empty = $("empty-state");
  const list = $("messages");
  if (state.messages.length === 0) {
    empty.classList.remove("hidden");
    list.classList.add("hidden");
    list.innerHTML = "";
    return;
  }
  empty.classList.add("hidden");
  list.classList.remove("hidden");
  list.innerHTML = state.messages
    .map((m) => {
      const cls = m.role === "user" ? "msg msg-user" : "msg msg-assistant";
      const body =
        m.role === "assistant" ? renderMarkdown(m.content) : escapeHtml(m.content);
      return `<div class="${cls}">${body}</div>`;
    })
    .join("");
  const scrollEl = $("chat-scroll");
  if (scrollEl) {
    scrollEl.scrollTop = scrollEl.scrollHeight;
  } else {
    list.scrollTop = list.scrollHeight;
  }
}

function showToast(msg) {
  const el = $("error-banner");
  el.textContent = msg;
  el.classList.remove("hidden");
  el.classList.add("toast-info");
  setTimeout(() => {
    el.classList.remove("toast-info");
    if (el.textContent === msg) {
      el.classList.add("hidden");
      el.textContent = "";
    }
  }, 2500);
}

function resultCardActions(card) {
  const loc = card.source_location
    ? `<span class="source-loc">${escapeHtml(card.source_location)}</span>`
    : "";
  const openSrc = card.source_url
    ? `<a class="btn-link-sm btn-download" href="${escapeHtml(card.source_url)}" target="_blank" rel="noopener" data-image-id="${escapeHtml(card.image_id)}" data-rank="${card.rank}">Open source</a>`
    : "";
  const copyPath =
    card.source_path
      ? `<button type="button" class="btn-link-sm btn-copy-path" data-path="${escapeHtml(card.source_path)}">Copy path</button>`
      : "";
  const similar = `<button type="button" class="btn-link-sm btn-similar" data-image-id="${escapeHtml(card.image_id)}" data-rank="${card.rank}">Find similar</button>`;
  return `<div class="result-actions">${loc}${openSrc}${copyPath}${similar}</div>`;
}

function renderResults(results) {
  const grid = $("results-grid");
  const countEl = $("results-count");
  if (!results || results.length === 0) {
    countEl.textContent = "";
    grid.innerHTML =
      '<div class="results-empty"><p>Results from your latest query appear here</p></div>';
    return;
  }
  countEl.textContent = `(${results.length} image${results.length !== 1 ? "s" : ""})`;
  grid.innerHTML = results
    .map((card) => {
      const chips = (card.provenance.chips || [])
        .map((c) => `<span class="chip-tag">${escapeHtml(c)}</span>`)
        .join("");
      const thumb = card.has_image_file
        ? `<img class="result-thumb-img" src="${escapeHtml(card.image_url)}" alt="" loading="lazy" data-image-id="${escapeHtml(card.image_id)}" data-rank="${card.rank}" />`
        : '<div class="no-img">Image unavailable</div>';
      const hint = card.match_hint
        ? `<p class="result-hint" title="${escapeHtml(card.match_hint)}">${escapeHtml(card.match_hint)}</p>`
        : "";
      return `
        <article class="result-card">
          <div class="result-thumb">
            <span class="result-rank">#${card.rank}</span>
            <span class="result-match">${card.match_percent}%</span>
            ${thumb}
          </div>
          <div class="result-body">
            <div class="chips">${chips}</div>
            ${card.caption ? `<p class="result-caption">${escapeHtml(card.caption)}</p>` : ""}
            ${hint}
            ${resultCardActions(card)}
          </div>
        </article>`;
    })
    .join("");

  grid.querySelectorAll(".result-thumb-img").forEach((img) => {
    img.addEventListener("click", () => {
      recordInteraction(img.dataset.imageId, "view", Number(img.dataset.rank));
    });
  });
  grid.querySelectorAll(".btn-download").forEach((btn) => {
    btn.addEventListener("click", () => {
      recordInteraction(btn.dataset.imageId, "download", Number(btn.dataset.rank));
    });
  });
  grid.querySelectorAll(".btn-similar").forEach((btn) => {
    btn.addEventListener("click", () => {
      recordInteraction(btn.dataset.imageId, "similar", Number(btn.dataset.rank));
      runSimilar({ imageId: btn.dataset.imageId });
    });
  });
  grid.querySelectorAll(".btn-copy-path").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const path = btn.dataset.path;
      try {
        await navigator.clipboard.writeText(path);
        showToast("Path copied to clipboard.");
      } catch {
        showToast(path);
      }
    });
  });
}

function renderParsedQuery(pq) {
  const details = $("search-details");
  const list = $("parsed-query-list");
  if (!pq) {
    details.classList.add("hidden");
    return;
  }
  const hasNotes = pq.interpretation_notes?.length;
  const hasParsed =
    pq.semantic_query ||
    pq.must_have_keywords?.length ||
    pq.must_avoid_keywords?.length ||
    pq.is_refinement;
  if (!hasNotes && !hasParsed) {
    details.classList.add("hidden");
    return;
  }
  details.classList.remove("hidden");
  const items = [];
  if (pq.interpretation_notes?.length) {
    items.push('<li class="interp-notes"><strong>Notes</strong><ul>');
    for (const note of pq.interpretation_notes) {
      items.push(`<li>${escapeHtml(note)}</li>`);
    }
    items.push("</ul></li>");
  }
  items.push(`<li>Semantic: <code>${escapeHtml(pq.semantic_query || "")}</code></li>`);
  if (pq.must_have_keywords?.length) {
    items.push(`<li>Must have: ${escapeHtml(pq.must_have_keywords.join(", "))}</li>`);
  }
  if (pq.must_avoid_keywords?.length) {
    items.push(`<li>Must avoid: ${escapeHtml(pq.must_avoid_keywords.join(", "))}</li>`);
  }
  const sf = pq.source_filters || {};
  const filterParts = [];
  if (sf.file_types?.length) filterParts.push(`types=${sf.file_types.join(",")}`);
  if (sf.filename_contains?.length) {
    filterParts.push(`filename~${sf.filename_contains.join("|")}`);
  }
  if (sf.authors?.length) filterParts.push(`authors=${sf.authors.join(",")}`);
  if (filterParts.length) {
    items.push(`<li>Source filters: ${escapeHtml(filterParts.join("; "))}</li>`);
  }
  const tf = pq.time_filter || {};
  const timeParts = [];
  if (tf.after) timeParts.push(`after ${tf.after}`);
  if (tf.before) timeParts.push(`before ${tf.before}`);
  if (timeParts.length) {
    items.push(`<li>Time: ${escapeHtml(timeParts.join(", "))}</li>`);
  }
  if (pq.is_refinement) {
    items.push("<li>Mode: refining previous results</li>");
  }
  list.innerHTML = items.join("");
}

async function applySearchResponse(res, userLabel) {
  if (res.session_id) state.sessionId = res.session_id;
  if (res.search_event_id) state.searchEventId = res.search_event_id;
  state.messages.push({ role: "assistant", content: res.assistant_message });
  renderMessages();
  renderResults(res.results);
  renderParsedQuery(res.parsed_query);
}

async function runSimilar({ imageId, file }) {
  if (state.loading) return;
  state.loading = true;
  $("btn-send").disabled = true;
  showError(null);
  const topK = Number($("top-k").value);
  const minMatch = Number($("min-match").value);

  const label = file ? "[similar to attached image]" : "[find similar]";
  state.messages.push({ role: "user", content: label });
  renderMessages();

  try {
    let res;
    if (file) {
      const form = new FormData();
      form.append("file", file);
      form.append("top_k", String(topK));
      form.append("min_match_percent", String(minMatch));
      form.append("similarity_axis", "balanced");
      if (state.sessionId) form.append("session_id", state.sessionId);
      res = await api("/api/similar", { method: "POST", body: form });
    } else {
      res = await api("/api/similar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_id: imageId,
          session_id: state.sessionId,
          top_k: topK,
          min_match_percent: minMatch,
          similarity_axis: "balanced",
        }),
      });
    }
    await applySearchResponse(res, label);
  } catch (e) {
    const msg = e.message || String(e);
    showError(msg);
    state.messages.push({ role: "assistant", content: `**Error:** ${msg}` });
    renderMessages();
  } finally {
    state.loading = false;
    $("btn-send").disabled = false;
    const qi = $("query-image");
    if (qi) qi.value = "";
    const hint = $("attach-hint");
    if (hint) hint.classList.add("hidden");
  }
}

async function sendMessage() {
  const input = $("query-input");
  const text = input.value.trim();
  if (!text || state.loading) return;

  state.loading = true;
  $("btn-send").disabled = true;
  showError(null);
  input.value = "";

  state.messages.push({ role: "user", content: text });
  renderMessages();

  const topK = Number($("top-k").value);
  const minMatch = Number($("min-match").value);

  try {
    const res = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: state.sessionId,
        top_k: topK,
        min_match_percent: minMatch,
      }),
    });
    await applySearchResponse(res, text);
  } catch (e) {
    const msg = e.message || String(e);
    showError(msg);
    state.messages.push({ role: "assistant", content: `**Error:** ${msg}` });
    renderMessages();
  } finally {
    state.loading = false;
    $("btn-send").disabled = false;
  }
}

async function clearSession() {
  if (state.sessionId) {
    try {
      await api("/api/session/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId }),
      });
    } catch {
      /* ignore */
    }
  }
  state.sessionId = null;
  state.messages = [];
  showError(null);
  renderMessages();
  renderResults([]);
  renderParsedQuery(null);
}

function openDrawer() {
  $("drawer-backdrop").classList.remove("hidden");
  $("corpus-drawer").classList.remove("hidden");
}

function closeDrawer() {
  $("drawer-backdrop").classList.add("hidden");
  $("corpus-drawer").classList.add("hidden");
}

async function runIngest() {
  const fileInput = $("ingest-files");
  if (!fileInput.files?.length) {
    $("ingest-status").textContent = "Select at least one file.";
    $("ingest-status").classList.remove("hidden");
    return;
  }

  const form = new FormData();
  for (const f of fileInput.files) {
    form.append("files", f);
  }
  form.append("skip_caption", $("skip-caption").checked);
  form.append("skip_ocr", $("skip-ocr").checked);
  form.append("force", $("force-ingest").checked);

  $("btn-ingest").disabled = true;
  const status = $("ingest-status");
  status.textContent = "Ingesting…";
  status.classList.remove("hidden");

  try {
    const adminKey = sessionStorage.getItem("imagecb.adminApiKey");
    const headers = adminKey ? { "X-Admin-Api-Key": adminKey } : {};
    const res = await api("/api/ingest", { method: "POST", headers, body: form });
    status.textContent = res.message;
    $("indexed-badge").textContent = `${res.indexed_count} indexed`;
    $("footer-count").textContent = res.indexed_count;
    fileInput.value = "";
  } catch (e) {
    status.textContent = e.message || String(e);
  } finally {
    $("btn-ingest").disabled = false;
  }
}

function initExamples() {
  const container = $("example-queries");
  EXAMPLES.forEach((ex) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.textContent = ex;
    btn.addEventListener("click", () => {
      $("query-input").value = ex;
      $("query-input").focus();
    });
    container.appendChild(btn);
  });
}

function init() {
  initExamples();
  refreshStatus();

  $("top-k").addEventListener("input", () => {
    $("top-k-val").textContent = $("top-k").value;
  });

  $("min-match").addEventListener("input", () => {
    $("min-match-val").textContent = $("min-match").value;
  });

  $("btn-send").addEventListener("click", sendMessage);
  $("query-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  $("btn-clear").addEventListener("click", clearSession);
  $("btn-corpus").addEventListener("click", openDrawer);
  $("btn-close-drawer").addEventListener("click", closeDrawer);
  $("drawer-backdrop").addEventListener("click", closeDrawer);
  $("btn-ingest").addEventListener("click", runIngest);

  const queryImage = $("query-image");
  if (queryImage) {
    queryImage.addEventListener("change", () => {
      const file = queryImage.files?.[0];
      const hint = $("attach-hint");
      if (!file) {
        hint.classList.add("hidden");
        return;
      }
      hint.textContent = `Attached: ${file.name} — searching…`;
      hint.classList.remove("hidden");
      runSimilar({ file });
    });
  }
}

init();
