import { firebaseBridge } from "/static/firebase-bridge.js";

const state = {
  polymers: {},
  routes: [],
  weights: [0.30, 0.15, 0.20, 0.10, 0.25],
  poly: "PP",
  t: 1500,
  I: 1.0,
  T: 50,
  RH: 60,
  spectralMode: "auto",
  CI: 2.46,
  HI: 1.0,
  dXc: 5.4,
  assessments: [],
  metrics: null,
  streamActive: false,
  lastPrediction: null,
  activeImportProfile: null,
  firebaseConnected: false,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

function toast(msg, isErr = false) {
  const box = $("#toastContainer");
  const el = document.createElement("div");
  el.className = `toast ${isErr ? "err" : ""}`;
  el.textContent = msg;
  box.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

let predictTimer = null;
function schedulePredict(delay = 30) {
  clearTimeout(predictTimer);
  predictTimer = setTimeout(runPredict, delay);
}

async function apiJson(url, opts = {}) {
  const res = await fetch(url, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// ---------- Initialization ----------
async function init() {
  bindTabs();
  bindThemeToggle();
  bindControls();
  bindFeedFilters();
  bindKaggleHub();
  bindFirebaseModal();

  const health = await apiJson("/api/health");
  state.polymers = health.polymers;
  state.routes = health.routes;
  state.weights = [...health.default_weights];
  state.streamActive = Boolean(health.stream?.active);

  renderPolymerGrid();
  updateDatasetPill(health.dataset_meta);
  updateStreamButton();

  await Promise.all([loadMetrics(), loadAssessments(), loadKaggleCatalog()]);
  await runPredict();
  connectSSE();

  firebaseBridge.on({
    onStatus: (connected, info) => {
      state.firebaseConnected = connected;
      const btn = $("#firebaseStatusBtn");
      const txt = $("#firebaseStatusText");
      if (connected) {
        btn.className = "status-pill pill-firebase";
        txt.textContent = `Firebase Live (${info})`;
      } else {
        btn.className = "status-pill pill-sse";
        txt.textContent = "Sync: Local Real-Time (SSE)";
      }
    },
    onAssessments: (items) => {
      if (items.length > 0) {
        state.assessments = mergeAssessments(items, state.assessments);
        renderFeed();
      }
    },
    onWeights: (weights) => {
      if (Array.isArray(weights) && weights.length === 5) {
        state.weights = weights.map(Number);
        syncWeightSlidersToState();
        schedulePredict(10);
      }
    },
  });
  await firebaseBridge.initFromServerOrStorage();
}

function mergeAssessments(primary, secondary) {
  const map = new Map();
  for (const r of [...primary, ...secondary]) {
    if (r && r.id && !map.has(r.id)) map.set(r.id, r);
  }
  return [...map.values()].sort((a, b) => (b.created_at || 0) - (a.created_at || 0)).slice(0, 120);
}

// ---------- Tabs & Theme ----------
function bindTabs() {
  $$(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      $$(".nav-tab").forEach((b) => b.classList.toggle("active", b === btn));
      $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${target}`));
    });
  });
}

function bindThemeToggle() {
  $("#themeToggleBtn").addEventListener("click", () => {
    const html = document.documentElement;
    html.dataset.theme = html.dataset.theme === "dark" ? "light" : "dark";
    if (state.lastPrediction) drawKineticsChart(state.lastPrediction);
  });
}

// ---------- Lab Controls ----------
function renderPolymerGrid() {
  const grid = $("#polymerSelector");
  grid.innerHTML = "";
  Object.entries(state.polymers).forEach(([abbr, info]) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `poly-chip ${state.poly === abbr ? "active" : ""}`;
    btn.innerHTML = `<span class="p-abbr">${abbr}</span><span class="p-code">Resin #${info.code}</span>`;
    btn.addEventListener("click", () => {
      state.poly = abbr;
      renderPolymerGrid();
      schedulePredict(0);
    });
    grid.appendChild(btn);
  });
}

function bindControls() {
  const sliders = [
    ["t", "input-t", "val-t", (v) => `${Math.round(v)}`],
    ["I", "input-I", "val-I", (v) => Number(v).toFixed(2)],
    ["T", "input-T", "val-T", (v) => `${Math.round(v)}`],
    ["RH", "input-RH", "val-RH", (v) => `${Math.round(v)}`],
    ["CI", "input-CI", "val-CI", (v) => Number(v).toFixed(2)],
    ["HI", "input-HI", "val-HI", (v) => Number(v).toFixed(2)],
    ["dXc", "input-dXc", "val-dXc", (v) => Number(v).toFixed(1)],
  ];
  sliders.forEach(([key, inputId, valId, fmt]) => {
    const inp = $(`#${inputId}`);
    inp.addEventListener("input", () => {
      state[key] = Number(inp.value);
      $(`#${valId}`).textContent = fmt(state[key]);
      schedulePredict(25);
    });
  });

  $$("#spectralModeSwitch .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.spectralMode = btn.dataset.mode;
      $$("#spectralModeSwitch .seg-btn").forEach((b) => b.classList.toggle("active", b === btn));
      $("#manualSpectralPanel").classList.toggle("hidden", state.spectralMode !== "manual");
      $("#spectralSourceTag").textContent = state.spectralMode === "manual" ? "Manual FTIR/DSC" : "Photo-Oxidation Model";
      schedulePredict(0);
    });
  });

  $$(".weight-slider").forEach((sl) => {
    sl.addEventListener("input", () => {
      const idx = Number(sl.dataset.idx);
      state.weights[idx] = Number(sl.value);
      updateWeightLabels();
      schedulePredict(25);
    });
    sl.addEventListener("change", () => {
      if (state.firebaseConnected) {
        firebaseBridge.pushWeights(state.weights).catch(() => {});
      }
    });
  });

  $("#resetWeightsBtn").addEventListener("click", () => {
    state.weights = [0.30, 0.15, 0.20, 0.10, 0.25];
    syncWeightSlidersToState();
    schedulePredict(0);
    if (state.firebaseConnected) firebaseBridge.pushWeights(state.weights).catch(() => {});
  });

  $("#resetDemoBtn").addEventListener("click", () => {
    state.poly = "PP";
    state.t = 1500;
    state.I = 1.0;
    state.T = 50;
    state.RH = 60;
    state.spectralMode = "auto";
    $("#input-t").value = 1500; $("#val-t").textContent = "1500";
    $("#input-I").value = 1.0; $("#val-I").textContent = "1.00";
    $("#input-T").value = 50; $("#val-T").textContent = "50";
    $("#input-RH").value = 60; $("#val-RH").textContent = "60";
    $$("#spectralModeSwitch .seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === "auto"));
    $("#manualSpectralPanel").classList.add("hidden");
    renderPolymerGrid();
    schedulePredict(0);
  });

  $("#saveAssessmentBtn").addEventListener("click", async () => {
    try {
      const payload = {
        sample_name: $("#sampleNameInput").value.trim() || `${state.poly}-Lab-${new Date().toTimeString().slice(0, 5)}`,
        batch_id: $("#batchIdInput").value.trim() || `LAB-${new Date().toISOString().slice(11, 16).replace(":", "")}`,
        poly: state.poly,
        t: state.t,
        I: state.I,
        T: state.T,
        RH: state.RH,
        spectral_mode: state.spectralMode,
        CI: state.CI,
        HI: state.HI,
        dXc: state.dXc,
        weights: state.weights,
        source: state.firebaseConnected ? "Lab Console (Firebase)" : "Lab Console",
      };
      const res = await apiJson("/api/assessments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (state.firebaseConnected) {
        await firebaseBridge.pushAssessment(res.item);
        toast(`Saved & synced ${res.item.sample_name} to Firebase Firestore!`);
      } else {
        toast(`Saved ${res.item.sample_name} to Real-Time Feed`);
      }
      $("#sampleNameInput").value = "";
    } catch (err) {
      toast(err.message, true);
    }
  });

  $("#streamToggleBtn").addEventListener("click", async () => {
    try {
      const data = await apiJson("/api/stream/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active: !state.streamActive, interval_sec: 2.5 }),
      });
      state.streamActive = data.stream.active;
      updateStreamButton();
      toast(state.streamActive ? "Live Conveyor Sensor Stream started" : "Sensor Stream paused");
    } catch (err) {
      toast(err.message, true);
    }
  });
}

function syncWeightSlidersToState() {
  $$(".weight-slider").forEach((sl) => {
    sl.value = state.weights[Number(sl.dataset.idx)];
  });
  updateWeightLabels();
}

function updateWeightLabels() {
  const sum = state.weights.reduce((a, b) => a + b, 0) || 1;
  state.weights.forEach((w, i) => {
    $(`#wVal-${i}`).textContent = `${Math.round((w / sum) * 100)}%`;
  });
}

function updateStreamButton() {
  const btn = $("#streamToggleBtn");
  const lbl = $("#streamToggleLabel");
  btn.classList.toggle("btn-streaming", state.streamActive);
  lbl.textContent = state.streamActive ? "Sensor Stream: LIVE" : "Start Live Sensor Stream";
}

// ---------- Inference & Live Visualization ----------
async function runPredict() {
  try {
    const data = await apiJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        poly: state.poly,
        t: state.t,
        I: state.I,
        T: state.T,
        RH: state.RH,
        spectral_mode: state.spectralMode,
        CI: state.CI,
        HI: state.HI,
        dXc: state.dXc,
        weights: state.weights,
      }),
    });
    state.lastPrediction = data;
    renderPrediction(data);
  } catch (err) {
    console.error("Prediction error:", err);
  }
}

function renderPrediction(d) {
  // Module 1
  const stg = d.module1.stage;
  const badge = $("#res-stage-badge");
  badge.textContent = stg;
  badge.className = `stage-badge stage-${stg.toLowerCase()}`;
  $("#res-poly-tag").textContent = `${d.input.poly} · Resin #${d.input.resin_code}`;

  const probsBox = $("#res-stage-probs");
  const colors = { Low: "#10b981", Moderate: "#f59e0b", Severe: "#f43f5e" };
  probsBox.innerHTML = ["Low", "Moderate", "Severe"]
    .map((s) => {
      const p = Math.round((d.module1.probabilities[s] || 0) * 100);
      return `<div class="prob-row">
        <span>${s}</span>
        <div class="prob-track"><div class="prob-fill" style="width:${p}%;background:${colors[s]}"></div></div>
        <span class="mono-val">${p}%</span>
      </div>`;
    })
    .join("");

  // Module 2
  const retPct = d.module2.ret_ml * 100;
  $("#res-ret-pct").textContent = `${retPct.toFixed(1)}%`;
  $("#res-ret-kin").textContent = `${(d.module2.ret_true_kinetics * 100).toFixed(1)}%`;
  $("#res-t50").textContent = d.module2.t50_ml ? `${d.module2.t50_ml} h` : ">3500 h";
  $("#res-ret-bar").style.width = `${Math.min(100, Math.max(2, retPct))}%`;

  // Spectral indicators
  if (state.spectralMode === "auto") {
    state.CI = d.spectral.CI;
    state.HI = d.spectral.HI;
    state.dXc = d.spectral.dXc;
    $("#input-CI").value = d.spectral.CI; $("#val-CI").textContent = d.spectral.CI.toFixed(2);
    $("#input-HI").value = d.spectral.HI; $("#val-HI").textContent = d.spectral.HI.toFixed(2);
    $("#input-dXc").value = d.spectral.dXc; $("#val-dXc").textContent = d.spectral.dXc.toFixed(1);
  }
  $("#spec-ci").textContent = d.spectral.CI.toFixed(2);
  $("#spec-hi").textContent = d.spectral.HI.toFixed(2);
  $("#spec-dxc").textContent = `${d.spectral.dXc.toFixed(1)}%`;
  $("#spec-ci-max").textContent = d.spectral.CI_max;
  $("#spec-hi-max").textContent = d.spectral.HI_max;
  $("#spec-dxc-max").textContent = d.spectral.dXc_max;
  $("#spec-ci-bar").style.width = `${Math.min(100, (d.spectral.CI / Math.max(0.5, d.spectral.CI_max)) * 100)}%`;
  $("#spec-hi-bar").style.width = `${Math.min(100, (d.spectral.HI / Math.max(0.5, d.spectral.HI_max)) * 100)}%`;
  $("#spec-dxc-bar").style.width = `${Math.min(100, Math.max(0, (d.spectral.dXc / Math.max(1, d.spectral.dXc_max)) * 100))}%`;

  // Module 3
  const top = d.module3.ranked[0];
  $("#res-best-route").textContent = top.route;
  $("#res-best-score").textContent = top.score.toFixed(3);
  $("#res-best-yield").textContent = `${Math.round(top.yield * 100)}%`;
  $("#res-best-co2").textContent = `${top.co2e_kg} kg`;

  $("#topsisRoutesGrid").innerHTML = d.module3.ranked
    .map(
      (r, idx) => `
      <div class="route-card ${idx === 0 ? "rank-1" : ""}">
        <div class="route-rank">Rank #${idx + 1} · C* = ${r.score.toFixed(3)}</div>
        <div class="route-name">${r.route}</div>
        <div class="route-score-bar"><div class="route-score-fill" style="width:${Math.round(r.score * 100)}%"></div></div>
        <div class="route-metrics">
          <span>Yield: <b>${Math.round(r.yield * 100)}%</b></span>
          <span>Quality: <b>${Math.round(r.quality * 100)}%</b></span>
          <span>Cost: <b>₹${r.cost_inr}/kg</b></span>
          <span>CO₂e: <b>${r.co2e_kg} kg</b></span>
          <span>Energy: <b>${r.energy_mj} MJ</b></span>
        </div>
      </div>`
    )
    .join("");

  drawKineticsChart(d);
}

function drawKineticsChart(d) {
  const svg = $("#kineticsChart");
  const W = 680, H = 280;
  const pad = { l: 46, r: 20, t: 18, b: 34 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const { t, ml, true: tru, kin } = d.module2.curve;
  const maxT = t[t.length - 1] || 2500;
  const xMap = (tv) => pad.l + (tv / maxT) * pw;
  const yMap = (yv) => pad.t + (1 - yv) * ph;

  const polyLine = (arr) => arr.map((v, i) => `${i === 0 ? "M" : "L"}${xMap(t[i]).toFixed(1)},${yMap(v).toFixed(1)}`).join(" ");
  const isLight = document.documentElement.dataset.theme === "light";
  const gridCol = isLight ? "rgba(15,23,42,0.08)" : "rgba(148,163,184,0.12)";
  const textCol = isLight ? "#475569" : "#94a3b8";

  let gridSvg = "";
  [0, 0.25, 0.5, 0.75, 1.0].forEach((yv) => {
    const y = yMap(yv);
    gridSvg += `<line x1="${pad.l}" y1="${y}" x2="${W - pad.r}" y2="${y}" stroke="${gridCol}" />`;
    gridSvg += `<text x="${pad.l - 8}" y="${y + 4}" text-anchor="end" font-size="10" fill="${textCol}">${Math.round(yv * 100)}%</text>`;
  });
  [0, 500, 1000, 1500, 2000, 2500].forEach((tv) => {
    const x = xMap(tv);
    gridSvg += `<line x1="${x}" y1="${pad.t}" x2="${x}" y2="${H - pad.b}" stroke="${gridCol}" />`;
    gridSvg += `<text x="${x}" y="${H - 12}" text-anchor="middle" font-size="10" fill="${textCol}">${tv}h</text>`;
  });

  // 50% threshold
  const y50 = yMap(0.5);
  const currX = xMap(Math.min(maxT, d.input.t));
  const currY = yMap(d.module2.ret_ml);

  svg.innerHTML = `
    ${gridSvg}
    <line x1="${pad.l}" y1="${y50}" x2="${W - pad.r}" y2="${y50}" stroke="#f43f5e" stroke-dasharray="4 4" opacity="0.65"/>
    <text x="${W - pad.r - 4}" y="${y50 - 5}" text-anchor="end" font-size="9.5" fill="#fb7185">50% Severe Threshold</text>
    <path d="${polyLine(kin)}" fill="none" stroke="#64748b" stroke-width="2" stroke-dasharray="3 3"/>
    <path d="${polyLine(tru)}" fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="6 4"/>
    <path d="${polyLine(ml)}" fill="none" stroke="#10b981" stroke-width="3"/>
    <line x1="${currX}" y1="${pad.t}" x2="${currX}" y2="${H - pad.b}" stroke="#0ea5e9" stroke-width="1.5" stroke-dasharray="3 3"/>
    <circle cx="${currX}" cy="${currY}" r="6" fill="#10b981" stroke="#ffffff" stroke-width="2"/>
  `;

  svg.onmousemove = (ev) => {
    const rect = svg.getBoundingClientRect();
    const rx = ((ev.clientX - rect.left) / rect.width) * W;
    const ratio = Math.min(1, Math.max(0, (rx - pad.l) / pw));
    const idx = Math.min(t.length - 1, Math.round(ratio * (t.length - 1)));
    const tip = $("#chartTooltip");
    tip.classList.remove("hidden");
    tip.style.left = `${Math.min(rect.width - 155, Math.max(10, ev.clientX - rect.left + 12))}px`;
    tip.style.top = `${Math.max(10, ev.clientY - rect.top - 55)}px`;
    tip.innerHTML = `
      <div><b>t = ${t[idx]} h</b></div>
      <div style="color:#34d399">ML GB: ${(ml[idx] * 100).toFixed(1)}%</div>
      <div style="color:#fbbf24">True Kinetics: ${(tru[idx] * 100).toFixed(1)}%</div>
      <div style="color:#94a3b8">Dose-Only: ${(kin[idx] * 100).toFixed(1)}%</div>
    `;
  };
  svg.onmouseleave = () => $("#chartTooltip").classList.add("hidden");
}

// ---------- Real-Time Plant Feed & SSE ----------
async function loadAssessments() {
  const data = await apiJson("/api/assessments");
  state.assessments = mergeAssessments(data.items || [], state.assessments);
  renderFeed();
}

function connectSSE() {
  const es = new EventSource("/api/stream");
  es.addEventListener("assessment_created", (e) => {
    const rec = JSON.parse(e.data);
    state.assessments = mergeAssessments([rec], state.assessments);
    renderFeed();
  });
  es.addEventListener("sensor_tick", (e) => {
    const rec = JSON.parse(e.data);
    state.assessments = mergeAssessments([rec], state.assessments);
    renderFeed();
    if (state.firebaseConnected) {
      firebaseBridge.pushAssessment(rec).catch(() => {});
    }
  });
  es.addEventListener("assessment_deleted", (e) => {
    const { id } = JSON.parse(e.data);
    state.assessments = state.assessments.filter((x) => x.id !== id);
    renderFeed();
  });
  es.addEventListener("model_retrained", (e) => {
    state.metrics = JSON.parse(e.data);
    updateDatasetPill(state.metrics.dataset_meta);
    renderTelemetry();
    schedulePredict(0);
    toast("Models retrained & live across all clients!");
  });
  es.addEventListener("stream_state", (e) => {
    const st = JSON.parse(e.data);
    state.streamActive = Boolean(st.active);
    updateStreamButton();
  });
}

function bindFeedFilters() {
  $("#filterPoly").addEventListener("change", renderFeed);
  $("#filterStage").addEventListener("change", renderFeed);
  $("#pulseSensorBtn").addEventListener("click", async () => {
    const data = await apiJson("/api/stream/tick", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: "Manual Sensor Pulse" }),
    });
    if (state.firebaseConnected && data.item) {
      firebaseBridge.pushAssessment(data.item).catch(() => {});
    }
  });
  $("#exportCsvBtn").addEventListener("click", () => {
    const headers = ["id", "sample_name", "batch_id", "poly", "t", "I", "T", "RH", "CI", "HI", "dXc", "ret", "stage", "route", "source"];
    const rows = [headers.join(",")];
    state.assessments.forEach((r) => {
      rows.push(headers.map((h) => JSON.stringify(r[h] ?? "")).join(","));
    });
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "polylife_assessments.csv";
    a.click();
  });
}

function renderFeed() {
  const fPoly = $("#filterPoly").value;
  const fStage = $("#filterStage").value;
  const rows = state.assessments.filter(
    (r) => (fPoly === "ALL" || r.poly === fPoly) && (fStage === "ALL" || r.stage === fStage)
  );

  $("#feedCountBadge").textContent = state.assessments.length;
  $("#kpiTotal").textContent = state.assessments.length;
  const mech = state.assessments.filter((r) => r.route === "Mechanical recycling").length;
  const sev = state.assessments.filter((r) => r.stage === "Severe").length;
  const meanRet = state.assessments.length
    ? (state.assessments.reduce((a, b) => a + Number(b.ret || 0), 0) / state.assessments.length) * 100
    : 0;
  $("#kpiMech").textContent = `${state.assessments.length ? Math.round((mech / state.assessments.length) * 100) : 0}%`;
  $("#kpiSevere").textContent = sev;
  $("#kpiMeanRet").textContent = `${meanRet.toFixed(1)}%`;

  const tbody = $("#feedTableBody");
  tbody.innerHTML = rows
    .map((r) => {
      const ts = new Date((r.created_at || Date.now() / 1000) * 1000).toLocaleTimeString();
      const stCls = `stage-${String(r.stage || "low").toLowerCase()}`;
      return `<tr>
        <td class="mono-val">${ts}</td>
        <td><b>${escapeHtml(r.sample_name)}</b><br/><small class="sub-hint">${escapeHtml(r.batch_id)}</small></td>
        <td><b>${escapeHtml(r.poly)}</b></td>
        <td class="mono-val">${Number(r.CI).toFixed(2)} / ${Number(r.HI).toFixed(2)}</td>
        <td class="mono-val">${(Number(r.ret) * 100).toFixed(1)}%</td>
        <td><span class="stage-badge ${stCls}">${escapeHtml(r.stage)}</span></td>
        <td><b>${escapeHtml(r.route)}</b></td>
        <td><small>${escapeHtml(r.source || "Lab")}</small></td>
        <td><button class="btn btn-ghost btn-xs del-row-btn" data-id="${escapeHtml(r.id)}">✕</button></td>
      </tr>`;
    })
    .join("");

  $$(".del-row-btn").forEach((b) => {
    b.addEventListener("click", async () => {
      const id = b.dataset.id;
      await apiJson(`/api/assessments/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (state.firebaseConnected) {
        firebaseBridge.deleteAssessment(id).catch(() => {});
      }
    });
  });
}

// ---------- Kaggle Dataset Hub ----------
async function loadKaggleCatalog() {
  const data = await apiJson("/api/kaggle/catalog");
  const grid = $("#kaggleCatalogGrid");
  grid.innerHTML = (data.catalog || [])
    .map(
      (c) => `
      <div class="cat-card">
        <div>
          <span class="cat-tag">${escapeHtml(c.category)}</span>
          <h4>${escapeHtml(c.title)}</h4>
          <p>${escapeHtml(c.description)}</p>
        </div>
        <div>
          <p><strong>PolyLife Use:</strong> ${escapeHtml(c.polylife_mapping)}</p>
          <div class="cat-actions">
            <a href="${escapeHtml(c.kaggle_url)}" target="_blank" rel="noopener">Open on Kaggle ↗</a>
          </div>
        </div>
      </div>`
    )
    .join("");
  renderDatasetLogs(data.logs || []);
}

function renderDatasetLogs(logs) {
  const box = $("#datasetLogsList");
  if (!logs.length) {
    box.innerHTML = `<div class="sub-hint">No external Kaggle datasets imported yet.</div>`;
    return;
  }
  box.innerHTML = logs
    .map(
      (l) => `<div class="log-row">
        <div><b>${escapeHtml(l.title)}</b> <small class="sub-hint">${escapeHtml(l.source)}</small></div>
        <span class="mono-val">${l.rows} rows</span>
      </div>`
    )
    .join("");
}

function bindKaggleHub() {
  $("#fetchKaggleBtn").addEventListener("click", async () => {
    const slug = $("#kaggleSlugInput").value.trim();
    const username = $("#kaggleUserInput").value.trim();
    const key = $("#kaggleKeyInput").value.trim();
    if (!slug) return toast("Enter a Kaggle dataset slug (owner/dataset-name).", true);
    try {
      toast("Downloading Kaggle dataset...");
      const profile = await apiJson("/api/kaggle/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slug, username, key }),
      });
      showMapper(profile);
      loadKaggleCatalog();
      toast(`Imported ${profile.title} (${profile.rows} rows)`);
    } catch (err) {
      toast(err.message, true);
    }
  });

  const dz = $("#csvDropZone");
  const inp = $("#csvFileInput");
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    if (e.dataTransfer.files[0]) handleCsvFile(e.dataTransfer.files[0]);
  });
  inp.addEventListener("change", () => {
    if (inp.files[0]) handleCsvFile(inp.files[0]);
  });

  $("#restoreBuiltinBtn").addEventListener("click", async () => {
    try {
      const data = await apiJson("/api/kaggle/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: "__builtin__" }),
      });
      state.metrics = data.metrics;
      updateDatasetPill(state.metrics.dataset_meta);
      renderTelemetry();
      schedulePredict(0);
      toast("Restored Built-in Chemistry-Informed Dataset (1,800 samples)");
    } catch (err) {
      toast(err.message, true);
    }
  });
}

async function handleCsvFile(file) {
  try {
    const text = await file.text();
    const profile = await apiJson("/api/kaggle/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file.name, csv_text: text }),
    });
    showMapper(profile);
    loadKaggleCatalog();
    toast(`Profiled ${file.name} (${profile.rows} rows)`);
  } catch (err) {
    toast(err.message, true);
  }
}

function showMapper(profile) {
  state.activeImportProfile = profile;
  $("#mapperSubtitle").textContent = `Loaded ${profile.title} (${profile.rows} rows, ${profile.columns.length} columns)`;
  const fields = [
    ["poly", "Polymer Type (HDPE/PP/PET...)"],
    ["t", "Exposure Time (h)"],
    ["I", "UV Irradiance (I)"],
    ["T", "Temperature (°C)"],
    ["RH", "Humidity (%)"],
    ["CI", "Carbonyl Index (CI)"],
    ["HI", "Hydroxyl Index (HI)"],
    ["dXc", "Crystallinity Gain (ΔXc)"],
    ["ret", "Mw / Property Retention (0-1 or %)"],
  ];
  const opts = [`<option value="">-- Auto-derive via Kinetics --</option>`]
    .concat(profile.columns.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`))
    .join("");

  const box = $("#mapperContainer");
  box.className = "";
  box.innerHTML = `
    <div class="mapper-grid">
      ${fields
        .map(
          ([k, lbl]) => `
        <div class="mapper-field">
          <label>${lbl}</label>
          <select class="map-sel" data-key="${k}">${opts}</select>
        </div>`
        )
        .join("")}
    </div>
    <button id="retrainMappedBtn" class="btn btn-primary btn-full">Retrain PolyLife ML Models on Mapped Dataset</button>
    <div class="table-wrap" style="margin-top:12px;max-height:200px;">
      <table class="data-table">
        <thead><tr>${profile.columns.slice(0, 6).map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>
        <tbody>${(profile.preview || []).slice(0, 5).map((r) => `<tr>${profile.columns.slice(0, 6).map((c) => `<td>${escapeHtml(String(r[c] ?? ""))}</td>`).join("")}</tr>`).join("")}</tbody>
      </table>
    </div>
  `;

  $$(".map-sel").forEach((sel) => {
    const sug = profile.suggested_mapping?.[sel.dataset.key];
    if (sug) sel.value = sug;
  });

  $("#retrainMappedBtn").addEventListener("click", async () => {
    const mapping = {};
    $$(".map-sel").forEach((sel) => {
      if (sel.value) mapping[sel.dataset.key] = sel.value;
    });
    try {
      const data = await apiJson("/api/kaggle/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: profile.dataset_id,
          mapping,
          title: profile.title,
        }),
      });
      state.metrics = data.metrics;
      updateDatasetPill(state.metrics.dataset_meta);
      renderTelemetry();
      schedulePredict(0);
      toast(`Models retrained! RF Acc: ${(data.metrics.m1.acc * 100).toFixed(1)}% · GB R²: ${data.metrics.m2.r2_ml.toFixed(3)}`);
    } catch (err) {
      toast(err.message, true);
    }
  });
}

function updateDatasetPill(meta) {
  if (!meta) return;
  $("#datasetPillText").textContent = `${meta.source} (${meta.rows} rows)`;
}

// ---------- Telemetry & TOPSIS Matrix ----------
async function loadMetrics() {
  state.metrics = await apiJson("/api/metrics");
  renderTelemetry();
}

function renderTelemetry() {
  const m = state.metrics;
  if (!m) return;

  $("#m1MetricsBody").innerHTML = `
    <div class="stat-tile"><span>RF Test Accuracy</span><b>${(m.m1.acc * 100).toFixed(1)}%</b></div>
    <div class="stat-tile"><span>5-Fold CV Accuracy</span><b>${(m.m1.cv * 100).toFixed(1)}%</b></div>
    <div class="stat-tile"><span>CI-Only LR Baseline</span><b>${(m.m1.base_acc * 100).toFixed(1)}%</b></div>
  `;
  $("#m2MetricsBody").innerHTML = `
    <div class="stat-tile"><span>GB Regressor R²</span><b>${m.m2.r2_ml.toFixed(3)}</b></div>
    <div class="stat-tile"><span>GB RMSE</span><b>${m.m2.rmse_ml.toFixed(4)}</b></div>
    <div class="stat-tile"><span>Dose-Only Kinetics R²</span><b>${m.m2.r2_kin.toFixed(3)}</b></div>
  `;

  // Confusion Matrix
  const stg = ["Low", "Moderate", "Severe"];
  const cm = m.m1.cm;
  $("#confusionMatrixBox").innerHTML = `
    <div class="cm-grid">
      <div></div>${stg.map((s) => `<div style="text-align:center;color:var(--text-muted)">Pred ${s}</div>`).join("")}
      ${cm
        .map(
          (row, rIdx) => `
          <div style="color:var(--text-muted)">True ${stg[rIdx]}</div>
          ${row.map((val, cIdx) => `<div class="cm-cell ${rIdx === cIdx ? "cm-diag" : ""}">${val}</div>`).join("")}
        `
        )
        .join("")}
    </div>
  `;

  // Feature Importance
  const fiEntries = Object.entries(m.m1.imp).sort((a, b) => b[1] - a[1]);
  $("#featureImportanceBox").innerHTML = `<div class="fi-list">${fiEntries
    .map(
      ([k, v]) => `
      <div class="fi-row">
        <b>${escapeHtml(k)}</b>
        <div class="prob-track"><div class="prob-fill" style="width:${Math.round(v * 100)}%;background:#10b981"></div></div>
        <span class="mono-val">${(v * 100).toFixed(1)}%</span>
      </div>`
    )
    .join("")}</div>`;

  // 18-case TOPSIS Matrix
  $("#m3StabilityNote").textContent = `Rank-1 sensitivity stability across ±20% weight perturbations: ${(m.m3_stability * 100).toFixed(1)}% (${m.m3_cases} scenarios)`;
  const polys = Object.keys(state.polymers);
  $("#topsisMatrixBox").innerHTML = `
    <table class="data-table">
      <thead><tr><th>Polymer</th><th>Low (90% Mw)</th><th>Moderate (65% Mw)</th><th>Severe (30% Mw)</th></tr></thead>
      <tbody>
        ${polys
          .map(
            (p) => `<tr>
              <td><b>${p}</b></td>
              ${stg
                .map((s) => {
                  const dec = m.m3_decisions[`${p}-${s}`] || "-";
                  const topScore = Math.max(...(m.m3_scores[`${p}-${s}`] || [0]));
                  return `<td><b>${escapeHtml(dec)}</b> <small class="mono-val">(${topScore.toFixed(3)})</small></td>`;
                })
                .join("")}
            </tr>`
          )
          .join("")}
      </tbody>
    </table>
  `;
}

// ---------- Firebase Modal ----------
function bindFirebaseModal() {
  const modal = $("#firebaseModal");
  const open = async () => {
    modal.classList.remove("hidden");
    try {
      const res = await apiJson("/api/firebase/config");
      if (res.config) $("#firebaseConfigTextarea").value = JSON.stringify(res.config, null, 2);
    } catch (_) {}
  };
  $("#firebaseStatusBtn").addEventListener("click", open);
  $("#closeFirebaseModal").addEventListener("click", () => modal.classList.add("hidden"));

  $("#saveFirebaseBtn").addEventListener("click", async () => {
    const msg = $("#firebaseModalMsg");
    msg.classList.add("hidden");
    try {
      const cfg = JSON.parse($("#firebaseConfigTextarea").value);
      await firebaseBridge.connect(cfg, { persistServer: true });
      msg.textContent = `Connected live to Firestore project "${cfg.projectId}"!`;
      msg.className = "modal-msg ok";
      toast(`Firebase Firestore live sync active (${cfg.projectId})`);
    } catch (err) {
      msg.textContent = `Connection failed: ${err.message}`;
      msg.className = "modal-msg err";
    }
  });

  $("#seedFirestoreBtn").addEventListener("click", async () => {
    const msg = $("#firebaseModalMsg");
    try {
      const count = await firebaseBridge.seedMany(state.assessments);
      msg.textContent = `Pushed ${count} assessments to Firestore!`;
      msg.className = "modal-msg ok";
    } catch (err) {
      msg.textContent = err.message;
      msg.className = "modal-msg err";
    }
  });

  $("#disconnectFirebaseBtn").addEventListener("click", async () => {
    await firebaseBridge.disconnect();
    modal.classList.add("hidden");
    toast("Switched back to Local Real-Time SSE Sync");
  });
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

init();
