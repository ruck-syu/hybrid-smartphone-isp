"use strict";
const $ = (s) => document.querySelector(s);
const COLORS = { traditional: "var(--traditional)", learned: "var(--learned)", hybrid: "var(--ours)",
  hybrid_teacher: "var(--teacher)", raw: "var(--muted)", gt: "var(--text)" };
const state = { meta: null, sample: null, file: null, result: null, views: [], a: null, b: null,
  split: 50, zoom: false, pan: { x: 0, y: 0 }, stageTab: null };

/* ------------------------------------------------------------------ boot */
async function boot() {
  try {
    const r = await fetch("/api/meta");
    if (!r.ok) throw new Error();
    state.meta = await r.json();
  } catch {
    status("The server is not responding. Reload the page in a moment.", "error");
    return;
  }
  renderSamples();
  renderPipes();
  bindForm();
  bindViewer();
  loadBenchmark();
  const first = state.meta.samples.find((s) => s.has_gt && s.iso >= 1600) || state.meta.samples[0];
  if (first) { selectSample(first.id); develop({ quiet: true }); }
}

function status(text, kind = "") {
  const el = $("#status");
  el.textContent = text;
  el.className = "status " + kind;
}

/* ------------------------------------------------------------------ inputs */
function renderSamples() {
  const gt = $("#samples-gt"), dng = $("#samples-dng");
  for (const s of state.meta.samples) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "sample";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", "false");
    b.dataset.id = s.id;
    const caption = s.iso ? `${s.camera}, ISO ${s.iso}` : s.camera;
    b.innerHTML = `<img src="${s.thumb}" alt=""><span>${caption}</span>`;
    b.setAttribute("aria-label", caption + (s.has_gt ? ", has ground truth" : ", no ground truth"));
    b.addEventListener("click", () => selectSample(s.id));
    (s.has_gt ? gt : dng).appendChild(b);
  }
  if (!dng.children.length) dng.previousElementSibling.hidden = true;
}

function selectSample(id) {
  state.sample = id;
  state.file = null;
  $("#file").value = "";
  $("#drop").classList.remove("chosen");
  $("#drop-main").textContent = "Drop a RAW file or choose one";
  document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.id === id)));
  const s = state.meta.samples.find((x) => x.id === id);
  status(s.has_gt ? "Ready. This scene has a ground-truth photo, so you will get scores."
                  : "Ready. No ground truth for this phone, so compare the results by eye.");
  updateButton();
}

function chooseFile(f) {
  if (!f) return;
  if (f.size > 80e6) { status("That file is larger than 80 MB. Choose a smaller RAW file.", "error"); return; }
  state.file = f;
  state.sample = null;
  document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-checked", "false"));
  $("#drop").classList.add("chosen");
  $("#drop-main").textContent = f.name;
  status(`Ready to develop ${f.name} (${(f.size / 1e6).toFixed(1)} MB).`);
  updateButton();
}

function renderPipes() {
  const box = $("#pipes");
  const on = new Set(["traditional", "learned", "hybrid"]);
  for (const p of state.meta.pipelines) {
    const l = document.createElement("label");
    l.className = "pipe";
    l.style.setProperty("--c", COLORS[p.id] || "var(--muted)");
    l.innerHTML = `<input type="checkbox" value="${p.id}" ${on.has(p.id) ? "checked" : ""}>
      <b>${p.name}${p.id === "hybrid" ? " (ours)" : ""}</b><small>${p.description}</small>`;
    l.querySelector("input").addEventListener("change", updateButton);
    box.appendChild(l);
  }
}

function selectedPipes() {
  return [...document.querySelectorAll("#pipes input:checked")].map((i) => i.value);
}

function updateButton() {
  $("#develop").disabled = !(state.sample || state.file) || !selectedPipes().length;
  if (!selectedPipes().length) status("Select at least one pipeline.", "error");
}

function bindForm() {
  const drop = $("#drop");
  $("#file").addEventListener("change", (e) => chooseFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) => document.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((ev) => document.addEventListener(ev, (e) => {
    e.preventDefault(); if (ev === "drop" || e.target === document.documentElement) drop.classList.remove("over");
  }));
  document.addEventListener("drop", (e) => chooseFile(e.dataTransfer.files[0]));

  const fmt = {
    noise: (v) => `×${v}`, exposure: (v) => (v == 0 ? "0 stops" : `${v} stops`),
    wb: (v) => `${v > 0 ? "+" : ""}${Math.round(v * 100)}%`,
  };
  for (const id of ["noise", "exposure", "wb"]) {
    const input = $("#" + id);
    const sync = () => { $(`#${id}-out`).textContent = fmt[id](Number(input.value)); condState(); };
    input.addEventListener("input", sync);
    sync();
  }
  $("#nometa").addEventListener("change", condState);
  $("#form").addEventListener("submit", (e) => { e.preventDefault(); develop(); });
}

function condState() {
  const on = Number($("#noise").value) > 1 || Number($("#exposure").value) < 0 ||
    Number($("#wb").value) !== 0 || $("#nometa").checked;
  const el = $("#cond-state");
  el.textContent = on ? "on" : "off";
  el.classList.toggle("on", on);
}

/* ------------------------------------------------------------------ develop */
let timer = null;
async function develop({ quiet = false } = {}) {
  const fd = new FormData();
  if (state.file) fd.append("file", state.file);
  else fd.append("sample", state.sample);
  fd.append("pipelines", selectedPipes().join(","));
  fd.append("noise", $("#noise").value);
  fd.append("exposure", $("#exposure").value);
  fd.append("wb_error", $("#wb").value);
  fd.append("drop_metadata", $("#nometa").checked);
  fd.append("stages", "true");

  const btn = $("#develop");
  btn.disabled = true;
  btn.textContent = "Developing…";
  $("#stage").dataset.state = "busy";
  $("#developing").hidden = false;
  $("#empty").hidden = true;
  const t0 = performance.now();
  timer = setInterval(() => { $("#elapsed").textContent = ((performance.now() - t0) / 1000).toFixed(1) + " s"; }, 100);
  status(state.file ? "Uploading and developing…" : "Developing…");
  if (!quiet && window.matchMedia("(max-width: 960px)").matches) $("#stage").scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const r = await fetch("/api/develop", { method: "POST", body: fd });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || `The server returned an error (${r.status}).`);
    state.result = body;
    showResult(body);
    status(`Developed in ${((performance.now() - t0) / 1000).toFixed(1)} s.`, "done");
  } catch (err) {
    status(err.message === "Failed to fetch" ? "Lost the connection to the server. Try again." : err.message, "error");
    if (!state.result) { $("#empty").hidden = false; }
  } finally {
    clearInterval(timer);
    $("#developing").hidden = true;
    $("#stage").dataset.state = state.result ? "ready" : "empty";
    btn.textContent = "Develop";
    updateButton();
  }
}

function showResult(res) {
  state.views = [{ id: "raw", name: "Sensor data (RAW mosaic)", image: res.raw }];
  if (res.gt) state.views.push({ id: "gt", name: "Ground truth", image: res.gt });
  for (const r of res.results) state.views.push({ id: r.id, name: r.name, image: r.image });
  const best = res.results.find((r) => r.id === "hybrid") || res.results[res.results.length - 1];
  const other = res.results.find((r) => r.id === "traditional" && r !== best) ||
    res.results.find((r) => r !== best);
  setViews(other ? other.id : "raw", best ? best.id : "raw");
  $("#frame").hidden = false;
  $("#viewer-bar").hidden = false;
  const inp = res.input;
  let line = `${res.source}${inp.iso ? `, ISO ${Math.round(inp.iso)}` : ""}. Developed at ${inp.width} × ${inp.height}`;
  if (inp.binned > 1) line += `, reduced from ${inp.megapixels_original} MP to keep it fast`;
  line += inp.metadata ? ", using the camera’s colour and noise metadata."
                       : ". No camera metadata, so white balance, colour and noise are estimated from the image.";
  $("#source").textContent = line;
  const im = new Image();
  im.onload = () => { $("#stage").style.setProperty("--ar", (im.naturalWidth / im.naturalHeight).toFixed(4)); };
  im.src = res.raw;
  renderScore(res);
  renderStages(res);
}

/* ------------------------------------------------------------------ viewer */
function fillSelect(sel, current) {
  sel.innerHTML = "";
  for (const v of state.views) {
    const o = document.createElement("option");
    o.value = v.id; o.textContent = v.name; o.selected = v.id === current;
    sel.appendChild(o);
  }
}

function viewById(id) { return state.views.find((v) => v.id === id) || state.views[0]; }

function setViews(a, b) {
  state.a = a; state.b = b;
  const va = viewById(a), vb = viewById(b);
  $("#img-a").src = va.image; $("#img-a").alt = va.name;
  $("#img-b").src = vb.image; $("#img-b").alt = vb.name;
  $("#tag-a").textContent = va.name; $("#tag-b").textContent = vb.name;
  const ca = COLORS[va.id.split(":")[0]] || "var(--muted)", cb = COLORS[vb.id.split(":")[0]] || "var(--muted)";
  $("#tag-a").style.setProperty("--c", ca); $("#dot-a").style.setProperty("--c", ca);
  $("#tag-b").style.setProperty("--c", cb); $("#dot-b").style.setProperty("--c", cb);
  fillSelect($("#sel-a"), a); fillSelect($("#sel-b"), b);
}

function setSplit(pct) {
  state.split = Math.max(0, Math.min(100, pct));
  $("#frame").style.setProperty("--split", state.split + "%");
  $("#handle").setAttribute("aria-valuenow", Math.round(state.split));
}

function bindViewer() {
  const frame = $("#frame"), handle = $("#handle");
  setSplit(50);
  $("#sel-a").addEventListener("change", (e) => setViews(e.target.value, state.b));
  $("#sel-b").addEventListener("change", (e) => setViews(state.a, e.target.value));
  $("#swap").addEventListener("click", () => setViews(state.b, state.a));
  $("#zoom").addEventListener("click", () => {
    state.zoom = !state.zoom;
    state.pan = { x: 0, y: 0 };
    frame.classList.toggle("zoomed", state.zoom);
    applyPan();
    $("#zoom").setAttribute("aria-pressed", String(state.zoom));
  });

  let mode = null, start = null;
  const pctAt = (e) => { const r = frame.getBoundingClientRect(); return ((e.clientX - r.left) / r.width) * 100; };
  handle.addEventListener("pointerdown", (e) => { mode = "split"; handle.setPointerCapture(e.pointerId); e.preventDefault(); });
  frame.addEventListener("pointerdown", (e) => {
    if (mode) return;
    if (state.zoom) { mode = "pan"; start = { x: e.clientX, y: e.clientY, p: { ...state.pan } }; frame.setPointerCapture(e.pointerId); }
    else if (e.pointerType === "mouse") { mode = "split"; setSplit(pctAt(e)); frame.setPointerCapture(e.pointerId); }
  });
  window.addEventListener("pointermove", (e) => {
    if (mode === "split") setSplit(pctAt(e));
    if (mode === "pan") {
      const r = frame.getBoundingClientRect();
      const lim = 25; // % of the frame, at 2x zoom
      state.pan.x = Math.max(-lim, Math.min(lim, start.p.x + ((e.clientX - start.x) / r.width) * 50));
      state.pan.y = Math.max(-lim, Math.min(lim, start.p.y + ((e.clientY - start.y) / r.height) * 50));
      applyPan();
    }
  });
  window.addEventListener("pointerup", () => { mode = null; });
  handle.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 10 : 2;
    if (e.key === "ArrowLeft") { setSplit(state.split - step); e.preventDefault(); }
    if (e.key === "ArrowRight") { setSplit(state.split + step); e.preventDefault(); }
    if (e.key === "Home") { setSplit(0); e.preventDefault(); }
    if (e.key === "End") { setSplit(100); e.preventDefault(); }
  });
}

function applyPan() {
  const f = $("#frame");
  f.style.setProperty("--px", state.pan.x + "%");
  f.style.setProperty("--py", state.pan.y + "%");
}

/* ------------------------------------------------------------------ scoreboard */
function renderScore(res) {
  const t = $("#score-table");
  const hasGt = res.results.some((r) => r.metrics);
  $("#score-note").textContent = hasGt
    ? "Scored against the ground-truth photo of this scene. Higher PSNR and SSIM are better; lower colour error (ΔE 2000) is better. Best value in each column is highlighted."
    : "There is no ground-truth photo for this file, so only timing is shown. Use the viewer to compare the results.";
  const cols = hasGt
    ? [["psnr", "PSNR (dB)", 1, 2], ["ssim", "SSIM", 1, 3], ["delta_e", "Colour error ΔE", -1, 2], ["ms", "Time (ms)", -1, 0]]
    : [["ms", "Time (ms)", -1, 0]];
  const val = (r, k) => (k === "ms" ? r.ms : r.metrics && r.metrics[k]);
  const best = {};
  for (const [k, , dir] of cols) {
    const vs = res.results.map((r) => val(r, k)).filter((v) => v != null);
    best[k] = dir > 0 ? Math.max(...vs) : Math.min(...vs);
  }
  let html = "<thead><tr><th scope=\"col\">Pipeline</th>" + cols.map((c) => `<th scope="col" class="col-${c[0]}">${c[1]}</th>`).join("") + "</tr></thead><tbody>";
  for (const r of res.results) {
    const d = r.decisions || {};
    const why = [];
    const wbName = { metadata: "camera metadata", grayworld: "grey-world estimate", shades_of_gray: "shades-of-grey estimate" };
    if (r.id === "learned") why.push("ignores camera metadata, learns colour itself");
    if (d.wb_source) why.push(`white balance from ${wbName[d.wb_source] || d.wb_source}`);
    if (d.noise_source) why.push(`noise level ${d.noise_source === "metadata" ? "from metadata" : "estimated from the image"}`);
    if (d.exposure_gain_k && d.exposure_gain_k > 1.01) why.push(`brightened ×${d.exposure_gain_k.toFixed(1)} for the network`);
    html += `<tr class="${r.id === "hybrid" ? "ours" : ""}"><td class="name" style="--c:${COLORS[r.id] || "var(--muted)"}">${r.name}${r.id === "hybrid" ? " (ours)" : ""}${why.length ? `<small>${why.join(", ")}</small>` : ""}</td>`;
    for (const [k, , , dp] of cols) {
      const v = val(r, k);
      html += `<td class="col-${k} ${v === best[k] && res.results.length > 1 ? "best" : ""}">${v == null ? "–" : Number(v).toFixed(dp)}</td>`;
    }
    html += "</tr>";
  }
  t.innerHTML = html + "</tbody>";
  $("#score").hidden = false;
}

/* ------------------------------------------------------------------ stages */
function renderStages(res) {
  const tabs = $("#stage-tabs");
  tabs.innerHTML = "";
  const withStages = res.results.filter((r) => r.stages && r.stages.length);
  if (!withStages.length) { $("#stages").hidden = true; return; }
  if (!withStages.some((r) => r.id === state.stageTab)) state.stageTab = (withStages.find((r) => r.id === "hybrid") || withStages[0]).id;
  for (const r of withStages) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "tab"; b.setAttribute("role", "tab");
    b.style.setProperty("--c", COLORS[r.id] || "var(--muted)");
    b.textContent = r.name;
    b.setAttribute("aria-selected", String(r.id === state.stageTab));
    b.addEventListener("click", () => { state.stageTab = r.id; renderStages(res); });
    tabs.appendChild(b);
  }
  const r = withStages.find((x) => x.id === state.stageTab);
  const film = $("#film");
  film.innerHTML = "";
  film.style.setProperty("--c", COLORS[r.id]);
  const timing = r.timings || {};
  r.stages.forEach((s, i) => {
    const li = document.createElement("li");
    const ms = timing[s.name];
    li.innerHTML = `<button type="button"><img src="${s.image}" alt="${s.name}" loading="lazy"><span class="cap">${s.name}${ms != null ? `<span class="ms">${Math.round(ms)} ms</span>` : ""}</span></button>`;
    li.querySelector("button").addEventListener("click", () => {
      const id = `${r.id}:stage${i}`;
      if (!state.views.find((v) => v.id === id)) state.views.push({ id, name: `${r.name}: ${s.name}`, image: s.image });
      setViews(id, r.id);
      $("#stage").scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    film.appendChild(li);
  });
  $("#stages").hidden = false;
}

/* ------------------------------------------------------------------ benchmark */
const BENCH_ROWS = [
  ["classical_bilateral", "Traditional", "traditional"],
  ["learned", "Learned (2.5M parameters)", "learned"],
  ["hybrid_teacher", "Hybrid teacher (2.5M, FP32)", "hybrid_teacher"],
  ["hybrid_student", "Hybrid (0.64M, FP32)", "hybrid"],
  ["hybrid_student_int8", "Hybrid (0.64M, INT8, deployed)", "hybrid"],
];

function benchTable(summary, rows, metric, label, dp, higherBetter) {
  const present = rows.filter(([k]) => summary[k]);
  if (!present.length) return "";
  const vals = present.map(([k]) => summary[k][metric]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  let h = `<table><thead><tr><th scope="col">Pipeline</th><th scope="col">${label}</th></tr></thead><tbody>`;
  for (const [k, name, c] of present) {
    const v = summary[k][metric];
    const frac = hi === lo ? 1 : (higherBetter ? (v - lo) / (hi - lo) : (hi - v) / (hi - lo));
    h += `<tr><td class="name" style="--c:${COLORS[c]}">${name}</td><td style="--c:${COLORS[c]}">${v.toFixed(dp)}<span class="bar" style="width:${Math.round(20 + 80 * frac)}%"></span></td></tr>`;
  }
  return h + "</tbody></table>";
}

async function loadBenchmark() {
  const body = $("#bench-body");
  try {
    const r = await fetch("/api/benchmark");
    if (!r.ok) throw new Error();
    const b = await r.json();
    const c = b.clean;
    let html = `<p class="lede">Averaged over ${b.n_crops} full-resolution crops from SIDD scenes that were never used for training or tuning (5 phone cameras, ISO 50 to 10,000).</p><div class="bench-grid">`;
    html += `<div><h3>Fidelity</h3><p class="lede">Peak signal-to-noise ratio against the ground truth. Higher is better.</p>${benchTable(c, BENCH_ROWS, "psnr", "PSNR (dB)", 2, true)}</div>`;
    html += `<div><h3>Colour accuracy</h3><p class="lede">Mean CIEDE2000 colour difference. Below 2 is barely visible.</p>${benchTable(c, BENCH_ROWS, "delta_e", "ΔE 2000", 2, false)}</div>`;
    html += `<div><h3>Perceptual quality</h3><p class="lede">LPIPS distance to the ground truth. Lower is better.</p>${benchTable(c, BENCH_ROWS, "lpips", "LPIPS", 3, false)}</div>`;
    if (b.efficiency) {
      const m = b.efficiency.models;
      const rows = [["classical", "Traditional", "traditional", null], ["learned_fp32.onnx", "Learned FP32", "learned", "learned.pt"],
        ["hybrid_student_fp32.onnx", "Hybrid FP32", "hybrid", "hybrid_student.pt"], ["hybrid_student_int8.onnx", "Hybrid INT8 (deployed)", "hybrid", "hybrid_student.pt"]];
      let t = `<table><thead><tr><th scope="col">Pipeline</th><th scope="col">Parameters</th><th scope="col">Size (MB)</th><th scope="col">1 MP (ms)</th><th scope="col">Memory (MB)</th></tr></thead><tbody>`;
      for (const [k, name, col, pt] of rows) {
        const e = m[k]; if (!e) continue;
        const p = pt && m[pt] ? (m[pt].params / 1e6).toFixed(2) + "M" : "–";
        const lat = e["1024x1024"] || {};
        t += `<tr><td class="name" style="--c:${COLORS[col]}">${name}</td><td>${p}</td><td>${e.file_size_mb ? e.file_size_mb.toFixed(2) : "–"}</td><td>${lat.latency_ms_median ? Math.round(lat.latency_ms_median) : "–"}</td><td>${lat.peak_rss_mb ? Math.round(lat.peak_rss_mb) : "–"}</td></tr>`;
      }
      html += `<div><h3>Cost</h3><p class="lede">Whole pipeline on a laptop CPU with 4 threads, 1-megapixel RAW.</p>${t}</tbody></table></div>`;
    }
    body.innerHTML = html + "</div>";
  } catch {
    body.innerHTML = `<p class="muted">Benchmark tables are not bundled with this deployment. See the repository for full results.</p>`;
  }
}

boot();
