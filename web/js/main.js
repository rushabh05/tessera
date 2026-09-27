// Live Detector tab. Vanilla JS, ES modules, no framework, no CDN - every
// dependency is a browser built-in (Web Crypto for SHA-256) or a file here.
// Boots on import (app.js imports it the first time the tab is needed).

import { forward } from "./forward.js";
import { MerkleLog, setSha256, verifyInclusion, bytesToHex, canonicalBytes } from "./merkle.js";
import { store } from "./store.js";
import { escapeHtml } from "./ui/dom.js";

setSha256(async (bytes) => new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)));

const MODALITY_LABELS = {
  m1_log: "Log templates",
  m2_metrics: "Network metrics",
  m3_identity: "Host identity",
  m4_graph: "Graph structure",
};
const MODALITY_ORDER = ["m1_log", "m2_metrics", "m3_identity", "m4_graph"];
// Real AIT host roles. (An earlier export swapped two of these labels; the
// role is derived from the host name here so the UI is right regardless.)
const HOST_LABELS = {
  vpn: "VPN Gateway",
  intranet_server: "Intranet Server",
  "inet-firewall": "Internet Firewall",
};
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 50;

let pretrainedWeights, activeWeights, demo, golden, log, windows;
let selectedIndex = null;
let liveAvailability = null;

async function rebuildLedger() {
  log = new MerkleLog();
  for (const w of windows) await log.appendJson(w.canonical_verdict);
}

const fmtPct = (x) => `${(x * 100).toFixed(1)}%`;
const hostLabel = (w) => HOST_LABELS[w.host] || w.host_label;
function scoreClass(score) {
  if (score < 0.3) return "score-low";
  if (score < 0.6) return "score-mid";
  return "score-high";
}

function scoreWith(w, availability) {
  return forward(w.features, availability, activeWeights);
}

function renderWindowList() {
  const list = document.getElementById("window-list");
  list.innerHTML = "";
  windows.forEach((w, i) => {
    const avail = MODALITY_ORDER.map((n) => (w.availability[n] ? 1 : 0));
    const score = activeWeights === pretrainedWeights ? w.score : scoreWith(w, avail).score;
    const row = document.createElement("button");
    row.type = "button";
    row.className = `window-row ${scoreClass(score)}` + (i === selectedIndex ? " selected" : "");
    row.style.animationDelay = `${Math.min(i, 24) * 18}ms`;
    row.setAttribute("aria-pressed", i === selectedIndex ? "true" : "false");
    row.innerHTML = `
      <span class="w-main"><span class="w-host">${escapeHtml(hostLabel(w))}</span><span class="w-id">${escapeHtml(w.window_id)}</span></span>
      <span class="w-meter" aria-hidden="true"><span style="width:${(score * 100).toFixed(1)}%"></span></span>
      <span class="w-score">${fmtPct(score)}</span>
    `;
    row.addEventListener("click", () => selectWindow(i));
    list.appendChild(row);
  });
}

async function selectWindow(i) {
  selectedIndex = i;
  const w = windows[i];
  liveAvailability = MODALITY_ORDER.map((name) => (w.availability[name] ? 1 : 0));
  document.querySelectorAll("#window-list .window-row").forEach((row, j) => {
    row.classList.toggle("selected", j === i);
    row.setAttribute("aria-pressed", j === i ? "true" : "false");
  });
  recompute();
  renderMerklePanel();
}

function recompute() {
  if (selectedIndex === null) return;
  const w = windows[selectedIndex];
  const { score, attribution } = scoreWith(w, liveAvailability);

  document.getElementById("detail-empty").hidden = true;
  document.getElementById("detail-content").hidden = false;
  document.getElementById("detail-host").textContent = hostLabel(w);
  document.getElementById("detail-id").textContent = w.window_id;
  document.getElementById("detail-score").textContent = fmtPct(score);
  const gauge = document.getElementById("score-gauge");
  gauge.className = `score-gauge ${scoreClass(score)}`;
  document.getElementById("gauge-arc").style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE * (1 - score));
  const verdict = document.getElementById("detail-verdict");
  const isAttack = score >= 0.5;
  verdict.innerHTML = isAttack
    ? `<span class="pill bad"><span class="dot"></span>Flagged as attack</span>`
    : `<span class="pill good"><span class="dot"></span>Looks benign</span>`;

  const bars = document.getElementById("attribution-bars");
  const existing = bars.querySelectorAll(".attr-row");
  if (existing.length !== MODALITY_ORDER.length) {
    bars.innerHTML = "";
    MODALITY_ORDER.forEach((name, m) => {
      const row = document.createElement("div");
      row.className = "attr-row";
      row.innerHTML = `
        <label class="attr-toggle switch">
          <input type="checkbox" data-modality="${m}" />
          <span></span>${MODALITY_LABELS[name]}
        </label>
        <div class="attr-bar-track"><div class="attr-bar-fill" style="width:0%"></div></div>
        <span class="attr-pct">0%</span>
      `;
      row.querySelector("input").addEventListener("change", (e) => {
        liveAvailability[m] = e.target.checked ? 1 : 0;
        recompute();
      });
      bars.appendChild(row);
    });
  }
  bars.querySelectorAll(".attr-row").forEach((row, m) => {
    const pct = attribution[m] * 100;
    const cb = row.querySelector("input");
    cb.checked = liveAvailability[m] > 0.5;
    // M3 (identity) is always derived, never missing in real data - keep it on.
    cb.disabled = MODALITY_ORDER[m] === "m3_identity";
    row.querySelector(".attr-bar-fill").style.width = `${pct}%`;
    row.querySelector(".attr-pct").textContent = `${pct.toFixed(1)}%`;
  });

  // Live parity check: if this window is also a golden vector, show that the
  // browser reproduces the automated Node.js parity test's expected score.
  const parityNote = document.getElementById("parity-note");
  const goldenMatch =
    activeWeights === pretrainedWeights &&
    golden.vectors.find((g) => JSON.stringify(g.input) === JSON.stringify(w.features));
  if (goldenMatch) {
    const delta = Math.abs(score - goldenMatch.expected_score);
    parityNote.textContent = `Parity check: this window is also a golden test vector. The live browser score matches PyTorch to within ${delta.toExponential(2)}.`;
    parityNote.hidden = false;
  } else {
    parityNote.hidden = true;
  }
}

function renderMerklePanel() {
  if (selectedIndex === null || !log) return;
  document.getElementById("merkle-empty").hidden = true;
  document.getElementById("merkle-content").hidden = false;
  document.getElementById("merkle-window-id").textContent = windows[selectedIndex].window_id;
  document.getElementById("tamper-result").hidden = true;
  document.getElementById("inclusion-result").hidden = true;
  document.getElementById("tamper-score-input").value = windows[selectedIndex].score.toFixed(4);
}

async function verifyInclusionForSelected() {
  const i = selectedIndex;
  const proof = await log.inclusionProof(i);
  const root = await log.root();
  const ok = await verifyInclusion(proof.leaf, i, proof.treeSize, proof.path, root);
  const out = document.getElementById("inclusion-result");
  out.innerHTML = ok
    ? `<span class="pill good"><span class="dot"></span>Inclusion proof verifies</span> <span class="small muted">path length ${proof.path.length}, root <span class="hash">${bytesToHex(root).slice(0, 16)}…</span></span>`
    : `<span class="pill bad"><span class="dot"></span>Proof failed to verify</span>`;
  out.hidden = false;
}

async function runTamperDemo() {
  const i = selectedIndex;
  const retainedRoot = await log.root();
  const before = await log.inclusionProof(i);

  const newScore = Number.parseFloat(document.getElementById("tamper-score-input").value);
  const tamperedVerdict = {
    ...windows[i].canonical_verdict,
    score: Number.isFinite(newScore) ? newScore.toFixed(6) : windows[i].canonical_verdict.score,
    verdict: 0,
  };
  await log.tamper(i, canonicalBytes(tamperedVerdict));

  const newRoot = await log.root();
  const rootChanged = bytesToHex(newRoot) !== bytesToHex(retainedRoot);
  const originalStillVerifies = await verifyInclusion(before.leaf, i, before.treeSize, before.path, retainedRoot);
  const tamperedLeafOnRetainedRoot = await verifyInclusion(
    (await log.inclusionProof(i)).leaf,
    i,
    before.treeSize,
    before.path,
    retainedRoot
  );

  const line = (ok, text) =>
    `<div class="row" style="flex-wrap:nowrap;align-items:flex-start"><span class="pill ${ok ? "good" : "bad"}"><span class="dot"></span>${ok ? "Pass" : "Fail"}</span><span class="small">${text}</span></div>`;
  const out = document.getElementById("tamper-result");
  out.hidden = false;
  out.innerHTML = `
    <div class="small">Root before edit: <span class="hash">${bytesToHex(retainedRoot).slice(0, 32)}…</span></div>
    <div class="small">Root after edit: <span class="hash">${bytesToHex(newRoot).slice(0, 32)}…</span></div>
    ${line(rootChanged, "Editing the verdict changed the tree root.")}
    ${line(originalStillVerifies, "The original proof still verifies against the original retained root, so nothing about the past was lost.")}
    ${line(!tamperedLeafOnRetainedRoot, "The edited entry's proof does <strong>not</strong> reconstruct the retained root. Anyone who saved that root can detect the edit.")}
  `;

  // Restore the log so the demo stays repeatable.
  await log.tamper(i, canonicalBytes(windows[i].canonical_verdict));
}

function renderModelSwitch() {
  const host = document.getElementById("model-switch");
  const lab = store.getLabModel();
  const kind = store.getActiveModelKind();
  if (!lab) {
    host.innerHTML = `<span class="badge"><span class="dot" style="color:var(--good)"></span>Model: pretrained on real AIT data</span>
      <a class="btn sm" href="#lab">Train your own in the Lab</a>`;
    return;
  }
  host.innerHTML = `
    <span class="label">Model</span>
    <div class="seg" role="group" aria-label="Model to score with">
      <button type="button" data-kind="pretrained" aria-pressed="${kind === "pretrained"}">Pretrained (real data)</button>
      <button type="button" data-kind="lab" aria-pressed="${kind === "lab"}">Your Lab model</button>
    </div>`;
  host.querySelectorAll("button[data-kind]").forEach((b) =>
    b.addEventListener("click", () => store.setActiveModelKind(b.dataset.kind))
  );
}

async function applyActiveModel() {
  const m = await store.getActiveModel();
  activeWeights = m.kind === "pretrained" ? pretrainedWeights : m.weights;
  document.getElementById("detail-model-label").textContent =
    m.kind === "pretrained"
      ? "Pretrained on real AIT data (7 replicas, santos held out)"
      : `${m.label} (trained in your browser on synthetic data)`;
  renderModelSwitch();
  renderWindowList();
  if (selectedIndex !== null) recompute();
}

function disableLedger(reason) {
  log = null;
  document.getElementById("root-hex").textContent = "unavailable";
  const empty = document.getElementById("merkle-empty");
  empty.hidden = false;
  empty.textContent = reason;
  document.getElementById("merkle-content").hidden = true;
}

async function boot() {
  [pretrainedWeights, demo, golden] = await Promise.all([
    store.loadPretrainedWeights(),
    store.loadJSON("data/demo_windows.json"),
    store.loadJSON("data/golden.json"),
  ]);
  activeWeights = pretrainedWeights;
  windows = demo.windows;
  // Register before resolving the active model: a Lab model may already exist
  // (or arrive while we boot), and the detector must score with it, not with a
  // hard-coded pretrained default.
  store.on("labModel", () => applyActiveModel());
  store.on("activeModel", () => applyActiveModel());

  // The ledger needs Web Crypto, which browsers only expose on https or
  // localhost. Scoring does not, so a missing ledger must not take the
  // detector down with it.
  if (!globalThis.crypto?.subtle) {
    disableLedger("The ledger needs Web Crypto, which your browser only enables on https:// or localhost. Scoring still works.");
  } else {
    try {
      await rebuildLedger();
      const rootHex = bytesToHex(await log.root());
      document.getElementById("root-hex").textContent = `${rootHex.slice(0, 16)}…`;
      document.getElementById("root-hex").title = rootHex;
    } catch (err) {
      console.error(err);
      disableLedger(`The ledger could not be built (${err.message}). Scoring still works.`);
    }
  }
  document.getElementById("n-windows").textContent = String(windows.length);
  const countLinear = (l) => l.weight.flat().length + l.bias.length;
  const w = pretrainedWeights;
  const nParams =
    w.encoders.reduce((n, e) => n + countLinear(e.linear0) + countLinear(e.linear1) + e.groupnorm.weight.length * 2, 0) +
    countLinear(w.fusion.gate_linear0) +
    countLinear(w.fusion.gate_linear1) +
    countLinear(w.head.linear0) +
    countLinear(w.head.linear1);
  document.getElementById("n-params").textContent = nParams.toLocaleString("en-US");

  document.getElementById("verify-inclusion-btn").addEventListener("click", verifyInclusionForSelected);
  document.getElementById("tamper-btn").addEventListener("click", runTamperDemo);

  await applyActiveModel();
  selectWindow(0);
}

boot().catch((err) => {
  console.error(err);
  const el = document.getElementById("boot-error");
  el.hidden = false;
  el.textContent = `Failed to load the detector: ${err.message}`;
});
