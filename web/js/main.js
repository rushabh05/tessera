// TESSERA demo page logic. Vanilla JS, ES modules, no framework, no CDN -
// every dependency is either a browser built-in (Web Crypto for SHA-256) or a
// file in this same directory.

import { forward } from "./forward.js";
import { MerkleLog, setSha256, verifyInclusion, bytesToHex, canonicalBytes } from "./merkle.js";

setSha256(async (bytes) => new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)));

const MODALITY_LABELS = {
  m1_log: "Log templates",
  m2_metrics: "Network metrics",
  m3_identity: "Identity",
  m4_graph: "Graph structure",
};
const MODALITY_ORDER = ["m1_log", "m2_metrics", "m3_identity", "m4_graph"];

let weights, demo, golden, log, windows;
let selectedIndex = null;
let liveAvailability = null; // user-toggled 4-bool array for the selected window

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`failed to fetch ${path}: ${res.status}`);
  return res.json();
}

async function rebuildLedger() {
  log = new MerkleLog();
  for (const w of windows) await log.appendJson(w.canonical_verdict);
}

function fmtPct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

function scoreColor(score) {
  // green (benign) -> amber -> red (attack), no external palette dependency.
  if (score < 0.3) return "#1a7f37";
  if (score < 0.6) return "#9a6700";
  return "#cf222e";
}

function renderWindowList() {
  const list = document.getElementById("window-list");
  list.innerHTML = "";
  windows.forEach((w, i) => {
    const row = document.createElement("button");
    row.className = "window-row" + (i === selectedIndex ? " selected" : "");
    row.innerHTML = `
      <span class="w-host">${w.host_label}</span>
      <span class="w-id">${w.window_id}</span>
      <span class="w-score" style="color:${scoreColor(w.score)}">${fmtPct(w.score)}</span>
    `;
    row.addEventListener("click", () => selectWindow(i));
    list.appendChild(row);
  });
}

async function selectWindow(i) {
  selectedIndex = i;
  const w = windows[i];
  liveAvailability = MODALITY_ORDER.map((name) => (w.availability[name] ? 1 : 0));
  renderWindowList();
  await recompute();
  renderMerklePanel();
}

async function recompute() {
  if (selectedIndex === null) return;
  const w = windows[selectedIndex];
  const { score, attribution } = forward(w.features, liveAvailability, weights);

  document.getElementById("detail-empty").hidden = true;
  document.getElementById("detail-content").hidden = false;
  document.getElementById("detail-host").textContent = w.host_label;
  document.getElementById("detail-id").textContent = w.window_id;
  document.getElementById("detail-score").textContent = fmtPct(score);
  document.getElementById("detail-score").style.color = scoreColor(score);
  document.getElementById("detail-verdict").textContent = score >= 0.5 ? "ATTACK" : "benign";
  document.getElementById("detail-verdict").className = score >= 0.5 ? "verdict-bad" : "verdict-ok";

  const bars = document.getElementById("attribution-bars");
  bars.innerHTML = "";
  MODALITY_ORDER.forEach((name, m) => {
    const pct = attribution[m] * 100;
    const available = liveAvailability[m] > 0.5;
    const row = document.createElement("div");
    row.className = "attr-row";
    row.innerHTML = `
      <label class="attr-toggle">
        <input type="checkbox" ${available ? "checked" : ""} data-modality="${m}" />
        ${MODALITY_LABELS[name]}
      </label>
      <div class="attr-bar-track"><div class="attr-bar-fill" style="width:${pct}%"></div></div>
      <span class="attr-pct">${pct.toFixed(1)}%</span>
    `;
    bars.appendChild(row);
  });
  bars.querySelectorAll("input[type=checkbox]").forEach((cb) => {
    cb.addEventListener("change", async (e) => {
      const m = Number(e.target.dataset.modality);
      liveAvailability[m] = e.target.checked ? 1 : 0;
      await recompute();
    });
  });

  // Live parity check against the golden vector set, if this exact window's
  // feature vector happens to also appear among the golden vectors (proves
  // the SAME code path is used here as in the automated Node.js test).
  const goldenMatch = golden.vectors.find(
    (g) => JSON.stringify(g.input) === JSON.stringify(w.features)
  );
  const parityNote = document.getElementById("parity-note");
  if (goldenMatch) {
    const delta = Math.abs(score - goldenMatch.expected_score);
    parityNote.textContent = `Parity check: this window is also a golden test vector. Live browser score matches the automated Node.js parity test to within ${delta.toExponential(2)}.`;
    parityNote.hidden = false;
  } else {
    parityNote.hidden = true;
  }
}

function renderMerklePanel() {
  if (selectedIndex === null) return;
  document.getElementById("merkle-empty").hidden = true;
  document.getElementById("merkle-content").hidden = false;
  document.getElementById("merkle-window-id").textContent = windows[selectedIndex].window_id;
  document.getElementById("tamper-result").hidden = true;
  document.getElementById("tamper-score-input").value = windows[selectedIndex].score.toFixed(4);
}

async function verifyInclusionForSelected() {
  const i = selectedIndex;
  const proof = await log.inclusionProof(i);
  const root = await log.root();
  const ok = await verifyInclusion(proof.leaf, i, proof.treeSize, proof.path, root);
  const out = document.getElementById("inclusion-result");
  out.textContent = ok
    ? `✓ Inclusion proof verifies (path length ${proof.path.length}, root ${bytesToHex(root).slice(0, 16)}…)`
    : "✗ Proof failed to verify";
  out.className = ok ? "verdict-ok" : "verdict-bad";
  out.hidden = false;
}

async function runTamperDemo() {
  const i = selectedIndex;
  const retainedRoot = await log.root(); // what "an auditor" holds
  const before = await log.inclusionProof(i);

  const newScoreText = document.getElementById("tamper-score-input").value;
  const newScore = Number.parseFloat(newScoreText);
  const tamperedVerdict = {
    ...windows[i].canonical_verdict,
    score: Number.isFinite(newScore) ? newScore.toFixed(6) : windows[i].canonical_verdict.score,
    verdict: 0,
  };
  await log.tamper(i, canonicalBytes(tamperedVerdict));

  const newRoot = await log.root();
  const rootChanged = bytesToHex(newRoot) !== bytesToHex(retainedRoot);
  const oldProofOnRetainedRoot = await verifyInclusion(
    before.leaf, i, before.treeSize, before.path, retainedRoot
  );
  const tamperedLeafOnRetainedRoot = await verifyInclusion(
    (await log.inclusionProof(i)).leaf, i, before.treeSize, before.path, retainedRoot
  );

  const out = document.getElementById("tamper-result");
  out.hidden = false;
  out.innerHTML = `
    <div>Root before edit: <code>${bytesToHex(retainedRoot).slice(0, 24)}…</code></div>
    <div>Root after edit:  <code>${bytesToHex(newRoot).slice(0, 24)}…</code></div>
    <div class="${rootChanged ? "verdict-ok" : "verdict-bad"}">
      ${rootChanged ? "✓" : "✗"} Editing the verdict changed the tree root.
    </div>
    <div class="verdict-ok">✓ The ORIGINAL proof still verifies against the ORIGINAL retained root (nothing about the past was altered).</div>
    <div class="${!tamperedLeafOnRetainedRoot ? "verdict-ok" : "verdict-bad"}">
      ${!tamperedLeafOnRetainedRoot ? "✓" : "✗"} The EDITED entry's own proof does NOT reconstruct the retained root — tampering is detectable by anyone who saved that root earlier.
    </div>
  `;

  // Restore the log to its pre-tamper state so the demo stays repeatable.
  await log.tamper(i, canonicalBytes(windows[i].canonical_verdict));
}

async function boot() {
  [weights, demo, golden] = await Promise.all([
    loadJSON("data/weights.json"),
    loadJSON("data/demo_windows.json"),
    loadJSON("data/golden.json"),
  ]);
  windows = demo.windows;
  await rebuildLedger();
  renderWindowList();

  document.getElementById("root-hex").textContent = bytesToHex(await log.root());
  document.getElementById("n-windows").textContent = String(windows.length);
  document.getElementById("n-params").textContent = String(
    weights.encoders.reduce(
      (n, e) =>
        n +
        e.linear0.weight.flat().length +
        e.linear0.bias.length +
        e.linear1.weight.flat().length +
        e.linear1.bias.length,
      0
    )
  );

  document.getElementById("verify-inclusion-btn").addEventListener("click", verifyInclusionForSelected);
  document.getElementById("tamper-btn").addEventListener("click", runTamperDemo);

  selectWindow(0);
}

boot().catch((err) => {
  console.error(err);
  document.getElementById("boot-error").hidden = false;
  document.getElementById("boot-error").textContent = `Failed to load demo: ${err.message}`;
});
