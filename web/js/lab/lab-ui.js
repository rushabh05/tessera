// Training Lab tab: pick the data, the train / validation / test split and the
// model, then watch the whole pipeline run in your browser (a module Web Worker)
// and explore the results.
//
// Honesty rules this module keeps:
//   * every window here is SYNTHETIC (datagen.js, calibrated from summary
//     statistics of the 8 real AIT replicas) and is labelled .tag-synthetic;
//   * the only real numbers shown are read from data/real_results.json and are
//     labelled .tag-real with the test / command that recorded them;
//   * average precision (AP) is the headline metric, MCC second; accuracy is
//     shown de-emphasised with a note;
//   * low-support folds (< 20 test attack windows) are shown but kept out of
//     summary means;
//   * the live leakage certificate is computed on exactly the split the run
//     will use (same corpus, same config, same feature set), so what you see
//     is what runs;
//   * the split experiment (random vs chronological, same model, same windows)
//     reports whatever gap it measures, in plain words, next to the REAL
//     recorded Finding 1; it never tunes anything to get a gap.
//
// The main thread generates the corpus only for the live split preview and the
// leakage certificate (cached per scale + seed). The run itself happens in
// worker.js; if a module worker cannot start, runPipeline runs here instead.

import { h, fmt, countUp, stagger, reducedMotion, toast as domToast, downloadFile, debounce, syncRangeFill, prefs, escapeHtml } from "../ui/dom.js";
import { icon } from "../ui/icons.js";
import {
  lineChart,
  curveChart,
  barChart,
  histogram as histogramChart,
  stackedBar,
  confusionMatrix,
  ring,
  withTableToggle,
  dataTable,
  hideTooltip,
} from "../ui/charts.js";
import { generateCorpus, corpusSummary } from "./datagen.js";
import {
  DEFAULT_SPLIT,
  PRETRAINED_TRAIN_REPLICAS,
  REPLICA_ORDER,
  buildSplit,
  cloneSplitConfig,
  leakageCertificate,
  loroFolds,
  validateSplitConfig,
} from "./splits.js";
import { metricsAt, MIN_SUPPORT_FOR_RATES } from "./metrics.js";
import { makeRng } from "./rng.js";
import { stageWeights, mergeCertificates, KNN_DEFAULT_K } from "./pipeline.js";

/* ================================================================== constants */

const STEPS = [
  { id: "generate", label: "Generate data", idle: "Sample synthetic windows from the real-data statistics", share: 5 },
  { id: "split", label: "Split", idle: "Assign every window to training, validation or test", share: 3 },
  { id: "leakage", label: "Leakage check", idle: "Make sure the test set cannot see what training saw", share: 5 },
  { id: "train", label: "Train", idle: "Fit the model on the training windows", share: 70 },
  { id: "test", label: "Test", idle: "Score the held-out test windows", share: 7 },
  { id: "evaluate", label: "Evaluate", idle: "Average precision, MCC, curves and calibration", share: 7 },
  { id: "ledger", label: "Commit to ledger", idle: "Write every verdict into a tamper-evident Merkle log", share: 3 },
];

const SCALES = [
  { id: "small", scale: 0.02, label: "Small", pct: "2%" },
  { id: "medium", scale: 0.05, label: "Medium", pct: "5%" },
  { id: "large", scale: 0.1, label: "Large", pct: "10%" },
];

const STRATEGIES = [
  {
    id: "default",
    label: "Default: hold out santos",
    short: "Default (santos held out)",
    honesty: { cls: "good", icon: "check", text: "Honest" },
    explain:
      "Train on 7 testbeds and test on santos, a testbed the model never sees. This reproduces the real project's headline neural-model fold (replica hold-out).",
  },
  {
    id: "pick",
    label: "Pick replicas",
    short: "Picked replicas",
    honesty: { cls: "good", icon: "check", text: "Honest" },
    explain:
      "Choose which testbeds the model learns from and which it is tested on. A testbed can only be on one side, so the test is always on unseen testbeds (replica hold-out).",
  },
  {
    id: "random",
    label: "Random split",
    short: "Random split",
    honesty: { cls: "bad", icon: "alert", text: "Leaky" },
    explain:
      "Shuffle every window and deal them out by percentage. Common in published work, but neighbouring minutes of the same attack land on both sides, so the test partly measures memory (row-level random split).",
  },
  {
    id: "chrono",
    label: "Chronological",
    short: "Chronological",
    honesty: { cls: "good", icon: "clock", text: "Honest in time" },
    explain:
      "Within each host's timeline, the earliest windows train and the latest test, so the future never trains the past. Windows that continue an attack or quiet period across a cut are set aside, for the same reason the real protocol leaves a 600-second gap after each cut. The test still comes from testbeds the model trained on, so it measures generalisation to later traffic on known networks, not to a new network (temporal split).",
  },
  {
    id: "loro",
    label: "All 8 folds (LORO)",
    short: "LORO (8 folds)",
    honesty: { cls: "good", icon: "shield", text: "Honest, strongest" },
    explain:
      "Run 8 experiments: each testbed is held out once while the model trains on the other 7. The strongest test of generalisation; the summary leaves out folds with fewer than 20 attack windows (leave-one-replica-out).",
  },
];
const STRATEGY = Object.fromEntries(STRATEGIES.map((s) => [s.id, s]));

const MODELS = [
  {
    id: "tessera",
    label: "TESSERA-base (train in your browser)",
    short: "TESSERA-base",
    desc: "The project's 5,005-parameter multimodal network, trained from scratch here on the synthetic training windows.",
  },
  {
    id: "pretrained",
    label: "Pretrained on real data",
    short: "Pretrained (real)",
    desc: "The same network, already trained on real AIT windows from the 7 replicas other than santos. No training step: it only scores the test set.",
  },
  {
    id: "logreg",
    label: "Logistic regression (baseline)",
    short: "Logistic regression",
    desc: "A simple linear baseline on log-scaled features. It trains in well under a second and shows what the network adds.",
  },
  {
    id: "knn",
    label: "Nearest neighbour (memoriser)",
    short: "Memoriser (k-NN)",
    desc: "Copies the labels of the most similar training windows - it shows how much a leaky split can flatter a model.",
  },
];
const MODEL = Object.fromEntries(MODELS.map((m) => [m.id, m]));

const FEATURES = [
  { id: "all", label: "All four sources (42)", sub: "logs, network metrics, host, graph", short: "All four sources (42)", table: "All 42" },
  { id: "m2", label: "Network metrics only (24)", sub: "the setup of real Finding 1", short: "Network metrics only (24)", table: "Network metrics (24)" },
];
const FEATURE = Object.fromEntries(FEATURES.map((f) => [f.id, f]));

/** The split experiment: the same model on the same windows, split two ways. */
const EXP_SPLITS = [
  { id: "random", strategy: "random", mode: "random", label: "Random split", short: "random" },
  { id: "chrono", strategy: "chrono", mode: "chronological", label: "Chronological split", short: "chronological" },
];
const EXP_MODELS = {
  knn: { label: "Memoriser (nearest neighbour)", chip: `Memoriser (nearest neighbour, k = ${KNN_DEFAULT_K})`, name: "memoriser" },
  tessera: { label: "TESSERA-base", chip: "TESSERA-base (default training settings)", name: "TESSERA-base network" },
};

const MODALITIES = [
  { id: "m1_log", code: "M1", label: "Log templates", color: "--series-1" },
  { id: "m2_metrics", code: "M2", label: "Network metrics", color: "--series-2" },
  { id: "m3_identity", code: "M3", label: "Host identity", color: "--series-3" },
  { id: "m4_graph", code: "M4", label: "Graph structure", color: "--series-7" },
];

const LR_OPTIONS = [3e-4, 1e-3, 3e-3];
const BATCH_OPTIONS = [64, 128, 256];
const WINDOWS_PER_DAY = 1440;
const LOGREG_STEPS = 200;
const TESSERA_PARAMS = 5005;
const PREFS_KEY = "lab-config";
const KEEP_RESULTS = 8;

const ROLE_INFO = {
  train: { label: "Train", color: "--c-train" },
  test: { label: "Test", color: "--c-test" },
  // neutral on purpose: aqua (--c-val) means Validation in the split bar below
  mixed: { label: "Train + test", color: "--text-2" },
  unused: { label: "Not used", color: "--c-unused" },
};

const SEVERITY = {
  ok: { cls: "good", icon: "check", word: "Pass" },
  warn: { cls: "warn", icon: "alert", word: "Warning" },
  fail: { cls: "bad", icon: "x", word: "Fail" },
};
/** The LEAKAGE verdict (certificate.overall): leakage checks only, never test support. */
const OVERALL = {
  ok: { cls: "good", icon: "check", word: "No leak found" },
  warn: { cls: "warn", icon: "alert", word: "Worth a look" },
  fail: { cls: "bad", icon: "x", word: "Leak found" },
};
/** Delay before a just-started run's Cancel button accepts clicks (double-click guard). */
const CANCEL_ARM_MS = 500;

const DICE_SVG =
  '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="4" width="16" height="16" rx="3.5"/><circle cx="9" cy="9" r="1.1" fill="currentColor"/><circle cx="15" cy="15" r="1.1" fill="currentColor"/><circle cx="15" cy="9" r="1.1" fill="currentColor"/><circle cx="9" cy="15" r="1.1" fill="currentColor"/><circle cx="12" cy="12" r="1.1" fill="currentColor"/></svg>';

/* ================================================================== small helpers */

const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const f3 = (v) => (isNum(v) ? v.toFixed(3) : "—");
const f4 = (v) => (isNum(v) ? v.toFixed(4) : "—");
const pct1 = (v) => (isNum(v) ? `${(v * 100).toFixed(1)}%` : "—");
const clampNum = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const plural = (n, one, many = `${one}s`) => `${fmt.int(n)} ${n === 1 ? one : many}`;
const lrText = (v) => (v === 1e-3 ? "0.001" : v === 3e-4 ? "0.0003" : v === 3e-3 ? "0.003" : String(v));
const plainIds = (ids) => (ids.length <= 3 ? ids.join(", ") : `${ids.length} replicas`);
const joinIds = (ids) => (ids.length <= 1 ? ids.join("") : `${ids.slice(0, -1).join(", ")} and ${ids[ids.length - 1]}`);
/** An AP difference in points, signed: 0.196 -> "+19.6 AP points". */
const signedPts = (v) => (isNum(v) ? `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(1)} AP points` : "—");

/**
 * False alerts per monitored host-day from what the test set measured:
 * fp of `benign` benign test windows flagged, scaled to `perDay` one-minute
 * windows. With fp = 0 the point estimate says nothing about resolution, so
 * `upper` is the 95% rule-of-three bound, 3 / benign x perDay.
 * -> {measurable, fp, benign, rate, upper} (rate / upper null when not measurable).
 */
export function falseAlertRate(fp, benign, perDay = WINDOWS_PER_DAY) {
  if (!(benign > 0)) return { measurable: false, fp, benign: 0, rate: null, upper: null };
  return { measurable: true, fp, benign, rate: (fp / benign) * perDay, upper: fp === 0 ? (3 / benign) * perDay : null };
}

/** A per-day alert count for prose: one decimal under 10, whole numbers above. */
export function alertCountText(v) {
  if (!isNum(v)) return "—";
  if (v < 10) return (Math.round(v * 10) / 10).toFixed(1).replace(/\.0$/, "");
  return fmt.int(v);
}

/**
 * The test-support verdict (certificate.support) as a pill spec, or null when
 * the test set has enough attacks. Never a leakage statement: a fold without
 * attacks is unscoreable, not leaky.
 */
export function supportInfo(cert) {
  if (!cert || !cert.support || cert.support === "ok") return null;
  const excluded = Array.isArray(cert.excludedLowSupport) ? cert.excludedLowSupport : null;
  if (excluded && cert.perFold) {
    // a merged leave-one-replica-out certificate
    if (cert.support === "fail") return { cls: "warn", icon: "alert", word: "No fold has enough attacks to score", short: "no fold scoreable" };
    const k = excluded.length;
    return { cls: "warn", icon: "alert", word: `${k} ${k === 1 ? "fold has" : "folds have"} too few attacks to score`, short: `${excluded.join(", ")}: too few attacks` };
  }
  if (cert.support === "fail") return { cls: "warn", icon: "alert", word: "No attacks to score", short: "no attacks in the test set" };
  return { cls: "warn", icon: "alert", word: "Too few attacks to score", short: "too few attacks" };
}

function iconSpan(name, cls = "lab-ico") {
  return h("span", { class: cls, html: icon(name), "aria-hidden": "true" });
}

function pill(cls, iconName, text, extra = {}) {
  return h("span", { class: `pill ${cls}`, html: `${icon(iconName)}<span>${escapeHtml(text)}</span>`, ...extra });
}

/**
 * Keep a row of .lab-run-chip spans in step with specs [{key, icon, text, cls?}]
 * without rebuilding it: a chip is created (and fades in) only the first time
 * its key appears, afterwards only its text changes, and a chip whose key is
 * missing is hidden. Dragging a slider therefore never restarts the fade.
 */
function syncChips(container, specs) {
  if (!container) return;
  const map = container._chips || (container._chips = new Map());
  const want = new Set(specs.map((s0) => s0.key));
  for (const [key, c] of map) c.el.hidden = !want.has(key);
  let prev = null;
  for (const s0 of specs) {
    let c = map.get(s0.key);
    if (!c) {
      const text = h("span", {}, s0.text);
      const el = h("span", { class: `lab-run-chip is-new${s0.cls ? ` ${s0.cls}` : ""}` }, iconSpan(s0.icon), text);
      el.addEventListener("animationend", () => el.classList.remove("is-new"), { once: true });
      c = { el, text, value: s0.text };
      map.set(s0.key, c);
    } else if (c.value !== s0.text) {
      c.text.textContent = s0.text;
      c.value = s0.text;
    }
    c.el.hidden = false;
    // keep the chips in spec order (a no-op once they are in place)
    const next = prev ? prev.nextSibling : container.firstChild;
    if (next !== c.el) container.insertBefore(c.el, next);
    prev = c.el;
  }
}

function notice(kind, iconName, ...children) {
  return h("div", { class: `notice ${kind || ""} lab-notice` }, h("span", { class: "notice-icon", html: icon(iconName) }), h("div", { class: "lab-notice-body" }, ...children));
}

/** Count a number up from whatever the element showed last. */
function tweenNumber(el, to, format, duration = 700) {
  if (!el) return;
  if (el._stopCount) el._stopCount();
  const from = isNum(el._v) ? el._v : 0;
  if (!isNum(to)) {
    el._v = 0;
    el.textContent = "—";
    return;
  }
  el._v = to;
  el._stopCount = countUp(el, to, { from, duration: from === to ? 0 : duration, format });
}

/** A card with a numbered step header. */
function stepCard(n, title, sub, { tour = null, id, right = null } = {}) {
  const titleId = `${id}-title`;
  const card = h("section", { class: "card lab-card", id, "data-tour": tour, "aria-labelledby": titleId });
  const head = h(
    "div",
    { class: "card-head lab-card-head" },
    h(
      "div",
      { class: "lab-head-main" },
      n != null ? h("span", { class: "lab-step-num", "aria-hidden": "true" }, String(n)) : null,
      h("div", { class: "lab-head-text" }, h("h2", { class: "card-title lab-card-title", id: titleId }, title), sub ? h("p", { class: "card-sub" }, sub) : null),
    ),
    right,
  );
  card.appendChild(head);
  return card;
}

/** Segmented control: .seg > button[aria-pressed]. */
function segmented(options, value, onChange, { label, cls = "" } = {}) {
  const el = h("div", { class: `seg ${cls}`, role: "group", "aria-label": label });
  let current = value;
  const buttons = options.map((o) => {
    const b = h(
      "button",
      {
        type: "button",
        "aria-pressed": String(o.id === value),
        title: o.title || null,
        onClick: () => {
          if (b.disabled) return;
          const changed = current !== o.id;
          set(o.id);
          if (changed) onChange(o.id);
        },
      },
      ...(o.content || [o.label]),
    );
    b.dataset.value = o.id;
    el.appendChild(b);
    return b;
  });
  function set(v) {
    current = v;
    buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.value === v)));
  }
  return {
    el,
    set,
    buttons,
    get value() {
      return current;
    },
  };
}

/** Labelled range slider with a live value readout. */
function slider({ id, label, min, max, step = 1, value, format = (v) => String(v), onInput, hint }) {
  const out = h("output", { class: "lab-slider-val", for: id });
  const input = h("input", { type: "range", id, min, max, step, value });
  input.value = String(value);
  const refill = syncRangeFill(input);
  out.textContent = format(value);
  input.addEventListener("input", () => {
    const v = Number(input.value);
    out.textContent = format(v);
    onInput(v);
  });
  const hintEl = hint ? h("div", { class: "hint" }, hint) : null;
  const el = h("div", { class: "field lab-slider" }, h("div", { class: "lab-slider-top" }, h("label", { class: "label", for: id }, label), out), input, hintEl);
  return {
    el,
    input,
    hintEl,
    set(v) {
      input.value = String(v);
      refill();
      out.textContent = format(v);
    },
  };
}

/** Exact synthetic window count for a scale (mirrors generateCorpus's T per host). */
function windowCount(stats, scale) {
  let n = 0;
  for (const r of stats.replicas) {
    for (const host of stats.hosts) {
      const hs = r.hosts?.[host.id];
      if (hs && hs.n_windows > 0) n += Math.max(4, Math.round(hs.n_windows * scale));
    }
  }
  return n;
}

function sameSet(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  const s = new Set(a);
  return b.every((x) => s.has(x));
}

/** True when a split config is exactly the project's default protocol. */
function isDefaultProtocol(split) {
  return !!split && split.mode === "replica" && sameSet(split.test, ["santos"]) && sameSet(split.train, DEFAULT_SPLIT.train);
}

function testReplicasOf(split, ids = REPLICA_ORDER) {
  if (!split) return [];
  if (split.mode === "replica") return split.test || [];
  if (split.mode === "loro") return split.replicas || ids;
  return split.replicas || [];
}

/** "recorded by tests/x.py::test_y" out of an exporter source string (never invented). */
function recordedBy(source) {
  const m = /(tests\/[\w./-]+::[\w]+)/.exec(String(source || ""));
  return m ? m[1] : null;
}

/**
 * "pooled over 7 scoreable folds (7 models)" for a leave-one-replica-out
 * result, whose result.metrics pool the test windows of the folds that are not
 * low-support (result.metricsScope); null for a single split.
 */
export function pooledLabel(result) {
  if (!result || result.mode !== "loro") return null;
  const sc = result.metricsScope;
  if (!sc || sc.kind !== "pooled-folds") return "pooled over the scoreable folds";
  return `pooled over ${sc.nFolds} scoreable fold${sc.nFolds === 1 ? "" : "s"} (${sc.nModels} model${sc.nModels === 1 ? "" : "s"})`;
}

const scoredCache = new WeakMap();
/**
 * The test rows result.metrics counts: every row for a single split, only the
 * rows with test.included = 1 (the scoreable folds) in leave-one-replica-out.
 * -> {y, scores, n}.
 */
export function scoredRows(result) {
  const t = result?.test;
  if (!t || !t.y || !t.scores) return null;
  if (result.mode !== "loro" || !t.included) return { y: t.y, scores: t.scores, n: t.n ?? t.y.length };
  if (scoredCache.has(result)) return scoredCache.get(result);
  let n = 0;
  for (let i = 0; i < t.included.length; i++) if (t.included[i]) n++;
  const y = new Uint8Array(n);
  const scores = new Float32Array(n);
  for (let i = 0, r = 0; i < t.included.length; i++) {
    if (!t.included[i]) continue;
    y[r] = t.y[i];
    scores[r] = t.scores[i];
    r++;
  }
  const out = { y, scores, n };
  scoredCache.set(result, out);
  return out;
}

/**
 * A JSON-safe copy: typed arrays become plain arrays and every non-finite
 * number (JSON has no Infinity or NaN, e.g. the ROC curve's first threshold is
 * +Infinity by convention) becomes the string "Infinity", "-Infinity" or "NaN".
 * `found` collects the paths of those values so the file can say so.
 */
function toPlain(value, path = "", found = null) {
  if (typeof value === "number") {
    if (Number.isFinite(value)) return value;
    if (found) found.push(path || "(value)");
    return Number.isNaN(value) ? "NaN" : value > 0 ? "Infinity" : "-Infinity";
  }
  if (value == null || typeof value !== "object") return value;
  if (ArrayBuffer.isView(value) || Array.isArray(value)) return Array.from(value, (v, i) => toPlain(v, `${path}[${i}]`, found));
  const out = {};
  for (const [k, v] of Object.entries(value)) out[k] = toPlain(v, path ? `${path}.${k}` : k, found);
  return out;
}

/* ================================================================== UI config */

function defaultUi() {
  return {
    scale: 0.05,
    dataSeed: 0,
    strategy: "default",
    roles: Object.fromEntries(REPLICA_ORDER.map((r) => [r, r === "santos" ? "test" : "train"])),
    include: Object.fromEntries(REPLICA_ORDER.map((r) => [r, true])),
    valPct: 15,
    trainPct: 70,
    holdValPct: 15,
    loroPreview: "santos",
    model: { kind: "tessera", epochs: 30, lr: 1e-3, batchSize: 256, patience: 5, seed: 0 },
    features: "all",
  };
}

function loadUi() {
  const ui = defaultUi();
  const p = prefs.get(PREFS_KEY, null);
  if (!p || typeof p !== "object") return ui;
  const intIn = (v, lo, hi) => Number.isInteger(v) && v >= lo && v <= hi;
  if (SCALES.some((s) => s.scale === p.scale)) ui.scale = p.scale;
  if (intIn(p.dataSeed, 0, 99999)) ui.dataSeed = p.dataSeed;
  if (STRATEGY[p.strategy]) ui.strategy = p.strategy;
  for (const r of REPLICA_ORDER) {
    if (["train", "test", "off"].includes(p.roles?.[r])) ui.roles[r] = p.roles[r];
    if (typeof p.include?.[r] === "boolean") ui.include[r] = p.include[r];
  }
  if (intIn(p.valPct, 5, 40)) ui.valPct = p.valPct;
  if (intIn(p.trainPct, 20, 90) && intIn(p.holdValPct, 5, 40) && p.trainPct + p.holdValPct <= 95) {
    ui.trainPct = p.trainPct;
    ui.holdValPct = p.holdValPct;
  }
  if (REPLICA_ORDER.includes(p.loroPreview)) ui.loroPreview = p.loroPreview;
  const m = p.model || {};
  if (MODEL[m.kind]) ui.model.kind = m.kind;
  if (intIn(m.epochs, 5, 40)) ui.model.epochs = m.epochs;
  if (LR_OPTIONS.includes(m.lr)) ui.model.lr = m.lr;
  if (BATCH_OPTIONS.includes(m.batchSize)) ui.model.batchSize = m.batchSize;
  if (intIn(m.patience, 3, 10)) ui.model.patience = m.patience;
  if (intIn(m.seed, 0, 99999)) ui.model.seed = m.seed;
  if (FEATURE[p.features]) ui.features = p.features;
  return ui;
}

/** The feature set a run really uses: the pretrained model always sees all 42 features. */
function effectiveFeatures(ui) {
  return ui.model.kind === "pretrained" ? "all" : ui.features;
}

function splitConfigFor(ui, ids) {
  const valFraction = ui.valPct / 100;
  switch (ui.strategy) {
    case "pick":
      return {
        mode: "replica",
        train: ids.filter((r) => ui.roles[r] === "train"),
        test: ids.filter((r) => ui.roles[r] === "test"),
        valFraction,
        seed: 0,
      };
    case "random":
      return {
        mode: "random",
        replicas: ids.filter((r) => ui.include[r]),
        trainPct: ui.trainPct,
        valPct: ui.holdValPct,
        testPct: 100 - ui.trainPct - ui.holdValPct,
        seed: 0,
      };
    case "chrono":
      return {
        mode: "chronological",
        replicas: ids.filter((r) => ui.include[r]),
        trainPct: ui.trainPct,
        valPct: ui.holdValPct,
        testPct: 100 - ui.trainPct - ui.holdValPct,
      };
    case "loro":
      return { mode: "loro", replicas: ids.slice(), valFraction, seed: 0 };
    default:
      return { ...cloneSplitConfig(DEFAULT_SPLIT), valFraction };
  }
}

function pipelineConfigFor(ui, ids) {
  return {
    dataset: { scale: ui.scale, seed: ui.dataSeed },
    split: splitConfigFor(ui, ids),
    model: { ...ui.model, ...(ui.model.kind === "knn" ? { k: KNN_DEFAULT_K } : {}) },
    features: effectiveFeatures(ui),
    threshold: 0.5,
    ledger: true,
  };
}

/** Rough run-time estimate (measured: about 15 µs per training row per epoch). */
function estimateMs(ui, preview) {
  if (!preview || !preview.built) return null;
  const K = preview.folds ? preview.folds.length : 1;
  const trainRows = preview.built.summary.n.train;
  const base = 400;
  if (ui.model.kind === "pretrained") return base + preview.corpus.n * 0.02;
  if (ui.model.kind === "logreg") return base + K * (trainRows * LOGREG_STEPS * 0.00022 + 60);
  if (ui.model.kind === "knn") {
    const d = effectiveFeatures(ui) === "m2" ? 24 : 42;
    return base + K * (trainRows * preview.built.summary.n.test * d * 6e-7 + 60);
  }
  return base + K * (trainRows * ui.model.epochs * 0.018 + 80);
}

function trainTestText(split) {
  if (!split) return "—";
  if (split.mode === "replica") return `${plainIds(split.train)} → ${split.test.join(", ")}`;
  if (split.mode === "loro") return `each of ${split.replicas.length} in turn`;
  const how = split.mode === "random" ? "shuffled" : "by time";
  return `${plainIds(split.replicas)}, ${how} ${split.trainPct}/${split.valPct}/${split.testPct}`;
}

/* ================================================================== mount */

export async function mount(el, ctx) {
  const toast = ctx?.toast || domToast;
  const store = ctx.store;
  el.replaceChildren(h("div", { class: "lab-loading" }, h("div", { class: "skeleton", style: { height: "120px" } }), h("div", { class: "skeleton", style: { height: "320px" } })));

  let stats;
  try {
    stats = await store.loadJSON("data/replica_stats.json");
  } catch (err) {
    el.replaceChildren(h("div", { class: "mount-error" }, `The Training Lab could not load its data statistics (replica_stats.json): ${err.message}.`));
    return null;
  }
  let realResults = null;
  store
    .loadJSON("data/real_results.json")
    .then((r) => {
      realResults = r;
      if (S.lastShown) renderResults(S.lastShown.result, S.lastShown.entry, { reveal: false });
      renderHistory();
      renderExpReal();
    })
    .catch(() => {
      realResults = null;
    });

  const replicaIds = stats.replicas.map((r) => r.id);
  const replicaStats = Object.fromEntries(stats.replicas.map((r) => [r.id, r]));

  /* ---------------------------------------------------------------- state */
  const S = {
    ui: loadUi(),
    corpusCache: new Map(),
    preview: null,
    run: null,
    runCounter: 0,
    history: [],
    results: new Map(), // k -> result (the last KEEP_RESULTS runs)
    lastShown: null,
    threshold: 0.5,
    worker: null,
    workerReady: false,
    workerFailed: false,
    pendingPost: null,
    // split experiment: results per model kind ('knn' | 'tessera'), each {scale, seed, runs: {random, chrono}}
    exp: { active: false, modelKind: "knn", cancelRequested: false, pending: null, results: {}, shownKind: null, startedAt: 0 },
    freshK: null, // the history row to fade in on the next renderHistory (the newest run only)
    previewCount: 0,
  };
  const ui = S.ui;
  const refs = {};
  const resultCharts = [];
  // timers used by hoisted helpers below (declared here, before the return, to avoid the TDZ)
  let previewTimer = 0;
  let thrRaf = 0;
  let thrPending = null;

  const saveUi = debounce(() => prefs.set(PREFS_KEY, ui), 300);
  // one screen-reader sentence per settled preview (final values only, never mid-animation)
  const announceSplit = debounce(() => {
    if (refs.splitStatus) refs.splitStatus.textContent = splitSentence(S.preview);
  }, 700);

  function getCorpus(scale, seed) {
    const key = `${scale}|${seed}`;
    let c = S.corpusCache.get(key);
    if (!c) {
      c = generateCorpus(stats, { scale, seed });
      c.summary = corpusSummary(c);
      if (S.corpusCache.size >= 4) S.corpusCache.delete(S.corpusCache.keys().next().value);
      S.corpusCache.set(key, c);
    }
    return c;
  }

  /**
   * Scroll a Lab card into view (html's scroll-padding-top keeps it clear of the
   * sticky top bar) and move keyboard focus to it, so focus never stays on a
   * control that was just replaced or scrolled away. Nothing happens while the
   * Lab tab is hidden.
   */
  function scrollToCard(id, { focus = true } = {}) {
    if (ctx.isActive && !ctx.isActive("lab")) return;
    const card = document.getElementById(id);
    if (!card) return;
    card.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
    if (focus) {
      if (!card.hasAttribute("tabindex")) card.setAttribute("tabindex", "-1");
      try {
        card.focus({ preventScroll: true });
      } catch {
        card.focus();
      }
    }
  }

  /** Double-click guard: true while a just-started run (or experiment) should ignore Cancel. */
  function cancelArming() {
    const run = S.run;
    if (!run || !run.active) return 0;
    const t0 = run.exp ? S.exp.startedAt || run.t0 : run.t0;
    return Math.max(0, CANCEL_ARM_MS - (performance.now() - t0));
  }

  /**
   * Show a Cancel button as not yet clickable for `ms` (aria-disabled, not the
   * disabled attribute, so keyboard focus stays on it), then re-render.
   */
  function setArming(btn, ms) {
    clearTimeout(btn._armTimer);
    if (ms > 0) {
      btn.setAttribute("aria-disabled", "true");
      btn.classList.add("is-arming");
      btn._armTimer = setTimeout(() => updateRunButton(), ms + 20);
    } else {
      btn.removeAttribute("aria-disabled");
      btn.classList.remove("is-arming");
    }
  }

  /* ---------------------------------------------------------------- layout */
  const root = h("div", { class: "lab" });
  const cards = [buildHead(), buildExperimentCard(), buildDataCard(), buildSplitCard(), buildModelCard(), buildRunBar(), buildProgressCard(), buildResultsCard(), buildHistoryCard()];
  root.append(...cards);
  el.replaceChildren(root);
  stagger(root);

  syncControlsFromUi();
  updateCorpus();
  computePreview();
  setRunState("idle");

  /* keyboard shortcut: Ctrl/Cmd + Enter runs (or cancels) */
  const onKey = (e) => {
    if (!(e.ctrlKey || e.metaKey) || e.key !== "Enter") return;
    if (ctx.isActive && !ctx.isActive("lab")) return;
    e.preventDefault();
    if (S.run && S.run.active) cancelRun();
    else startRun();
  };
  document.addEventListener("keydown", onKey);

  // start the background worker early so the first run does not wait for it
  setTimeout(() => ensureWorker(), 800);

  return {
    onShow() {
      // charts re-measure through their ResizeObserver; nudge anything that mounted hidden
      window.dispatchEvent(new Event("resize"));
    },
    onHide() {
      hideTooltip();
    },
  };

  /* ================================================================ head */

  function buildHead() {
    const head = h(
      "header",
      { class: "page-head lab-head" },
      h(
        "div",
        { class: "lab-head-copy" },
        h("div", { class: "eyebrow" }, "Training Lab"),
        h("h1", {}, "Train and test the detector yourself"),
        h(
          "p",
          { class: "lede" },
          "Choose the data, decide how it is split into training and testing, pick a model, and watch every stage run in your browser. The leakage check updates as you go, so you can see why the way a model is tested matters as much as the model.",
        ),
      ),
    );
    const honesty = notice(
      "warn",
      "info",
      h("span", { class: "tag-synthetic" }, "Synthetic data"),
      " ",
      h(
        "span",
        {},
        "Everything here runs on synthetic windows generated from summary statistics of the 8 real AIT replicas (no real rows are shipped). The real held-out numbers are in ",
        h("a", { href: "#results" }, "Real Results"),
        ".",
      ),
    );
    return h("div", { class: "lab-top" }, head, honesty);
  }

  /* ================================================================ experiment */

  /** The experiment's fixed setup: all 8 replicas, 70/15/15, network metrics only, the Lab's data size and seed. */
  function experimentConfig(kind, mode) {
    const pcts = { trainPct: 70, valPct: 15, testPct: 15 };
    return {
      dataset: { scale: ui.scale, seed: ui.dataSeed },
      split: mode === "random" ? { mode: "random", replicas: replicaIds.slice(), ...pcts, seed: 0 } : { mode: "chronological", replicas: replicaIds.slice(), ...pcts },
      model: kind === "knn" ? { kind: "knn", k: KNN_DEFAULT_K, seed: 0 } : { kind: "tessera", epochs: 30, lr: 1e-3, batchSize: 256, patience: 5, seed: 0 },
      features: "m2",
      threshold: 0.5,
      ledger: false,
    };
  }

  function buildExperimentCard() {
    const titleId = "lab-experiment-title";
    const card = h("section", { class: "card lab-card lab-exp", id: "lab-experiment", "data-tour": "lab-experiment", "aria-labelledby": titleId });
    const head = h(
      "div",
      { class: "card-head lab-card-head lab-exp-head" },
      h(
        "div",
        { class: "lab-head-main" },
        h("span", { class: "lab-exp-badge", "aria-hidden": "true", html: icon("flask") }),
        h(
          "div",
          { class: "lab-head-text" },
          h("div", { class: "eyebrow lab-exp-eyebrow" }, "Experiment"),
          h("h2", { class: "card-title lab-card-title", id: titleId }, "Does the way you split change the score?"),
        ),
      ),
      h("span", { class: "tag-synthetic" }, "Synthetic data"),
    );
    const lede = h(
      "p",
      { class: "lab-exp-lede" },
      "A random split deals every window out by chance, so near-copies of a test window, such as the minute just before or after it, sit in training. A chronological split trains on the past and tests on the future, the way a deployed detector meets new traffic. Run both on the same windows with a model that can memorise, and see how far the scores move apart.",
    );
    refs.expChips = h("div", { class: "lab-exp-chips", "aria-label": "Fixed setup of the experiment" });

    refs.expRunBtn = h("button", {
      type: "button",
      class: "btn primary lab-exp-go",
      onClick: (e) => {
        // the second click of a double-click must not cancel what the first one started
        if (e && e.detail > 1) return;
        if (S.exp.active) cancelRun();
        else startExperiment("knn");
      },
    });
    refs.expTesseraBtn = h("button", {
      type: "button",
      class: "btn",
      title: "Re-run the experiment with the neural model, which cannot store the training windows",
      html: `${icon("cpu")}<span>Try it with TESSERA-base</span>`,
      onClick: (e) => {
        if (e && e.detail > 1) return;
        startExperiment("tessera");
      },
    });
    refs.expLoadBtn = h("button", {
      type: "button",
      class: "btn ghost",
      title: "Set steps 1 to 3 to the random-split setup, so you can inspect its leakage certificate",
      html: `${icon("arrowRight")}<span>Load the random-split setup into the Lab</span>`,
      onClick: loadRandomSetup,
    });
    refs.expHint = h("span", { class: "tiny faint lab-exp-hint" });
    const actions = h("div", { class: "lab-exp-actions" }, refs.expRunBtn, refs.expTesseraBtn, refs.expLoadBtn, refs.expHint);

    refs.expRuns = {};
    const runsBox = h("div", { class: "lab-exp-runs", hidden: true });
    refs.expStatus = h("p", { class: "sr-only", role: "status", "aria-live": "polite" });
    for (const sp of EXP_SPLITS) {
      const st = STRATEGY[sp.strategy];
      const fill = h("div", { class: "progress-fill" });
      const bar = h("div", { class: "progress lab-exp-progress", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": "0", "aria-label": `${sp.label} progress` }, fill);
      const pct = h("span", { class: "lab-exp-run-pct tabular" }, "0%");
      const detail = h("div", { class: "lab-exp-run-detail tiny" }, "Waiting");
      const row = h(
        "div",
        { class: "lab-exp-run", "data-state": "pending" },
        h("div", { class: "lab-exp-run-head" }, h("span", { class: "lab-exp-run-name" }, sp.label), pill(st.honesty.cls, st.honesty.icon, st.honesty.text), pct),
        bar,
        detail,
      );
      refs.expRuns[sp.id] = { row, fill, bar, pct, detail, shown: 0 };
      runsBox.appendChild(row);
    }
    refs.expRunsBox = runsBox;
    refs.expError = h("div", { hidden: true });
    refs.expResult = h("div", { class: "lab-exp-result", hidden: true });
    refs.expRealHost = h("div", { class: "lab-exp-panel lab-exp-real" });

    card.append(head, lede, refs.expChips, actions, runsBox, refs.expStatus, refs.expError, refs.expResult);
    renderExpChips();
    return card;
  }

  function renderExpChips() {
    if (!refs.expChips) return;
    const sc = SCALES.find((x) => x.scale === ui.scale);
    const kind = S.exp.active ? S.exp.modelKind : "knn";
    syncChips(refs.expChips, [
      { key: "reps", icon: "database", text: `All ${replicaIds.length} replicas` },
      { key: "feat", icon: "layers", text: "Network metrics only (24)" },
      { key: "model", icon: "cpu", text: EXP_MODELS[kind].chip },
      { key: "pcts", icon: "split", text: "70 / 15 / 15" },
      { key: "data", icon: "flask", text: `${sc ? `${sc.label} data (${sc.pct})` : `scale ${ui.scale}`} · seed ${ui.dataSeed}` },
    ]);
  }

  function updateExpButtons() {
    if (!refs.expRunBtn) return;
    const exp = S.exp;
    const mainBusy = !!(S.run && S.run.active && !S.run.exp);
    const cancelling = exp.active && S.run && S.run.cancelling;
    if (exp.active) {
      const arming = cancelArming();
      refs.expRunBtn.classList.add("is-cancel");
      refs.expRunBtn.innerHTML = `${icon("stop")}<span>${cancelling ? "Cancelling…" : "Cancel the experiment"}</span>`;
      refs.expRunBtn.disabled = !!cancelling;
      setArming(refs.expRunBtn, arming);
    } else {
      refs.expRunBtn.classList.remove("is-cancel");
      refs.expRunBtn.innerHTML = `${icon("play")}<span>Run the experiment</span>`;
      refs.expRunBtn.disabled = mainBusy;
      setArming(refs.expRunBtn, 0);
    }
    refs.expTesseraBtn.disabled = exp.active || mainBusy;
    refs.expLoadBtn.disabled = exp.active || mainBusy;
    refs.expHint.textContent = mainBusy
      ? "A training run is in progress: the experiment can start when it finishes."
      : exp.active
        ? ""
        : "Two runs back to back: about a second with the memoriser, longer with TESSERA-base.";
  }

  function startExperiment(kind) {
    if (S.run && S.run.active) {
      toast(S.run.exp ? "The experiment is already running." : "Wait for the current run to finish (or cancel it) first.", { kind: "warn" });
      return;
    }
    const exp = S.exp;
    exp.active = true;
    exp.startedAt = performance.now();
    exp.modelKind = kind;
    exp.cancelRequested = false;
    exp.pending = { kind, scale: ui.scale, seed: ui.dataSeed, runs: {} };
    refs.expError.hidden = true;
    refs.expError.replaceChildren();
    refs.expResult.classList.add("is-stale");
    refs.expRunsBox.hidden = false;
    for (const sp of EXP_SPLITS) setExpRow(sp.id, "pending", 0, "Waiting");
    renderExpChips();
    runExpStep(0);
  }

  function runExpStep(i) {
    const sp = EXP_SPLITS[i];
    const kind = S.exp.modelKind;
    const k = ++S.runCounter;
    const run = {
      id: `lab-exp-${k}-${Date.now().toString(36)}`,
      k,
      active: true,
      cancelling: false,
      config: experimentConfig(kind, sp.mode),
      strategy: sp.strategy,
      t0: performance.now(),
      exp: { index: i, split: sp, label: `${EXP_MODELS[kind].label}, ${sp.short} split` },
      loss: [],
      ap: [],
      foldsDone: [],
    };
    S.run = run;
    setExpRow(sp.id, "running", 0, "Starting…");
    refs.expStatus.textContent = `Running the ${sp.short} split (${i + 1} of ${EXP_SPLITS.length}) with the ${EXP_MODELS[kind].name}…`;
    updateRunButton();
    dispatchRun(run);
  }

  function setExpRow(id, state, overall, detail) {
    const r = refs.expRuns[id];
    if (!r) return;
    r.row.dataset.state = state;
    r.bar.classList.toggle("is-running", state === "running");
    r.bar.classList.toggle("is-done", state === "done");
    r.bar.classList.toggle("is-error", state === "error");
    r.bar.classList.toggle("is-skipped", state === "skipped");
    if (isNum(overall)) {
      const v = clampNum(state === "done" ? 1 : overall, 0, 1);
      r.fill.style.width = `${v * 100}%`;
      const p = Math.floor(v * 100);
      if (p !== r.shown || state !== "running") {
        r.shown = p;
        r.pct.textContent = state === "skipped" ? "Not run" : state === "error" ? "Error" : `${p}%`;
        r.bar.setAttribute("aria-valuenow", String(p));
      }
    }
    if (detail != null) r.detail.textContent = detail;
  }

  function expEvent(run, ev) {
    const id = run.exp.split.id;
    switch (ev.type) {
      case "progress":
        setExpRow(id, "running", ev.overall, null);
        break;
      case "stage":
        if (ev.status === "running" || ev.status === "done" || ev.status === "skipped") {
          const label = STEPS.find((x) => x.id === ev.stage)?.label || ev.stage;
          if (ev.detail && !(ev.status === "skipped" && ev.stage === "ledger")) setExpRow(id, "running", null, `${label}: ${ev.detail}`);
        }
        break;
      case "result":
        if (ev.result && ev.result.cancelled) expFinish(run, "cancelled");
        else expFinish(run, "done", ev.result);
        break;
      case "error":
        expFinish(run, "error", null, ev.message);
        break;
      default:
        break;
    }
  }

  function expCancelling(run) {
    setExpRow(run.exp.split.id, "running", null, "Cancelling…");
  }

  function expFinish(run, status, result = null, message = "") {
    if (!run.active) return;
    run.active = false;
    clearTimeout(run.cancelTimer);
    const exp = S.exp;
    const sp = run.exp.split;
    if (status === "done" && result) {
      const entry = addHistory(run, result);
      exp.pending.runs[sp.id] = {
        k: entry.k,
        ap: result.metrics.ap,
        prevalence: result.metrics.prevalence,
        n: result.metrics.n,
        nPositive: result.metrics.nPositive,
        cert: result.certificate
          ? { overall: result.certificate.overall, activeDuplicates: result.certificate.activeDuplicates, stretchOverlap: result.certificate.stretchOverlap, attackEpisodeOverlap: result.certificate.attackEpisodeOverlap }
          : null,
        elapsedMs: result.elapsedMs,
      };
      setExpRow(sp.id, "done", 1, `Done in ${fmt.ms(result.elapsedMs)} · AP ${f3(result.metrics.ap)} on ${plural(result.metrics.nPositive, "attack window")} (no-skill ${f3(result.metrics.prevalence)})`);
      refs.expStatus.textContent = `${sp.label} done: AP ${f3(result.metrics.ap)}.`;
      renderHistory();
      if (run.exp.index + 1 < EXP_SPLITS.length && !exp.cancelRequested) {
        runExpStep(run.exp.index + 1);
        return;
      }
      exp.active = false;
      if (exp.cancelRequested) {
        expStopped("cancelled", run.exp.index + 1);
      } else {
        exp.results[exp.pending.kind] = exp.pending;
        renderExpResult(exp.pending.kind, { reveal: true });
        const pr = exp.pending.runs;
        refs.expStatus.textContent = `Experiment finished: random split AP ${f3(pr.random.ap)}, chronological split AP ${f3(pr.chrono.ap)}, a difference of ${signedPts(pr.random.ap - pr.chrono.ap)}.`;
        toast("Experiment finished. Both runs are in the run history.", { kind: "good" });
      }
    } else {
      exp.active = false;
      if (status === "error") {
        setExpRow(sp.id, "error", null, message || "Unknown error.");
        refs.expError.hidden = false;
        refs.expError.replaceChildren(notice("bad", "alert", h("strong", {}, "The experiment stopped with an error. "), message || "Unknown error."));
        expStopped("error", run.exp.index + 1);
      } else {
        setExpRow(sp.id, "skipped", null, "Cancelled");
        expStopped("cancelled", run.exp.index + 1);
      }
    }
    updateRunButton();
    renderExpChips();
  }

  /** Close the rows that did not run, and restore the previous result (if any). */
  function expStopped(status, from) {
    for (let i = from; i < EXP_SPLITS.length; i++) setExpRow(EXP_SPLITS[i].id, "skipped", 0, status === "cancelled" ? "Not run (cancelled)" : "Not run");
    refs.expResult.classList.remove("is-stale");
    if (status === "cancelled") toast("Experiment cancelled. Finished runs stay in the run history.", { kind: "warn" });
    refs.expStatus.textContent = status === "cancelled" ? "Experiment cancelled." : "The experiment stopped with an error.";
    S.exp.active = false;
    updateRunButton();
    renderExpChips();
  }

  function loadRandomSetup() {
    if (S.run && S.run.active) return;
    const kind = S.exp.shownKind || "knn";
    ui.strategy = "random";
    for (const r of REPLICA_ORDER) ui.include[r] = true;
    ui.trainPct = 70;
    ui.holdValPct = 15;
    ui.model.kind = kind;
    if (kind === "tessera") Object.assign(ui.model, { epochs: 30, lr: 1e-3, batchSize: 256, patience: 5, seed: 0 });
    ui.features = "m2";
    syncControlsFromUi();
    onStrategyChange();
    saveUi();
    scrollToCard("lab-split");
    toast(`Random-split setup loaded (${EXP_MODELS[kind].label}, network metrics only). The leakage certificate in step 2 shows what leaks.`, { kind: "info", timeout: 5200 });
  }

  /**
   * Horizontal AP bars, 0..1, each with its no-skill baseline (the test
   * prevalence) as a tick. rows: [{label, ap, base?, color, tag?}].
   */
  function compareBars(rows, { ariaLabel = "Average precision by split" } = {}) {
    const wrap = h("div", { class: "lab-cmp", role: "group", "aria-label": ariaLabel });
    const fills = [];
    for (const r of rows) {
      const fill = h("div", { class: "lab-cmp-fill", style: { background: `var(${r.color || "--series-1"})` } });
      const hasBase = isNum(r.base);
      const base = hasBase ? h("div", { class: "lab-cmp-base", style: { left: `${clampNum(r.base, 0, 1) * 100}%` }, "aria-hidden": "true" }) : null;
      wrap.appendChild(
        h(
          "div",
          { class: "lab-cmp-row" },
          h("div", { class: "lab-cmp-label" }, h("span", {}, r.label), r.tag || null),
          h("div", { class: "lab-cmp-track" }, fill, base),
          h("div", { class: "lab-cmp-val tabular" }, h("span", { class: "sr-only" }, "AP "), f3(r.ap)),
          h("div", { class: "lab-cmp-sub tiny" }, hasBase ? `no-skill baseline (attack share of the test set) ${f3(r.base)}` : r.sub || ""),
        ),
      );
      fills.push([fill, isNum(r.ap) ? clampNum(r.ap, 0, 1) : 0]);
    }
    wrap.appendChild(
      h(
        "div",
        { class: "lab-cmp-row lab-cmp-axis-row", "aria-hidden": "true" },
        h("div", { class: "lab-cmp-label" }),
        h("div", { class: "lab-cmp-axis" }, ...[0, 0.25, 0.5, 0.75, 1].map((v) => h("span", { style: { left: `${v * 100}%` } }, v === 0 || v === 1 ? String(v) : v.toFixed(2)))),
      ),
    );
    const grow = () => fills.forEach(([f, v]) => (f.style.width = `${v * 100}%`));
    if (reducedMotion()) grow();
    else requestAnimationFrame(() => requestAnimationFrame(grow));
    return wrap;
  }

  function expTakeaway(res) {
    const R = res.runs.random;
    const C = res.runs.chrono;
    const name = EXP_MODELS[res.kind].name;
    const gap = R.ap - C.ap;
    const lift = R.ap - R.prevalence - (C.ap - C.prevalence);
    const baseNote =
      R.prevalence - C.prevalence > 0.005
        ? ` Part of the gap is the baseline: the random test set holds more attack windows (no-skill ${f3(R.prevalence)} against ${f3(C.prevalence)}). Measured above each split's own no-skill line, the random split is ${lift >= 0.005 ? `still ${signedPts(lift).replace("+", "")} ahead` : lift <= -0.005 ? "not ahead at all" : "level"}.`
        : C.prevalence - R.prevalence > 0.005
          ? ` The chronological test set holds more attack windows (no-skill ${f3(C.prevalence)} against ${f3(R.prevalence)}), which makes it the easier baseline, not the harder one.`
          : "";
    const why =
      R.cert && (R.cert.activeDuplicates > 0 || R.cert.stretchOverlap > 0)
        ? ` The random split's leakage certificate shows the mechanism: ${plural(R.cert.activeDuplicates || 0, "test window")} ${R.cert.activeDuplicates === 1 ? "is an exact copy" : "are exact copies"} of a training window with real network activity, and ${plural(R.cert.stretchOverlap || 0, "attack or quiet period")} ${R.cert.stretchOverlap === 1 ? "is" : "are"} cut in two.`
        : "";
    if (gap >= 0.02) {
      return `The same ${name}, on the same synthetic windows, scores ${signedPts(gap).replace("+", "")} higher when the test windows are dealt out at random than when it is tested on the future.${baseNote}${why}`;
    }
    if (gap > -0.02) {
      return `Here the two splits score about the same (${signedPts(gap)} for the random split), so this run shows no clear inflation and we do not claim one.${baseNote}${why ? `${why} The leak is there even though the score barely moved.` : ""}`;
    }
    return `Here the chronological split scored higher, by ${signedPts(-gap).replace("+", "")}: the opposite of what leakage usually does. We report it as measured rather than hide it; try another data seed or size to see how much this varies.${baseNote}`;
  }

  function renderExpResult(kind, { reveal = false } = {}) {
    const res = S.exp.results[kind];
    if (!res || !refs.expResult) return;
    S.exp.shownKind = kind;
    const R = res.runs.random;
    const C = res.runs.chrono;
    const sc = SCALES.find((x) => x.scale === res.scale);
    const gapNum = h("div", { class: "lab-exp-gap-num tabular" }, "+0.0");
    const gap = R.ap - C.ap;
    const mine = h(
      "div",
      { class: "lab-exp-panel" },
      h(
        "div",
        { class: "lab-sub-title" },
        h("span", { class: "tag-synthetic" }, "Synthetic"),
        `${EXP_MODELS[kind].label} in your browser`,
        h("span", { class: "faint tiny lab-sub-note" }, `${sc ? `${sc.label} data` : `scale ${res.scale}`}, seed ${res.seed} · runs #${R.k} and #${C.k}`),
      ),
      compareBars(
        [
          { label: "Random split", ap: R.ap, base: R.prevalence, color: "--series-2", tag: pill("bad", "alert", "Leaky") },
          { label: "Chronological split", ap: C.ap, base: C.prevalence, color: "--series-1" },
        ],
        { ariaLabel: `Average precision of the ${EXP_MODELS[kind].label}: random split ${f3(R.ap)}, chronological split ${f3(C.ap)}` },
      ),
      h("div", { class: "lab-exp-gap" }, gapNum, h("div", { class: "lab-exp-gap-label small" }, gap >= 0 ? "AP points the random split gains over the chronological one" : "AP points the random split loses against the chronological one")),
      h("p", { class: "lab-takeaway" }, iconSpan("sparkles"), h("span", {}, expTakeaway(res))),
    );
    const cmp = modelComparison();
    renderExpReal();
    const body = h("div", { class: "grid-2 lab-exp-grid" }, mine, refs.expRealHost);
    refs.expResult.replaceChildren(...[body, cmp].filter(Boolean));
    refs.expResult.hidden = false;
    refs.expResult.classList.remove("is-stale");
    if (reveal && !reducedMotion()) {
      refs.expResult.classList.remove("reveal");
      void refs.expResult.offsetWidth;
      refs.expResult.classList.add("reveal");
    }
    const pts = gap * 100;
    const format = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}`;
    gapNum.textContent = format(reveal ? 0 : pts);
    if (reveal) requestAnimationFrame(() => countUp(gapNum, pts, { from: 0, duration: 1100, format }));
    else gapNum.textContent = format(pts);
  }

  /** Memoriser vs TESSERA-base, when both were run on the same data size and seed. */
  function modelComparison() {
    const a = S.exp.results.knn;
    const b = S.exp.results.tessera;
    if (!a || !b || a.scale !== b.scale || a.seed !== b.seed) return null;
    const ga = a.runs.random.ap - a.runs.chrono.ap;
    const gb = b.runs.random.ap - b.runs.chrono.ap;
    const text =
      gb < ga - 0.02
        ? `The memoriser gains ${signedPts(ga)} from the random split; TESSERA-base, whose 5,005 parameters cannot store the training windows, gains ${signedPts(gb)}. A model that can memorise is flattered more by the leak. (Real Finding 1 used LightGBM, a tree ensemble that can also memorise.)`
        : gb > ga + 0.02
          ? `Here TESSERA-base gains more from the random split (${signedPts(gb)}) than the memoriser (${signedPts(ga)}), so memorising is not the whole story on this data.`
          : `Both models gain about the same from the random split: the memoriser ${signedPts(ga)}, TESSERA-base ${signedPts(gb)}.`;
    return h("p", { class: "lab-takeaway lab-exp-models" }, iconSpan("cpu"), h("span", {}, text));
  }

  /** The real recorded Finding 1, from data/real_results.json (never invented). */
  function renderExpReal() {
    const host = refs.expRealHost;
    if (!host) return;
    const ld = realResults?.leakage_duplicates;
    if (!ld || !isNum(ld.r0_random_ap) || !isNum(ld.r1_chronological_ap)) {
      host.replaceChildren(h("div", { class: "lab-sub-title" }, h("span", { class: "tag-real" }, "Real"), "Recorded result"), h("p", { class: "muted small" }, realResults ? "The recorded real result is not in real_results.json." : "Loading the recorded real result…"));
      return;
    }
    const drop = ld.r0_random_ap - ld.r1_chronological_ap;
    const by = recordedBy(ld.source);
    const lgbm = /LightGBM/.test(ld.source || "");
    host.replaceChildren(
      ...[
      h("div", { class: "lab-sub-title" }, h("span", { class: "tag-real" }, "Real"), "Real Finding 1, recorded", h("span", { class: "faint tiny lab-sub-note" }, `${ld.replica} replica · ${lgbm ? "LightGBM" : "recorded model"}`)),
      compareBars(
        [
          { label: "Random split", ap: ld.r0_random_ap, color: "--series-2", tag: pill("bad", "alert", "Leaky"), sub: "real windows" },
          { label: "Chronological split", ap: ld.r1_chronological_ap, color: "--series-1", sub: "real windows, 600 s gap" },
        ],
        { ariaLabel: `Real recorded average precision: random split ${f3(ld.r0_random_ap)}, chronological split ${f3(ld.r1_chronological_ap)}` },
      ),
      h("div", { class: "lab-exp-gap" }, h("div", { class: "lab-exp-gap-num tabular" }, `${drop >= 0 ? "+" : "−"}${Math.abs(drop * 100).toFixed(1)}`), h("div", { class: "lab-exp-gap-label small" }, "AP points the random split gained on real data")),
      h(
        "p",
        { class: "small muted lab-exp-real-note" },
        `${ld.features}, ${fmt.int(ld.n_windows)} real windows. ${isNum(ld.test_rows_identical_to_train) && isNum(ld.n_test_rows) ? `${fmt.int(ld.test_rows_identical_to_train)} of the ${fmt.int(ld.n_test_rows)} random-split test windows were identical to a training window.` : ""}`,
      ),
      by ? h("span", { class: "tiny faint lab-prov", title: ld.source }, `${ld.provenance ? `${ld.provenance[0].toUpperCase()}${ld.provenance.slice(1)} · ` : ""}recorded by ${by}`) : null,
      ].filter(Boolean),
    );
  }

  /* ================================================================ 1 data */

  function buildDataCard() {
    const card = stepCard(1, "Choose your data", "How many synthetic windows to generate, as a share of the real replicas' size.", {
      id: "lab-data",
      tour: "lab-data",
      right: h("span", { class: "tag-synthetic" }, "Synthetic"),
    });
    refs.scaleSeg = segmented(
      SCALES.map((sc) => ({
        id: sc.id,
        content: [
          h("span", { class: "lab-seg-main" }, `${sc.label} · ${sc.pct}`),
          h("span", { class: "lab-seg-sub" }, `≈ ${fmt.compact(windowCount(stats, sc.scale))} windows`),
        ],
      })),
      SCALES.find((sc) => sc.scale === ui.scale)?.id || "medium",
      (id) => {
        ui.scale = SCALES.find((sc) => sc.id === id).scale;
        onDataChange();
      },
      { label: "Dataset size", cls: "lab-seg-tall" },
    );

    const seedInput = h("input", { type: "number", id: "lab-data-seed", min: 0, max: 99999, step: 1, inputmode: "numeric", value: ui.dataSeed, class: "lab-seed-input" });
    seedInput.addEventListener("change", () => {
      const v = Math.round(Number(seedInput.value));
      if (!Number.isFinite(v) || v < 0 || v > 99999) {
        seedInput.value = String(ui.dataSeed);
        return;
      }
      seedInput.value = String(v);
      ui.dataSeed = v;
      onDataChange();
    });
    const dice = h("button", {
      type: "button",
      class: "icon-btn lab-dice",
      "aria-label": "New random data seed",
      title: "New data seed",
      html: DICE_SVG,
      onClick: () => {
        ui.dataSeed = makeRng(ui.dataSeed, "lab-dice").int(100000);
        seedInput.value = String(ui.dataSeed);
        dice.classList.remove("is-rolling");
        void dice.offsetWidth;
        dice.classList.add("is-rolling");
        onDataChange();
      },
    });
    refs.seedInput = seedInput;

    const controls = h(
      "div",
      { class: "lab-data-controls" },
      h("div", { class: "field" }, h("span", { class: "label", id: "lab-scale-label" }, "Dataset size"), refs.scaleSeg.el),
      h(
        "div",
        { class: "field" },
        h("label", { class: "label", for: "lab-data-seed" }, "Data seed"),
        h("div", { class: "lab-seed-row" }, seedInput, dice),
        h("div", { class: "hint" }, "Same seed, same windows: every run is reproducible."),
      ),
    );
    refs.scaleSeg.el.setAttribute("aria-labelledby", "lab-scale-label");

    const stat = (label, sub) => {
      const value = h("div", { class: "stat-value tabular" }, "—");
      const subEl = h("div", { class: "stat-sub" }, sub);
      return { el: h("div", { class: "stat lab-mini-stat" }, h("div", { class: "stat-label" }, label), value, subEl), value, sub: subEl };
    };
    refs.cWindows = stat("Windows", "60-second windows, 3 hosts per replica");
    refs.cAttacks = stat("Attack windows", "labelled as under attack");
    refs.cPrev = stat("Attack share", "prevalence (the no-skill AP)");
    refs.cEpisodes = stat("Attack episodes", "contiguous attack periods");
    const summary = h("div", { class: "lab-corpus grid-4" }, refs.cWindows.el, refs.cAttacks.el, refs.cPrev.el, refs.cEpisodes.el);

    card.append(controls, summary);
    return card;
  }

  function onDataChange() {
    saveUi();
    renderExpChips();
    updateCorpus();
    rebuildReplicaGrid();
    schedulePreview();
  }

  function updateCorpus() {
    const c = getCorpus(ui.scale, ui.dataSeed);
    const sm = c.summary;
    const eps = sm.perReplica.reduce((a, r) => a + r.nEpisodes, 0);
    tweenNumber(refs.cWindows.value, sm.n, (v) => fmt.int(v));
    tweenNumber(refs.cAttacks.value, sm.nPositive, (v) => fmt.int(v));
    tweenNumber(refs.cPrev.value, sm.prevalence * 100, (v) => `${v.toFixed(1)}%`);
    tweenNumber(refs.cEpisodes.value, eps, (v) => fmt.int(v));
  }

  /* ================================================================ 2 split */

  function buildSplitCard() {
    const reset = h(
      "button",
      { type: "button", class: "btn sm ghost", onClick: resetSplit, html: `${icon("reset")}<span>Reset to default</span>` },
    );
    const card = stepCard(2, "Split into training and testing", "Decide which windows the model learns from and which it is marked on.", {
      id: "lab-split",
      tour: "lab-split",
      right: reset,
    });

    refs.strategySeg = segmented(
      STRATEGIES.map((st) => ({
        id: st.id,
        content: st.id === "default" ? [h("span", {}, st.label), h("span", { class: "pill info lab-rec" }, "Recommended")] : [st.label],
      })),
      ui.strategy,
      (id) => {
        ui.strategy = id;
        onStrategyChange();
      },
      { label: "Split strategy", cls: "lab-strategy-seg" },
    );
    refs.strategyExplain = h("p", { class: "lab-explain" });
    refs.honestyPill = h("span", { class: "lab-honesty" });
    const explainRow = h("div", { class: "lab-explain-row" }, refs.honestyPill, refs.strategyExplain);

    refs.replicaGrid = h("div", { class: "lab-replicas", role: "list", "aria-label": "Replicas (testbeds)" });
    refs.loroNote = h("p", { class: "hint lab-loro-note", hidden: true });

    // percentage controls
    refs.valSlider = slider({
      id: "lab-val-pct",
      label: "Validation share of the training windows",
      min: 5,
      max: 40,
      value: ui.valPct,
      format: (v) => `${v}%`,
      hint: "Held back from training to pick the best epoch (early stopping). Never used for the test score.",
      onInput: (v) => {
        ui.valPct = v;
        saveUi();
        schedulePreview();
      },
    });
    refs.trainSlider = slider({
      id: "lab-train-pct",
      label: "Training",
      min: 20,
      max: 90,
      value: ui.trainPct,
      format: (v) => `${v}%`,
      onInput: (v) => {
        ui.trainPct = v;
        if (ui.trainPct + ui.holdValPct > 95) {
          ui.holdValPct = Math.max(5, 95 - ui.trainPct);
          refs.holdValSlider.set(ui.holdValPct);
        }
        syncTestPct();
        saveUi();
        schedulePreview();
      },
    });
    refs.holdValSlider = slider({
      id: "lab-holdval-pct",
      label: "Validation",
      min: 5,
      max: 40,
      value: ui.holdValPct,
      format: (v) => `${v}%`,
      onInput: (v) => {
        ui.holdValPct = v;
        if (ui.trainPct + ui.holdValPct > 95) {
          ui.trainPct = Math.max(20, 95 - ui.holdValPct);
          refs.trainSlider.set(ui.trainPct);
        }
        syncTestPct();
        saveUi();
        schedulePreview();
      },
    });
    refs.testPctOut = h("div", { class: "lab-test-pct" });
    refs.pctReplica = h("div", { class: "lab-pcts" }, refs.valSlider.el);
    refs.pctRows = h(
      "div",
      { class: "lab-pcts lab-pcts-3" },
      refs.trainSlider.el,
      refs.holdValSlider.el,
      h("div", { class: "field" }, h("span", { class: "label" }, "Test (the rest)"), refs.testPctOut, h("div", { class: "hint" }, "At least 5% test and 20% training.")),
    );

    // live split visual
    refs.splitBarHost = h("div", { class: "lab-splitbar" });
    refs.splitBar = stackedBar(refs.splitBarHost, {
      segments: partSegments({ train: 0, val: 0, test: 0, unused: 0 }),
      height: 34,
      showPercent: true,
      format: (v) => `${fmt.int(v)} windows`,
      ariaLabel: "Share of windows in training, validation, test and unused",
    });
    refs.partTiles = {};
    const tiles = h("div", { class: "lab-part-tiles" });
    for (const [key, label, role] of [
      ["train", "Training", "train"],
      ["val", "Validation", "val"],
      ["test", "Test", "test"],
      ["unused", "Unused", "unused"],
    ]) {
      const n = h("div", { class: "lab-part-n tabular" }, "0");
      const pos = h("div", { class: "lab-part-pos tiny" }, "0 attack windows");
      const pct = h("span", { class: "lab-part-pct tabular" }, "0%");
      const labelEl = h("span", { class: "lab-part-label" }, label);
      const tile = h(
        "div",
        { class: "lab-part", "data-part": key },
        h("div", { class: "lab-part-head" }, h("span", { class: "swatch", style: { background: `var(--c-${role})` } }), labelEl, pct),
        n,
        pos,
      );
      refs.partTiles[key] = { tile, n, pos, pct, labelEl, label };
      tiles.appendChild(tile);
    }
    // chronological only: the windows set aside at each cut (splits.js summary.purged)
    refs.purgeNote = h("p", { class: "lab-purge-note small", hidden: true });
    refs.splitVisual = h(
      "div",
      { class: "lab-split-visual" },
      h("div", { class: "lab-sub-title" }, "What goes where", h("span", { class: "faint tiny lab-sub-note" }, "updates live")),
      refs.splitBarHost,
      tiles,
      refs.purgeNote,
    );
    refs.splitStatus = h("p", { class: "sr-only", role: "status" });

    refs.splitErrors = h("div", { class: "lab-errors", role: "alert", hidden: true });

    // leakage certificate
    refs.certOverall = h("div", { class: "lab-cert-overall" });
    refs.certRows = h("div", { class: "lab-cert-rows" });
    refs.certWhy = notice(
      "bad",
      "alert",
      h("strong", {}, "Why this is a leak. "),
      "Attacks last many minutes and neighbouring minutes look almost identical. When a split puts minute 12 of an attack in training and minute 13 in the test set, the model can recognise the attack instead of detecting it, and the score goes up for the wrong reason.",
    );
    refs.certWhy.hidden = true;
    refs.foldChips = h("div", { class: "lab-fold-chips", hidden: true, role: "group", "aria-label": "Preview a leave-one-replica-out fold" });
    refs.foldSummary = h("p", { class: "lab-fold-summary small", hidden: true });
    const cert = h(
      "div",
      { class: "card inset lab-cert", "data-tour": "lab-leakage" },
      h(
        "div",
        { class: "lab-cert-head" },
        h("div", { class: "lab-cert-title" }, iconSpan("shield", "lab-ico lab-cert-ico"), h("div", {}, h("div", { class: "card-title" }, "Leakage certificate"), h("div", { class: "card-sub" }, "Can the test set see what training saw? Checked live on this exact split."))),
        refs.certOverall,
      ),
      refs.foldChips,
      refs.foldSummary,
      refs.certRows,
      refs.certWhy,
    );

    card.append(refs.strategySeg.el, explainRow, refs.replicaGrid, refs.loroNote, refs.pctReplica, refs.pctRows, refs.splitErrors, refs.splitVisual, cert, refs.splitStatus);
    rebuildReplicaGrid();
    return card;
  }

  function partSegments(n, purged = 0) {
    return [
      { id: "train", label: "Training", value: n.train, color: "train" },
      { id: "val", label: "Validation", value: n.val, color: "val" },
      { id: "test", label: "Test", value: n.test, color: "test" },
      { id: "unused", label: purged ? "Unused or set aside at the cuts" : "Unused", value: n.unused, color: "unused" },
    ];
  }

  /** The number of windows a chronological split set aside at its cuts (0 otherwise). */
  function purgedCount(summary) {
    const p = summary && summary.purged;
    return p ? (p.val || 0) + (p.test || 0) : 0;
  }

  /** One sentence of final values for the split card's status region. */
  function splitSentence(p) {
    if (!p) return "";
    if (!p.built || !p.cert) return p.errors && p.errors.length ? `This split cannot run yet: ${p.errors.join(" ")}` : "";
    const n = p.built.summary.n;
    const purged = purgedCount(p.built.summary);
    const parts = [`${fmt.int(n.train)} training`, `${fmt.int(n.val)} validation`, `${fmt.int(n.test)} test windows`];
    const rest = n.unused ? `, ${fmt.int(n.unused)} ${purged ? `unused or set aside at the cuts (${fmt.int(purged)} set aside)` : "unused"}` : "";
    const fold = p.folds ? ` Previewing the fold that holds out ${p.previewFold}.` : "";
    const sup = supportInfo(p.cert);
    return `Split: ${parts.join(", ")}${rest}.${fold} Leakage certificate: ${OVERALL[p.cert.overall].word}.${sup ? ` Test support: ${sup.word.toLowerCase()}, which is not a leak.` : ""}`;
  }

  function syncTestPct() {
    const t = 100 - ui.trainPct - ui.holdValPct;
    refs.testPctOut.textContent = `${t}%`;
    refs.testPctOut.classList.toggle("is-low", t < 5);
  }

  function resetSplit() {
    const d = defaultUi();
    ui.strategy = d.strategy;
    ui.roles = d.roles;
    ui.include = d.include;
    ui.valPct = d.valPct;
    ui.trainPct = d.trainPct;
    ui.holdValPct = d.holdValPct;
    ui.loroPreview = d.loroPreview;
    syncControlsFromUi();
    onStrategyChange();
  }

  function onStrategyChange() {
    saveUi();
    const st = STRATEGY[ui.strategy];
    refs.strategySeg.set(ui.strategy);
    refs.strategyExplain.textContent = st.explain;
    refs.honestyPill.replaceChildren(pill(st.honesty.cls, st.honesty.icon, st.honesty.text));
    const replicaMode = ["default", "pick", "loro"].includes(ui.strategy);
    refs.pctReplica.hidden = !replicaMode;
    refs.pctRows.hidden = replicaMode;
    rebuildReplicaGrid();
    schedulePreview(0);
  }

  /** Rebuild the 8 replica cards (on strategy / data change only, so focus survives role clicks). */
  function rebuildReplicaGrid() {
    if (!refs.replicaGrid) return;
    const corpus = getCorpus(ui.scale, ui.dataSeed);
    const per = Object.fromEntries(corpus.summary.perReplica.map((r) => [r.id, r]));
    refs.replicaGrid.replaceChildren();
    refs.repCards = {};
    const maxPrev = 0.3;
    replicaIds.forEach((id, i) => {
      const r = per[id] || { n: 0, nPositive: 0, prevalence: 0 };
      const real = replicaStats[id];
      const few = real.n_positive < MIN_SUPPORT_FOR_RATES * 2 || r.nPositive < MIN_SUPPORT_FOR_RATES;
      const badge = h("span", { class: "lab-rep-role" });
      const meterFill = h("span", { class: "lab-meter-fill", style: { width: `${Math.min(100, (r.prevalence / maxPrev) * 100)}%` } });
      const ctl = h("div", { class: "lab-rep-ctl" });
      const card = h(
        "div",
        { class: "lab-rep", role: "listitem", "data-role": "train", style: { "--i": String(i) } },
        h("div", { class: "lab-rep-head" }, h("span", { class: "lab-rep-name" }, id), badge),
        h(
          "div",
          { class: "lab-rep-nums" },
          h("span", {}, h("b", { class: "tabular" }, fmt.int(r.n)), " windows"),
          h("span", {}, h("b", { class: "tabular" }, fmt.int(r.nPositive)), " attacks"),
        ),
        h("div", { class: "lab-meter", title: "Attack share (bar scale 0–30%)", "aria-hidden": "true" }, meterFill),
        h(
          "div",
          { class: "lab-rep-foot" },
          h("span", { class: "tiny faint tabular" }, `${pct1(r.prevalence)} attack windows`),
          few ? pill("warn", "alert", "Few attacks", { title: `Only ${real.n_positive} attack windows in the real ${id} replica` }) : null,
        ),
        ctl,
      );
      refs.repCards[id] = { card, badge, ctl };
      buildReplicaControl(id, ctl);
      refs.replicaGrid.appendChild(card);
    });
    refs.replicaGrid.classList.remove("is-swapping");
    void refs.replicaGrid.offsetWidth;
    refs.replicaGrid.classList.add("is-swapping");
    refs.loroNote.hidden = ui.strategy !== "loro";
    applyReplicaRoles();
  }

  function buildReplicaControl(id, ctl) {
    ctl.replaceChildren();
    if (ui.strategy === "pick") {
      const seg = segmented(
        [
          { id: "train", label: "Train" },
          { id: "test", label: "Test" },
          { id: "off", label: "Off" },
        ],
        ui.roles[id],
        (v) => {
          ui.roles[id] = v;
          saveUi();
          applyReplicaRoles();
          schedulePreview();
        },
        { label: `Role of ${id}`, cls: "lab-seg-sm" },
      );
      ctl.appendChild(seg.el);
    } else if (ui.strategy === "random" || ui.strategy === "chrono") {
      const input = h("input", { type: "checkbox", "aria-label": `Include ${id}` });
      input.checked = !!ui.include[id];
      input.addEventListener("change", () => {
        ui.include[id] = input.checked;
        saveUi();
        applyReplicaRoles();
        schedulePreview();
      });
      ctl.appendChild(h("label", { class: "switch lab-switch" }, input, h("span", {}), "Include"));
    } else if (ui.strategy === "loro") {
      const btn = h(
        "button",
        {
          type: "button",
          class: "btn sm ghost lab-preview-btn",
          "aria-pressed": String(ui.loroPreview === id),
          onClick: () => {
            ui.loroPreview = id;
            saveUi();
            schedulePreview(0);
          },
        },
        "Preview this fold",
      );
      ctl.appendChild(btn);
    } else {
      ctl.appendChild(h("span", { class: "tiny faint" }, id === "santos" ? "Held out by the default protocol" : "Trains in the default protocol"));
    }
  }

  /** Card border / badge per replica from the preview (or the UI when the config is invalid). */
  function applyReplicaRoles() {
    if (!refs.repCards) return;
    const summary = S.preview && S.preview.built && S.preview.key === previewKey() ? S.preview.built.summary.perReplica : null;
    for (const id of replicaIds) {
      const rc = refs.repCards[id];
      let role;
      if (summary && summary[id]) role = summary[id].role;
      else if (ui.strategy === "pick") role = ui.roles[id] === "off" ? "unused" : ui.roles[id];
      else if (ui.strategy === "random" || ui.strategy === "chrono") role = ui.include[id] ? "mixed" : "unused";
      else if (ui.strategy === "loro") role = id === ui.loroPreview ? "test" : "train";
      else role = id === "santos" ? "test" : "train";
      const info = ROLE_INFO[role] || ROLE_INFO.unused;
      if (rc.card.dataset.role !== role) {
        rc.card.dataset.role = role;
        rc.badge.classList.remove("pop");
        void rc.badge.offsetWidth;
        rc.badge.classList.add("pop");
      }
      rc.badge.replaceChildren(h("span", { class: "dot", style: { color: `var(${info.color})` } }), info.label);
      if (ui.strategy === "loro") {
        const b = rc.ctl.querySelector(".lab-preview-btn");
        if (b) b.setAttribute("aria-pressed", String(ui.loroPreview === id));
      }
    }
  }

  /* ---------------------------------------------------------------- preview */

  function previewKey() {
    return JSON.stringify([ui.scale, ui.dataSeed, splitConfigFor(ui, replicaIds), ui.model.kind, effectiveFeatures(ui), ui.loroPreview]);
  }

  function schedulePreview(delay = 150) {
    clearTimeout(previewTimer);
    applyReplicaRoles();
    updateRunSummary();
    previewTimer = setTimeout(computePreview, delay);
  }

  function computePreview() {
    const key = previewKey();
    const corpus = getCorpus(ui.scale, ui.dataSeed);
    const split = splitConfigFor(ui, corpus.replicaIds);
    const v = validateSplitConfig(split, corpus.replicaIds);
    const featureSet = effectiveFeatures(ui);
    const certOpts = ui.model.kind === "pretrained" ? { pretrainedTrainReplicas: [...PRETRAINED_TRAIN_REPLICAS], featureSet } : { featureSet };
    let preview = { key, corpus, split, errors: v.errors, built: null, cert: null, folds: null };
    if (v.ok) {
      try {
        if (split.mode === "loro") {
          const folds = loroFolds(split, corpus.replicaIds).map((fc) => {
            const built = buildSplit(corpus, fc);
            return { heldOut: fc.test[0], built, cert: leakageCertificate(corpus, built, certOpts) };
          });
          const shown = folds.find((f) => f.heldOut === ui.loroPreview) || folds[0];
          preview = { ...preview, built: shown.built, cert: shown.cert, folds, previewFold: shown.heldOut };
        } else {
          const built = buildSplit(corpus, split);
          preview = { ...preview, built, cert: leakageCertificate(corpus, built, certOpts) };
        }
      } catch (err) {
        preview.errors = [err.message];
      }
    }
    S.preview = preview;
    renderPreview();
  }

  function renderPreview() {
    const p = S.preview;
    // errors
    refs.splitErrors.replaceChildren();
    refs.splitErrors.hidden = !p.errors.length;
    if (p.errors.length) {
      refs.splitErrors.appendChild(
        notice("bad", "alert", h("strong", {}, "This split cannot run yet. "), h("ul", { class: "lab-error-list" }, ...p.errors.map((e) => h("li", {}, e)))),
      );
    }
    refs.splitVisual.classList.toggle("is-stale", !p.built);
    refs.purgeNote.hidden = true;
    if (p.built) {
      const sm = p.built.summary;
      const purged = purgedCount(sm);
      refs.splitBar.update({ segments: partSegments(sm.n, purged) });
      for (const key of ["train", "val", "test", "unused"]) {
        const t = refs.partTiles[key];
        tweenNumber(t.n, sm.n[key], (v) => fmt.int(v), 600);
        t.pos.textContent =
          key === "unused" && purged
            ? `${fmt.int(purged)} set aside at the cuts · ${plural(sm.positives[key], "attack window")}`
            : `${plural(sm.positives[key], "attack window")}`;
        t.labelEl.textContent = key === "unused" && purged ? "Unused / set aside" : t.label;
        t.pct.textContent = `${sm.pct[key].toFixed(1)}%`;
        t.tile.classList.toggle("is-empty", sm.n[key] === 0);
      }
      if (purged) {
        const pp = sm.purged.positives || { val: 0, test: 0 };
        refs.purgeNote.hidden = false;
        refs.purgeNote.replaceChildren(
          iconSpan("clock"),
          h(
            "span",
            {},
            h("b", {}, `${plural(purged, "window")} set aside at the cuts`),
            ` (${fmt.int(sm.purged.val || 0)} before validation, ${fmt.int(sm.purged.test || 0)} before test; ${plural((pp.val || 0) + (pp.test || 0), "attack window")}): they continue an attack or quiet period from the part before, so keeping them would put neighbouring minutes on both sides. The real chronological protocol drops a 600-second gap after each cut for the same reason.`,
          ),
        );
      }
    }
    applyReplicaRoles();
    renderCertificate();
    renderLoroChips();
    renderModelWarning();
    if (ui.model.kind === "knn") renderModelInfo(); // the badge counts the stored training windows
    updateRunSummary();
    updateRunButton();
    // the first preview is the page loading: nothing to announce yet
    if (S.previewCount++ > 0) announceSplit();
  }

  function renderLoroChips() {
    const p = S.preview;
    refs.foldChips.replaceChildren();
    refs.foldChips.hidden = !p.folds;
    refs.loroNote.hidden = !p.folds;
    if (!p.folds) return;
    refs.loroNote.textContent = `Previewing the fold that holds out ${p.previewFold}. Each of the ${p.folds.length} folds trains on the other ${p.folds.length - 1} replicas and tests on one. Pick a replica to preview its fold.`;
    refs.foldChips.appendChild(h("span", { class: "label lab-fold-chips-label" }, "Folds"));
    for (const f of p.folds) {
      const sev = OVERALL[f.cert.overall];
      const sup = supportInfo(f.cert);
      const what = `${f.heldOut} held out: ${sev.word}${sup ? `; ${sup.word.toLowerCase()} (not a leak)` : ""}`;
      const chip = h("button", {
        type: "button",
        class: `lab-fold-chip ${sev.cls}${sup ? " has-support-note" : ""}`,
        "aria-pressed": String(f.heldOut === p.previewFold),
        "aria-label": `Preview the fold with ${what}`,
        title: what,
        html: `${icon(sev.icon)}<span>${escapeHtml(f.heldOut)}</span>${sup ? `<span class="lab-fold-chip-sup">${icon("alert")}<span>few attacks</span></span>` : ""}`,
        onClick: () => {
          ui.loroPreview = f.heldOut;
          saveUi();
          schedulePreview(0);
        },
      });
      refs.foldChips.appendChild(chip);
    }
    // the verdict over all folds, as the run will report it
    let merged = null;
    try {
      merged = mergeCertificates(p.folds);
    } catch {
      merged = null;
    }
    refs.foldSummary.hidden = !merged;
    if (merged) {
      const ov = OVERALL[merged.overall];
      const sup = supportInfo(merged);
      refs.foldSummary.replaceChildren(
        h("span", { class: "faint" }, `All ${p.folds.length} folds: `),
        pill(ov.cls, ov.icon, ov.word),
        sup ? pill(sup.cls, sup.icon, sup.word) : null,
        sup ? h("span", { class: "faint" }, `${sup.short}; shown, but left out of the summary. Missing attacks are a support limit, not a leak.`) : null,
      );
    }
  }

  function renderCertificate() {
    const p = S.preview;
    const cert = p.cert;
    refs.certOverall.replaceChildren();
    if (!cert) {
      refs.certRows.replaceChildren(h("p", { class: "muted small" }, "Fix the split settings above to see the certificate."));
      refs.certWhy.hidden = true;
      return;
    }
    // two separate verdicts: leakage (the checks that can inflate a score) and
    // test support (whether a score can be measured at all; never a leak)
    const ov = OVERALL[cert.overall];
    const sup = supportInfo(cert);
    refs.certOverall.appendChild(
      h(
        "div",
        { class: "lab-cert-verdicts" },
        h("span", { class: "lab-cert-verdict" }, h("span", { class: "tiny faint" }, "Leakage"), pill(ov.cls, ov.icon, ov.word)),
        sup ? h("span", { class: "lab-cert-verdict" }, h("span", { class: "tiny faint" }, "Test support"), pill(sup.cls, sup.icon, sup.word)) : null,
      ),
    );
    const existing = new Map([...refs.certRows.querySelectorAll(".lab-cert-row")].map((r) => [r.dataset.id, r]));
    const rowFor = (ch) => {
      const sev = SEVERITY[ch.severity];
      let row = existing.get(ch.id);
      const content = [
        h("span", { class: "lab-cert-pill" }, pill(sev.cls, sev.icon, sev.word)),
        h("div", { class: "lab-cert-text" }, h("div", { class: "lab-cert-label" }, ch.label), h("div", { class: "lab-cert-detail" }, ch.detail)),
      ];
      if (!row) {
        row = h("div", { class: "lab-cert-row" });
        row.dataset.id = ch.id;
      }
      const changed = row.dataset.sev && row.dataset.sev !== ch.severity;
      row.dataset.sev = ch.severity;
      row.replaceChildren(...content);
      if (changed) {
        row.classList.remove("lab-flash");
        void row.offsetWidth;
        row.classList.add("lab-flash");
      }
      return row;
    };
    const isSupport = (ch) => (ch.group ? ch.group === "support" : ch.id === "test-support");
    const leakRows = cert.checks.filter((ch) => !isSupport(ch)).map(rowFor);
    const supRows = cert.checks.filter(isSupport).map(rowFor);
    refs.certRows.replaceChildren(
      h("div", { class: "lab-cert-group label" }, "Leakage checks"),
      ...leakRows,
      ...(supRows.length ? [h("div", { class: "lab-cert-group label" }, "Test support ", h("span", { class: "lab-cert-group-note" }, "(can the score be measured? not a leakage check)")), ...supRows] : []),
    );
    const sev = (id) => cert.checks.find((c) => c.id === id)?.severity;
    // the leak mechanisms the explainer describes: a cut stretch, or repeated active windows
    refs.certWhy.hidden = !(sev("stretch-disjoint") === "fail" || sev("duplicate-rows") === "fail");
  }

  /* ================================================================ 3 model */

  function buildModelCard() {
    const card = stepCard(3, "Choose a model", "What learns from the training windows.", { id: "lab-model", tour: "lab-model" });
    refs.modelSeg = segmented(
      MODELS.map((m) => ({ id: m.id, label: m.label })),
      ui.model.kind,
      (id) => {
        ui.model.kind = id;
        onModelChange();
      },
      { label: "Model", cls: "lab-model-seg" },
    );
    refs.modelDesc = h("p", { class: "lab-explain" });
    refs.modelBadges = h("div", { class: "row lab-model-badges" });
    refs.modelWarn = h("div", { hidden: true });

    refs.featSeg = segmented(
      FEATURES.map((f) => ({ id: f.id, content: [h("span", { class: "lab-seg-main" }, f.label), h("span", { class: "lab-seg-sub" }, f.sub)] })),
      ui.features,
      (id) => {
        ui.features = id;
        onFeaturesChange();
      },
      { label: "Features the model sees", cls: "lab-seg-tall lab-feat-seg" },
    );
    refs.featSeg.el.setAttribute("aria-labelledby", "lab-feat-label");
    refs.featSeg.el.setAttribute("aria-describedby", "lab-feat-note");
    refs.featNote = h("p", { class: "hint lab-feat-note", id: "lab-feat-note" });
    const featField = h("div", { class: "field lab-feat" }, h("span", { class: "label", id: "lab-feat-label" }, "What the model sees"), refs.featSeg.el, refs.featNote);

    const epochs = slider({
      id: "lab-epochs",
      label: "Epochs (passes over the training data)",
      min: 5,
      max: 40,
      value: ui.model.epochs,
      onInput: (v) => {
        ui.model.epochs = v;
        onModelParam();
      },
    });
    const patience = slider({
      id: "lab-patience",
      label: "Early-stopping patience (epochs without improvement)",
      min: 3,
      max: 10,
      value: ui.model.patience,
      onInput: (v) => {
        ui.model.patience = v;
        onModelParam();
      },
    });
    const lrSel = h("select", { id: "lab-lr" }, ...LR_OPTIONS.map((v) => h("option", { value: String(v) }, `${lrText(v)}${v === 1e-3 ? " (default)" : ""}`)));
    lrSel.value = String(ui.model.lr);
    lrSel.addEventListener("change", () => {
      ui.model.lr = Number(lrSel.value);
      onModelParam();
    });
    const bsSel = h("select", { id: "lab-batch" }, ...BATCH_OPTIONS.map((v) => h("option", { value: String(v) }, `${v}${v === 256 ? " (default)" : ""}`)));
    bsSel.value = String(ui.model.batchSize);
    bsSel.addEventListener("change", () => {
      ui.model.batchSize = Number(bsSel.value);
      onModelParam();
    });
    const seed = h("input", { type: "number", id: "lab-train-seed", min: 0, max: 99999, step: 1, inputmode: "numeric", value: ui.model.seed });
    seed.addEventListener("change", () => {
      const v = Math.round(Number(seed.value));
      if (!Number.isFinite(v) || v < 0 || v > 99999) {
        seed.value = String(ui.model.seed);
        return;
      }
      seed.value = String(v);
      ui.model.seed = v;
      onModelParam();
    });
    refs.modelCtl = { epochs, patience, lrSel, bsSel, seed };
    refs.advNote = h("p", { class: "hint lab-adv-note" });
    refs.advFieldset = h(
      "fieldset",
      { class: "lab-adv-grid" },
      h("legend", { class: "sr-only" }, "Training settings"),
      epochs.el,
      patience.el,
      h("div", { class: "field" }, h("label", { class: "label", for: "lab-lr" }, "Learning rate"), lrSel),
      h("div", { class: "field" }, h("label", { class: "label", for: "lab-batch" }, "Batch size"), bsSel),
      h("div", { class: "field" }, h("label", { class: "label", for: "lab-train-seed" }, "Training seed"), seed),
    );
    const adv = h(
      "details",
      { class: "acc lab-adv" },
      h("summary", {}, h("span", {}, "Advanced training settings", h("span", { class: "faint small lab-adv-sum" }))),
      h("div", { class: "acc-body" }, refs.advNote, refs.advFieldset, h("p", { class: "hint" }, "Optimiser: AdamW (weight decay 0.01) with a cosine learning-rate schedule, gradient clipping at 1.0 and class weighting, as in the project's Python training code.")),
    );
    refs.advSummary = adv.querySelector(".lab-adv-sum");
    card.append(refs.modelSeg.el, refs.modelDesc, refs.modelBadges, refs.modelWarn, featField, adv);
    return card;
  }

  function onFeaturesChange() {
    saveUi();
    renderModelInfo();
    schedulePreview(0); // the certificate compares rows on the columns the model sees
  }

  function renderFeatures() {
    if (!refs.featSeg) return;
    const pretrained = ui.model.kind === "pretrained";
    const eff = effectiveFeatures(ui);
    refs.featSeg.set(eff);
    refs.featSeg.buttons.forEach((b) => {
      b.disabled = pretrained;
      b.setAttribute("aria-disabled", String(pretrained));
    });
    refs.featSeg.el.classList.toggle("is-disabled", pretrained);
    refs.featNote.textContent = pretrained
      ? "The pretrained model was trained on all 42 features of the real data, so it always sees all four sources."
      : eff === "m2"
        ? "Only the 24 network-metric columns (Suricata-style flow and alert aggregates): the log, host and graph sources are switched off. The leakage certificate then compares rows on those 24 columns."
        : "Every window's 42 features: log templates, network metrics, host identity and graph structure.";
  }

  function onModelChange() {
    saveUi();
    renderModelInfo();
    schedulePreview(0); // the certificate gains / loses the pretrained-contamination check
  }

  function onModelParam() {
    saveUi();
    renderModelInfo();
    updateRunSummary();
  }

  function renderModelInfo() {
    const m = MODEL[ui.model.kind];
    refs.modelSeg.set(ui.model.kind);
    refs.modelDesc.textContent = m.desc;
    const eff = effectiveFeatures(ui);
    const params = ui.model.kind === "logreg" ? (eff === "m2" ? 24 : 42) + 5 : TESSERA_PARAMS;
    const trainRows = S.preview && S.preview.built ? S.preview.built.summary.n.train : null;
    refs.modelBadges.replaceChildren(
      ui.model.kind === "knn"
        ? h("span", { class: "badge" }, iconSpan("database"), h("span", { class: "tabular" }, trainRows != null ? `No parameters: stores all ${fmt.int(trainRows)} training windows` : "No parameters: stores every training window"))
        : h("span", { class: "badge" }, iconSpan("cpu"), h("span", { class: "tabular" }, `${fmt.int(params)} parameters`)),
      ui.model.kind === "pretrained"
        ? h("span", { class: "tag-real", title: "Trained on real AIT windows from the 7 replicas other than santos" }, "Real weights")
        : ui.model.kind === "knn"
          ? h("span", { class: "badge" }, iconSpan("flask"), `k = ${KNN_DEFAULT_K} nearest windows`)
          : h("span", { class: "badge" }, iconSpan("flask"), "Trains in your browser"),
      (refs.estBadge = h("span", { class: "badge" }, iconSpan("clock"), h("span", { class: "tabular lab-est" }, "…"))),
    );
    const tessera = ui.model.kind === "tessera";
    const c = refs.modelCtl;
    for (const input of [c.epochs.input, c.patience.input, c.lrSel, c.bsSel]) input.disabled = !tessera;
    c.seed.disabled = ui.model.kind === "pretrained" || ui.model.kind === "knn";
    refs.advFieldset.classList.toggle("is-disabled", !tessera);
    refs.advNote.textContent = tessera
      ? ""
      : ui.model.kind === "logreg"
        ? `Logistic regression uses fixed settings (${LOGREG_STEPS} full-batch steps, Adam, learning rate 0.05); only the training seed applies.`
        : ui.model.kind === "knn"
          ? `The memoriser has one setting, k = ${KNN_DEFAULT_K}: it scores a window by the labels of its ${KNN_DEFAULT_K} nearest training windows, the nearest counting most (weights 1, 1/2, 1/4, ...). It has nothing to train, so these settings do not apply.`
          : "The pretrained model is not trained here, so these settings do not apply.";
    refs.advNote.hidden = tessera;
    refs.advSummary.textContent = tessera
      ? ` · ${ui.model.epochs} epochs · lr ${lrText(ui.model.lr)} · batch ${ui.model.batchSize}`
      : "";
    renderFeatures();
    renderModelWarning();
    updateRunSummary();
    if (!(S.run && S.run.active)) applyStageShares(stageWeights(ui.model.kind));
  }

  function renderModelWarning() {
    if (!refs.modelWarn) return;
    refs.modelWarn.replaceChildren();
    const split = S.preview ? S.preview.split : splitConfigFor(ui, replicaIds);
    const seen = ui.model.kind === "pretrained" ? testReplicasOf(split, replicaIds).filter((r) => PRETRAINED_TRAIN_REPLICAS.includes(r)) : [];
    refs.modelWarn.hidden = !seen.length;
    if (seen.length) {
      refs.modelWarn.appendChild(
        notice(
          "warn",
          "alert",
          h("strong", {}, "The pretrained model has already seen these testbeds. "),
          `Its real training data included ${plainIds(seen)}, and the synthetic test windows are calibrated from that same data, so its score here will look better than it should. Only santos is unseen by it. The leakage certificate flags this too.`,
        ),
      );
    }
  }

  /* ================================================================ run bar */

  function buildRunBar() {
    refs.runBtn = h("button", {
      type: "button",
      class: "btn primary lg lab-run-btn",
      onClick: (e) => {
        // the second click of a double-click must not cancel the run the first one started
        if (e && e.detail > 1) return;
        if (S.run && S.run.active) cancelRun();
        else startRun();
      },
    });
    refs.resetBtn = h("button", { type: "button", class: "btn ghost", onClick: resetAll, html: `${icon("reset")}<span>Reset</span>` });
    refs.runSummary = h("div", { class: "lab-run-summary small" });
    const kbd = h("div", { class: "tiny faint lab-kbd" }, h("kbd", {}, /Mac|iPhone|iPad/.test(navigator.platform || "") ? "⌘" : "Ctrl"), " + ", h("kbd", {}, "Enter"));
    const bar = h(
      "div",
      { class: "lab-runbar", "data-tour": "lab-run" },
      h("div", { class: "lab-runbar-info" }, refs.runSummary, kbd),
      h("div", { class: "lab-runbar-actions" }, refs.resetBtn, refs.runBtn),
    );
    return bar;
  }

  function updateRunSummary() {
    if (!refs.runSummary) return;
    const sc = SCALES.find((x) => x.scale === ui.scale);
    const est = estimateMs(ui, S.preview && S.preview.key === previewKey() ? S.preview : null);
    const estText = est == null ? "" : `about ${fmt.ms(Math.max(500, Math.round(est / 100) * 100))}`;
    // built once and updated in place: this runs on every slider input event
    const chips = [
      { key: "data", icon: "database", text: `${sc ? sc.label : ""} data` },
      { key: "split", icon: "split", text: STRATEGY[ui.strategy].short },
      { key: "model", icon: "cpu", text: MODEL[ui.model.kind].short },
      { key: "feat", icon: "layers", text: FEATURE[effectiveFeatures(ui)].short },
    ];
    if (estText) chips.push({ key: "est", icon: "clock", text: estText, cls: "faint" });
    syncChips(refs.runSummary, chips);
    if (refs.estBadge) {
      const e = refs.estBadge.querySelector(".lab-est");
      if (e) e.textContent = estText ? `Estimated run time ${estText.replace("about ", "≈ ")}` : "Estimated run time —";
      refs.estBadge.title = "Rough estimate measured on a laptop; early stopping can make it shorter";
    }
  }

  function updateRunButton() {
    const btn = refs.runBtn;
    if (!btn) return;
    updateExpButtons();
    if (S.run && S.run.active && S.run.exp) {
      setArming(btn, 0);
      btn.classList.remove("is-cancel");
      btn.innerHTML = `${icon("play")}<span>Experiment running…</span>`;
      btn.disabled = true;
      btn.setAttribute("aria-label", "Run training and evaluation (unavailable while the split experiment runs)");
      refs.resetBtn.disabled = true;
      return;
    }
    const active = S.run && S.run.active;
    const cancelling = active && S.run.cancelling;
    const invalid = !S.preview || S.preview.errors.length > 0;
    if (active) {
      btn.classList.add("is-cancel");
      btn.innerHTML = `${icon("stop")}<span>${cancelling ? "Cancelling…" : "Cancel"}</span>`;
      btn.disabled = !!cancelling;
      btn.setAttribute("aria-label", "Cancel the run");
      setArming(btn, cancelling ? 0 : cancelArming());
    } else {
      btn.classList.remove("is-cancel");
      btn.innerHTML = `${icon("play")}<span>Run training + evaluation</span>`;
      btn.disabled = invalid;
      btn.setAttribute("aria-label", invalid ? "Run training and evaluation (fix the split settings first)" : "Run training and evaluation");
      setArming(btn, 0);
    }
    refs.resetBtn.disabled = !!active;
  }

  function resetAll() {
    if (S.run && S.run.active) return;
    const d = defaultUi();
    Object.assign(ui, d);
    ui.model = d.model;
    syncControlsFromUi();
    updateCorpus();
    onStrategyChange();
    renderModelInfo();
    resetProgressUi(null);
    setRunState("idle");
    toast("Settings reset to the default protocol.", { kind: "info" });
  }

  /** Push every control to match `ui` (after load / reset). */
  function syncControlsFromUi() {
    refs.scaleSeg.set(SCALES.find((sc) => sc.scale === ui.scale)?.id || "medium");
    refs.seedInput.value = String(ui.dataSeed);
    refs.strategySeg.set(ui.strategy);
    refs.valSlider.set(ui.valPct);
    refs.trainSlider.set(ui.trainPct);
    refs.holdValSlider.set(ui.holdValPct);
    syncTestPct();
    const c = refs.modelCtl;
    c.epochs.set(ui.model.epochs);
    c.patience.set(ui.model.patience);
    c.lrSel.value = String(ui.model.lr);
    c.bsSel.value = String(ui.model.batchSize);
    c.seed.value = String(ui.model.seed);
    const st = STRATEGY[ui.strategy];
    refs.strategyExplain.textContent = st.explain;
    refs.honestyPill.replaceChildren(pill(st.honesty.cls, st.honesty.icon, st.honesty.text));
    const replicaMode = ["default", "pick", "loro"].includes(ui.strategy);
    refs.pctReplica.hidden = !replicaMode;
    refs.pctRows.hidden = replicaMode;
    rebuildReplicaGrid();
    renderModelInfo();
  }

  /* ================================================================ 4 progress */

  function buildProgressCard() {
    refs.runStatus = h("span", {});
    const card = stepCard(4, "Watch it run", "Every stage reports how much of its work is done.", { id: "lab-progress", tour: "lab-progress", right: refs.runStatus });
    card.classList.add("lab-progress-card");

    // the ring is drawn inside a progressbar, so assistive tech can query the overall %
    refs.ringHost = h("div", { class: "lab-ring", role: "progressbar", "aria-label": "Overall progress", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": "0" });
    refs.ring = ring(refs.ringHost, { value: 0, size: 132, stroke: 11, label: "0%", sublabel: "overall", ariaLabel: "Overall progress" });
    refs.elapsed = h("span", { class: "tabular" }, "0.0 s");
    refs.eta = h("span", { class: "tabular" }, "—");
    const ringBox = h(
      "div",
      { class: "lab-ring-box" },
      refs.ringHost,
      h(
        "dl",
        { class: "lab-ring-meta" },
        h("div", {}, h("dt", {}, "Elapsed"), h("dd", {}, refs.elapsed)),
        h("div", {}, h("dt", {}, "Time left"), h("dd", {}, refs.eta)),
      ),
    );

    refs.steps = {};
    const stepper = h("ol", { class: "stepper lab-stepper", "aria-label": "Pipeline stages" });
    for (const st of STEPS) {
      const iconEl = h("span", { class: "step-icon", "aria-hidden": "true" }, h("span", { class: "lab-step-idx" }, String(STEPS.indexOf(st) + 1)));
      const fill = h("div", { class: "progress-fill" });
      const bar = h("div", { class: "progress thin", role: "progressbar", "aria-label": `${st.label} progress`, "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": "0" }, fill);
      const detail = h("div", { class: "step-detail" }, st.idle);
      const pctEl = h("div", { class: "step-pct" }, "0%");
      const stateText = h("span", { class: "sr-only" }, "pending");
      const shareEl = h("span", { class: "lab-step-share faint" }, `${st.share}% of the work`);
      const li = h(
        "li",
        { class: "step", "data-state": "pending" },
        iconEl,
        h("div", { class: "step-body" }, h("div", { class: "step-label" }, st.label, shareEl, stateText), detail, bar),
        pctEl,
      );
      refs.steps[st.id] = { li, iconEl, fill, bar, detail, pctEl, stateText, shareEl, state: "pending", progress: 0 };
      stepper.appendChild(li);
    }

    refs.liveStatus = h("p", { class: "lab-live small muted", role: "status", "aria-live": "polite" }, "Ready. Press Run to start.");
    refs.runError = h("div", { hidden: true });

    // LORO fold grid
    refs.foldGrid = h("div", { class: "lab-fold-grid" });
    refs.foldApHost = h("div", {});
    refs.loroBox = h(
      "div",
      { class: "lab-loro-live", hidden: true },
      h("div", { class: "lab-sub-title" }, "Folds", h("span", { class: "faint tiny lab-sub-note" }, "each testbed held out once")),
      refs.foldGrid,
      refs.foldApHost,
    );

    // live training curves (two charts: never one dual-axis chart)
    refs.lossHost = h("div", {});
    refs.apHost = h("div", {});
    refs.curveSub = h("span", { class: "faint tiny lab-sub-note" });
    refs.lossChart = lineChart(refs.lossHost, {
      series: [{ id: "loss", label: "Training loss", color: "train", points: [] }],
      xLabel: "Epoch",
      yLabel: "Loss",
      height: 210,
      emptyText: "Waiting for the first epoch…",
      yFormat: (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2)),
      ariaLabel: "Training loss per epoch",
    });
    refs.apTitle = h("span", {}, "Validation AP per epoch");
    refs.apNote = h("span", { class: "faint tiny" }, "higher is better");
    refs.apChart = lineChart(refs.apHost, {
      series: [{ id: "ap", label: "Validation AP", color: "val", points: [] }],
      xLabel: "Epoch",
      yLabel: "AP",
      height: 210,
      emptyText: "Waiting for the first epoch…",
      yFormat: (v) => v.toFixed(3),
      ariaLabel: "Validation average precision per epoch",
    });
    refs.curves = h(
      "div",
      { class: "lab-curves", hidden: true },
      h("div", { class: "lab-sub-title" }, "Learning curves", refs.curveSub),
      h(
        "div",
        { class: "grid-2" },
        h("div", { class: "lab-chart-box" }, h("div", { class: "lab-chart-title" }, "Training loss per epoch", h("span", { class: "faint tiny" }, "lower is better")), refs.lossHost),
        h("div", { class: "lab-chart-box" }, h("div", { class: "lab-chart-title" }, refs.apTitle, refs.apNote), refs.apHost),
      ),
    );

    card.append(h("div", { class: "lab-progress-top" }, ringBox, stepper), refs.liveStatus, refs.runError, refs.loroBox, refs.curves);
    return card;
  }

  /** Mirror the overall % on the ring's progressbar (valuetext for a cancelled run). */
  function setRingAria(pct, text = null) {
    if (!refs.ringHost) return;
    refs.ringHost.setAttribute("aria-valuenow", String(clampNum(Math.floor(pct), 0, 100)));
    if (text) refs.ringHost.setAttribute("aria-valuetext", text);
    else refs.ringHost.removeAttribute("aria-valuetext");
  }

  /** The "x% of the work" labels, from the pipeline's own per-model stage weights. */
  function applyStageShares(weights) {
    if (!refs.steps || !weights) return;
    const key = JSON.stringify(weights);
    if (refs.sharesKey === key) return;
    refs.sharesKey = key;
    for (const st of STEPS) {
      const w = weights[st.id];
      if (!refs.steps[st.id] || !isNum(w)) continue;
      refs.steps[st.id].shareEl.textContent = w === 0 ? "no work for this model" : `${w}% of the work`;
    }
  }

  function setRunState(state) {
    const map = {
      idle: ["neutral", "clock", "Idle"],
      running: ["info", "cpu", "Running"],
      done: ["good", "check", "Done"],
      cancelled: ["warn", "stop", "Cancelled"],
      error: ["bad", "x", "Error"],
    };
    const [cls, ic, text] = map[state] || map.idle;
    refs.runStatus.replaceChildren(pill(cls, ic, text, { class: `pill ${cls} lab-status-pill${state === "running" ? " is-running" : ""}` }));
    updateRunButton();
  }

  function setStep(id, state, progress, detail) {
    const st = refs.steps[id];
    if (!st) return;
    const p = clampNum(isNum(progress) ? progress : st.progress, 0, 1);
    const prevState = st.state;
    // 'partial': a LORO stage that finished one fold but has more folds to go
    const domState = state === "partial" ? "pending" : state;
    st.state = state;
    st.progress = p;
    st.li.dataset.state = domState;
    st.li.classList.toggle("is-partial", state === "partial");
    st.bar.classList.toggle("is-running", state === "running");
    st.bar.classList.toggle("is-done", state === "done");
    st.bar.classList.toggle("is-error", state === "error");
    st.bar.classList.toggle("is-skipped", state === "skipped");
    st.fill.style.width = `${(state === "done" ? 1 : p) * 100}%`;
    const pctNow = Math.floor((state === "done" ? 1 : p) * 100);
    st.pctEl.textContent = state === "skipped" ? "Skipped" : state === "error" ? "Error" : `${pctNow}%`;
    st.bar.setAttribute("aria-valuenow", String(pctNow));
    if (state === "skipped" || state === "error") st.bar.setAttribute("aria-valuetext", state === "skipped" ? "Skipped" : "Error");
    else st.bar.removeAttribute("aria-valuetext");
    st.stateText.textContent = state === "partial" ? "in progress" : state;
    if (detail) st.detail.textContent = detail;
    st.detail.title = st.detail.textContent;
    if (prevState !== state && !(state === "partial" && (prevState === "running" || prevState === "pending"))) {
      st.iconEl.innerHTML =
        state === "done" ? icon("check") : state === "error" ? icon("x") : state === "skipped" ? icon("arrowRight") : `<span class="lab-step-idx">${STEPS.findIndex((x) => x.id === id) + 1}</span>`;
      if (state === "done") {
        st.iconEl.classList.remove("pop");
        void st.iconEl.offsetWidth;
        st.iconEl.classList.add("pop");
      }
    }
  }

  function resetProgressUi(config) {
    for (const st of STEPS) setStep(st.id, "pending", 0, st.idle);
    refs.ring.update({ value: 0, label: "0%", sublabel: "overall", color: "--accent" });
    setRingAria(0);
    refs.elapsed.textContent = "0.0 s";
    refs.eta.textContent = "—";
    refs.runError.hidden = true;
    refs.runError.replaceChildren();
    if (!config) {
      refs.curves.hidden = true;
      refs.loroBox.hidden = true;
      refs.liveStatus.textContent = "Ready. Press Run to start.";
      return;
    }
    const kind = config.model.kind;
    const isLoro = config.split.mode === "loro";
    applyStageShares(stageWeights(kind));
    const xLabel = kind === "logreg" ? "Step" : "Epoch";
    const empty =
      kind === "pretrained"
        ? "No training step: the pretrained model is used as-is"
        : kind === "knn"
          ? "No learning curve: the memoriser has nothing to fit"
          : "Waiting for the first epoch…";
    refs.apTitle.textContent = `Validation AP per ${kind === "logreg" ? "step" : "epoch"}`;
    refs.apNote.textContent = "higher is better";
    refs.lossChart.update({ series: [{ id: "loss", label: "Training loss", color: "train", points: [] }], markers: [], xLabel, emptyText: empty });
    refs.apChart.update({ series: [{ id: "ap", label: "Validation AP", color: "val", points: [] }], markers: [], xLabel, emptyText: empty, yLabel: "AP" });
    refs.curveSub.textContent =
      kind === "logreg" ? "one point per full-batch step" : kind === "pretrained" || kind === "knn" ? "not trained in this run" : "one point per epoch";
    refs.curves.hidden = false;
    refs.curves.classList.remove("reveal");
    void refs.curves.offsetWidth;
    refs.curves.classList.add("reveal");
    refs.loroBox.hidden = !isLoro;
    if (isLoro) buildFoldGrid(config);
    refs.liveStatus.textContent = "Starting…";
  }

  function buildFoldGrid(config) {
    const ids = config.split.replicas || replicaIds;
    refs.foldCells = {};
    refs.foldGrid.replaceChildren();
    ids.forEach((id, i) => {
      const ap = h("div", { class: "lab-fold-ap tabular" }, "—");
      const st = h("div", { class: "lab-fold-state tiny" }, "Waiting");
      const cell = h("div", { class: "lab-fold", "data-state": "pending", style: { "--i": String(i) } }, h("div", { class: "lab-fold-name" }, id), ap, st);
      refs.foldCells[id] = { cell, ap, st };
      refs.foldGrid.appendChild(cell);
    });
    if (refs.foldAp) refs.foldAp.destroy();
    refs.foldBars = ids.map((id) => ({ label: id, value: null }));
    refs.foldAp = withTableToggle(refs.foldApHost, {
      title: "Average precision per held-out replica",
      render: (host) =>
        barChart(host, {
          bars: refs.foldBars,
          yDomain: [0, 1],
          yFormat: (v) => v.toFixed(v === 0 || v === 1 ? 1 : 3),
          height: 220,
          valueLabels: "ends",
          valueName: "AP",
          flaggedLabel: "Low support (fewer than 20 attack windows; left out of the mean)",
          ariaLabel: "Average precision per held-out replica",
        }),
      table: { columns: foldColumns(), rows: [] },
    });
  }

  function foldColumns() {
    return [
      { key: "heldOut", label: "Held out" },
      { key: "n", label: "Test windows", num: true, format: (v) => fmt.int(v) },
      { key: "nPositive", label: "Attack windows", num: true, format: (v) => fmt.int(v) },
      { key: "ap", label: "AP", num: true, format: f4 },
      { key: "rocAuc", label: "ROC-AUC", num: true, format: f4 },
      { key: "mcc", label: "MCC @ 0.5", num: true, format: f3 },
      { key: "note", label: "Note" },
    ];
  }

  function foldRow(f) {
    return { ...f, note: f.lowSupport ? "Low support: shown, excluded from the mean" : "", _flagged: !!f.lowSupport };
  }

  /* ---------------------------------------------------------------- run lifecycle */

  function startRun() {
    if (S.run && S.run.active) {
      if (S.run.exp) toast("The split experiment is running. Wait for it to finish or cancel it first.", { kind: "warn" });
      return;
    }
    if (!S.preview || S.preview.errors.length || S.preview.key !== previewKey()) {
      computePreview();
      if (S.preview.errors.length) {
        toast("Fix the split settings first.", { kind: "warn" });
        return;
      }
    }
    const config = pipelineConfigFor(ui, replicaIds);
    const k = ++S.runCounter;
    const id = `lab-${k}-${Date.now().toString(36)}`;
    S.run = {
      id,
      k,
      active: true,
      cancelling: false,
      config,
      strategy: ui.strategy,
      t0: performance.now(),
      loss: [],
      ap: [],
      curveFold: -1,
      trainDone: false,
      earlyStopAt: null,
      lastRingAt: 0,
      foldsDone: [],
    };
    resetProgressUi(config);
    setRunState("running");
    refs.elapsedTimer = setInterval(tickElapsed, 200);
    scrollToCard("lab-progress");
    dispatchRun(S.run);
  }

  function tickElapsed() {
    if (!S.run || !S.run.active) return;
    refs.elapsed.textContent = fmt.ms(performance.now() - S.run.t0);
  }

  function cancelRun() {
    const run = S.run;
    if (!run || !run.active || run.cancelling) return;
    // a Cancel within CANCEL_ARM_MS of the start is the tail of a double-click (or a
    // key repeat), not a decision: ignore it
    if (cancelArming() > 0) return;
    run.cancelling = true;
    if (run.exp) S.exp.cancelRequested = true;
    updateRunButton();
    if (run.exp) expCancelling(run);
    else refs.liveStatus.textContent = "Cancelling…";
    if (run.viaWorker && S.worker) S.worker.postMessage({ type: "cancel", runId: run.id });
    run.cancelFlag = true;
    // safety net: if the worker does not answer, drop it and end the run here
    run.cancelTimer = setTimeout(() => {
      if (S.run === run && run.active) {
        if (run.viaWorker && S.worker) {
          S.worker.terminate();
          S.worker = null;
          S.workerReady = false;
        }
        finishRun(run, "cancelled");
      }
    }, 5000);
  }

  function ensureWorker() {
    if (S.worker || S.workerFailed) return S.worker;
    try {
      const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
      w.addEventListener("message", (e) => onWorkerMessage(e.data));
      w.addEventListener("error", (e) => {
        e.preventDefault?.();
        onWorkerFailure(e.message || "the background worker failed to start");
      });
      S.worker = w;
      S.workerReady = false;
      S.readyTimer = setTimeout(() => {
        if (!S.workerReady) onWorkerFailure("the background worker did not start in time");
      }, 8000);
    } catch (err) {
      S.workerFailed = true;
      S.worker = null;
    }
    return S.worker;
  }

  function onWorkerFailure(reason) {
    const wasReady = S.workerReady;
    clearTimeout(S.readyTimer);
    if (S.worker) S.worker.terminate();
    S.worker = null;
    S.workerReady = false;
    const run = S.run;
    if (!wasReady) {
      // module workers unsupported or blocked: run on the main thread from now on
      S.workerFailed = true;
      console.warn(`Training Lab: ${reason}; running on the main thread instead.`);
      if (run && run.active && run.viaWorker && !run.gotEvent) {
        run.viaWorker = false;
        S.pendingPost = null;
        runOnMainThread(run);
      }
      return;
    }
    if (run && run.active && run.viaWorker) {
      showRunError(run, `The background worker stopped unexpectedly (${reason}).`);
    }
  }

  function dispatchRun(run) {
    const w = ensureWorker();
    if (!w) {
      run.viaWorker = false;
      runOnMainThread(run);
      return;
    }
    run.viaWorker = true;
    const msg = { type: "run", runId: run.id, config: run.config };
    if (S.workerReady) w.postMessage(msg);
    else S.pendingPost = msg;
  }

  function onWorkerMessage(msg) {
    if (!msg) return;
    if (msg.type === "ready") {
      S.workerReady = true;
      clearTimeout(S.readyTimer);
      if (S.pendingPost && S.worker) {
        const m = S.pendingPost;
        S.pendingPost = null;
        if (S.run && S.run.active && S.run.id === m.runId && !S.run.cancelFlag) S.worker.postMessage(m);
        else if (S.run && S.run.id === m.runId && S.run.cancelFlag) finishRun(S.run, "cancelled");
      }
      return;
    }
    handleEvent(msg);
  }

  async function runOnMainThread(run) {
    if (!run.exp) refs.liveStatus.textContent = "Running on the main thread (background workers are unavailable in this browser)…";
    try {
      // The pretrained weights are only needed by the 'pretrained' model and the
      // optional fidelity number: without them every other model still runs (the
      // pipeline rejects 'pretrained' with a clear message).
      const [{ runPipeline }, weights] = await Promise.all([
        import("./pipeline.js"),
        Promise.resolve()
          .then(() => store.loadPretrainedWeights())
          .catch(() => null),
      ]);
      if (S.run !== run || !run.active) return;
      // Web Crypto exists only in a secure context (https:// or localhost); without
      // it the pipeline skips the ledger stage and keeps the results.
      const subtle = globalThis.crypto && globalThis.crypto.subtle;
      const sha256 = subtle ? async (bytes) => new Uint8Array(await subtle.digest("SHA-256", bytes)) : null;
      await runPipeline(run.config, {
        stats,
        pretrainedWeights: weights,
        runId: run.id,
        sha256,
        shouldCancel: () => !!run.cancelFlag,
        onEvent: handleEvent,
      });
    } catch (err) {
      if (S.run === run && run.active && !err?.pipelineReported) showRunError(run, err && err.message ? err.message : String(err));
    }
  }

  function handleEvent(ev) {
    const run = S.run;
    if (!ev || !run || ev.runId !== run.id || !run.active) return; // stale run: ignore
    run.gotEvent = true;
    if (run.exp) {
      expEvent(run, ev);
      return;
    }
    switch (ev.type) {
      case "plan":
        applyStageShares(ev.weights);
        break;
      case "stage":
        onStage(run, ev);
        break;
      case "progress":
        onProgress(run, ev);
        break;
      case "epoch":
        onEpoch(run, ev);
        break;
      case "split":
        if (ev.foldIndex != null) markFold(run, ev.foldIndex, "running");
        break;
      case "fold":
        onFold(run, ev);
        break;
      case "result":
        if (ev.result && ev.result.cancelled) finishRun(run, "cancelled");
        else finishRun(run, "done", ev.result);
        break;
      case "error":
        showRunError(run, ev.message);
        break;
      default:
        break;
    }
  }

  function onStage(run, ev) {
    if (ev.weights) applyStageShares(ev.weights);
    // In LORO the per-fold stages report cumulative progress; a stage is only
    // really done after the last fold.
    const partial = ev.status === "done" && isNum(ev.progress) && ev.progress < 0.999;
    setStep(ev.stage, partial ? "partial" : ev.status, ev.progress, ev.detail);
    if (ev.status === "running" && ev.foldIndex != null && ev.stage === "split") markFold(run, ev.foldIndex, "running");
    if (ev.status === "running" || ev.status === "done" || ev.status === "skipped") {
      const label = STEPS.find((s) => s.id === ev.stage)?.label || ev.stage;
      if (ev.status !== "running" || ev.stage !== "train" || !run.announcedTrain) {
        refs.liveStatus.textContent = `${label}: ${ev.detail || ev.status}`;
        if (ev.stage === "train" && ev.status === "running") run.announcedTrain = true;
      }
    }
    if (ev.stage === "train" && ev.status === "done") {
      run.announcedTrain = false;
      const planned = run.config.model.kind === "tessera" ? run.config.model.epochs : null;
      if (planned && run.loss.length > 0 && run.loss.length < planned) run.earlyStopAt = run.loss[run.loss.length - 1].x;
      flushCurves(run, true);
    }
  }

  function onProgress(run, ev) {
    const now = performance.now();
    run.overall = ev.overall;
    if (now - run.lastRingAt > 220 || ev.overall >= 1) {
      run.lastRingAt = now;
      refs.ring.update({ value: ev.overall, label: `${Math.floor(ev.overall * 100)}%` });
      setRingAria(ev.overall * 100);
    }
    refs.eta.textContent = ev.etaMs == null ? "estimating…" : ev.etaMs <= 0 ? "—" : `about ${fmt.ms(ev.etaMs)}`;
  }

  function onEpoch(run, ev) {
    const fold = ev.foldIndex ?? 0;
    if (fold !== run.curveFold) {
      run.curveFold = fold;
      run.loss = [];
      run.ap = [];
      run.earlyStopAt = null;
      if (ev.heldOut) refs.curveSub.textContent = `fold ${fold + 1} of ${ev.nFolds} · ${ev.heldOut} held out`;
    }
    if (isNum(ev.trainLoss)) run.loss.push({ x: ev.epoch, y: ev.trainLoss });
    if (isNum(ev.valAp)) run.ap.push({ x: ev.epoch, y: ev.valAp });
    else if (isNum(ev.valLoss)) {
      // validation is all one class, so AP is undefined: early stopping watches the validation loss
      if (run.valMode !== "loss") {
        run.valMode = "loss";
        refs.apTitle.textContent = `Validation loss per ${run.config.model.kind === "logreg" ? "step" : "epoch"}`;
        refs.apNote.textContent = "AP undefined (one-class validation) · lower is better";
      }
      run.ap.push({ x: ev.epoch, y: ev.valLoss });
    }
    flushCurves(run, false);
  }

  function flushCurves(run, now) {
    if (run.curveTimer && !now) return;
    const go = () => {
      run.curveTimer = 0;
      if (S.run !== run) return;
      const markers = [];
      const byLoss = run.valMode === "loss";
      if (run.ap.length) {
        let best = run.ap[0];
        for (const p of run.ap) if (byLoss ? p.y < best.y : p.y > best.y) best = p;
        markers.push({ x: best.x, label: "Best" });
      }
      if (run.earlyStopAt != null) markers.push({ x: run.earlyStopAt, label: "Early stop" });
      const epochs = run.config.model.kind === "logreg" ? LOGREG_STEPS : run.config.model.epochs;
      refs.lossChart.update({ series: [{ id: "loss", label: "Training loss", color: "train", points: run.loss.slice() }], markers, xDomain: [1, Math.max(2, epochs)] });
      refs.apChart.update({
        series: [{ id: "ap", label: byLoss ? "Validation loss" : "Validation AP", color: "val", points: run.ap.slice() }],
        markers,
        xDomain: [1, Math.max(2, epochs)],
        yLabel: byLoss ? "Loss" : "AP",
      });
    };
    clearTimeout(run.curveTimer);
    if (now) go();
    else run.curveTimer = setTimeout(go, 140);
  }

  function markFold(run, foldIndex, state) {
    const ids = run.config.split.replicas || replicaIds;
    const id = ids[foldIndex];
    const c = refs.foldCells && refs.foldCells[id];
    if (!c || c.cell.dataset.state === "done") return;
    c.cell.dataset.state = state;
    c.st.textContent = state === "running" ? "Running…" : c.st.textContent;
  }

  function onFold(run, ev) {
    const c = refs.foldCells && refs.foldCells[ev.heldOut];
    if (c) {
      c.cell.dataset.state = "done";
      c.cell.classList.add("pop");
      tweenNumber(c.ap, ev.ap, (v) => v.toFixed(3), 600);
      c.st.replaceChildren(ev.lowSupport ? pill("warn", "alert", "Low support") : h("span", { class: "ok-ink" }, `${plural(ev.nPositive, "attack")}`));
    }
    run.foldsDone.push({ foldIndex: ev.foldIndex, heldOut: ev.heldOut, n: ev.n, nPositive: ev.nPositive, lowSupport: ev.lowSupport, ap: ev.ap, rocAuc: ev.rocAuc, mcc: ev.mcc });
    const i = refs.foldBars.findIndex((b) => b.label === ev.heldOut);
    if (i >= 0) refs.foldBars[i] = { label: ev.heldOut, value: ev.ap, flagged: ev.lowSupport, note: ev.lowSupport ? "Low support" : "" };
    refs.foldAp.chart.update({ bars: refs.foldBars.slice() });
    refs.foldAp.update({ table: { rows: run.foldsDone.map(foldRow) } });
  }

  function showRunError(run, message) {
    if (S.run !== run || !run.active) return;
    if (run.exp) {
      expFinish(run, "error", null, message);
      return;
    }
    refs.runError.hidden = false;
    refs.runError.replaceChildren(notice("bad", "alert", h("strong", {}, "The run stopped with an error. "), message || "Unknown error."));
    finishRun(run, "error");
  }

  function finishRun(run, status, result = null) {
    if (!run.active) return;
    if (run.exp) {
      expFinish(run, status, result);
      return;
    }
    run.active = false;
    clearInterval(refs.elapsedTimer);
    clearTimeout(run.cancelTimer);
    clearTimeout(run.curveTimer);
    refs.elapsed.textContent = fmt.ms(performance.now() - run.t0);
    refs.eta.textContent = "—";
    if (status === "done" && result) {
      refs.ring.update({ value: 1, label: "100%", sublabel: "done", color: "--good" });
      setRingAria(100, "Done");
      for (const st of STEPS) if (refs.steps[st.id].state === "running") setStep(st.id, "done", 1);
      refs.liveStatus.textContent = `Finished in ${fmt.ms(result.elapsedMs)}. Results are below.${result.ledgerStatus === "skipped-no-crypto" ? " The ledger was skipped: this page has no Web Crypto (it needs https:// or localhost)." : ""}`;
      setRunState("done");
      const entry = addHistory(run, result);
      renderResults(result, entry, { reveal: true });
      renderHistory();
      setTimeout(() => {
        // only if the viewer has not moved on to another run in the meantime
        if (!(S.run && S.run.active)) scrollToCard("lab-results");
      }, reducedMotion() ? 0 : 450);
    } else if (status === "cancelled") {
      for (const st of STEPS) {
        const s0 = refs.steps[st.id];
        if (s0.state === "running" || s0.state === "pending") setStep(st.id, "skipped", s0.progress, s0.state === "running" ? "Cancelled" : "Not run (cancelled)");
      }
      refs.ring.update({ sublabel: "cancelled" });
      setRingAria(Math.floor((run.overall || 0) * 100), "Cancelled");
      refs.liveStatus.textContent = "Run cancelled. Nothing was kept.";
      setRunState("cancelled");
      toast("Run cancelled.", { kind: "warn" });
    } else {
      for (const st of STEPS) if (refs.steps[st.id].state === "running") setStep(st.id, "error", refs.steps[st.id].progress);
      setRingAria(Math.floor((run.overall || 0) * 100), "Stopped with an error");
      refs.liveStatus.textContent = "The run stopped with an error.";
      setRunState("error");
    }
    updateRunButton();
  }

  /* ================================================================ 5 results */

  function buildResultsCard() {
    refs.resultsRight = h("div", { class: "row" }, h("span", { class: "tag-synthetic" }, "Synthetic test windows"));
    const card = stepCard(5, "Results", "How well the model separates attack windows from benign ones on the test set.", { id: "lab-results", tour: "lab-results", right: refs.resultsRight });
    refs.resultsBody = h(
      "div",
      { class: "lab-results-body" },
      h("div", { class: "lab-empty" }, iconSpan("chart", "lab-empty-ico"), h("p", {}, "Run the pipeline to see average precision, the confusion matrix, PR and ROC curves and more."), h("p", { class: "tiny faint" }, "Tip: try the Random split next to the default and compare them in the run history.")),
    );
    card.appendChild(refs.resultsBody);
    return card;
  }

  function destroyResultCharts() {
    while (resultCharts.length) {
      const c = resultCharts.pop();
      try {
        c.destroy();
      } catch {
        /* already gone */
      }
    }
  }

  function renderResults(result, entry, { reveal = true } = {}) {
    if (!result || !result.metrics) return;
    destroyResultCharts();
    S.lastShown = { result, entry };
    const isLoro = result.mode === "loro";
    S.threshold = 0.5;
    const body = refs.resultsBody;
    body.replaceChildren();
    refs.resultsRight.replaceChildren(h("span", { class: "badge" }, `Run #${entry.k}`), h("span", { class: "tag-synthetic" }, "Synthetic test windows"));

    const blocks = [
      resultsHeadline(result, entry),
      resultsThreshold(result),
      isLoro ? resultsLoro(result) : null,
      resultsBreakdown(result),
      resultsAttribution(result),
      resultsChecks(result, entry),
      resultsActions(result, entry),
    ].filter(Boolean);
    body.append(...blocks);
    if (reveal && !reducedMotion()) {
      blocks.forEach((b, i) => {
        b.classList.add("reveal");
        b.style.setProperty("--i", String(i));
      });
    }
    applyThreshold(S.threshold, "init");
  }

  function resultsHeadline(result, entry) {
    const isLoro = result.mode === "loro";
    const m = result.metrics;
    const ls = result.loroSummary;
    const apVal = isLoro ? ls?.ap?.mean : m.ap;
    const heroValue = h("div", { class: "hero-number tabular lab-hero-ap" }, "—");
    const sub = [];
    if (isLoro) {
      sub.push(
        h(
          "div",
          { class: "stat-sub" },
          `Mean over ${ls.ap.n} folds ± ${f4(ls.ap.std)} (std)`,
          ls.excludedLowSupport.length ? ` · ${ls.excludedLowSupport.join(", ")} left out (fewer than ${ls.minSupport ?? MIN_SUPPORT_FOR_RATES} attack windows)` : "",
        ),
      );
    } else {
      sub.push(h("div", { class: "stat-sub" }, `${fmt.int(m.n)} test windows · ${fmt.int(m.nPositive)} attacks · no-skill AP = prevalence ${f3(m.prevalence)}`));
      if (m.ap == null) sub.push(h("div", { class: "stat-sub warn-ink" }, "AP is undefined: the test set has only one class."));
      else if (m.lowSupport) sub.push(h("div", { class: "stat-sub warn-ink" }, `Low support: only ${m.nPositive} attack windows, so this number is noisy.`));
    }
    const hero = h(
      "div",
      { class: "stat primary lab-hero" },
      h("div", { class: "stat-label" }, "Average precision (AP)", h("span", { class: "pill info lab-primary-pill" }, "Primary metric")),
      heroValue,
      ...sub,
      comparisonLine(result),
    );
    requestAnimationFrame(() => tweenNumber(heroValue, apVal, (v) => v.toFixed(4), 1100));

    // In LORO the tiles (and everything in the threshold panel) pool the test
    // windows of the scoreable folds, scored by one model per fold; the hero is
    // the mean over those folds. Every pooled number says so.
    const pooled = isLoro ? "pooled · " : "";
    const tile = (key, label, sub2, cls = "") => {
      const value = h("div", { class: "stat-value tabular" }, "—");
      const el = h("div", { class: `stat lab-tile ${cls}` }, h("div", { class: "stat-label" }, label), value, h("div", { class: "stat-sub" }, `${pooled}${sub2}`));
      return { key, el, value };
    };
    refs.tiles = {
      rocAuc: tile("rocAuc", "ROC-AUC", "ranking quality, threshold-free"),
      mcc: tile("mcc", "MCC", "Matthews correlation · second metric"),
      f1: tile("f1", "F1", "balance of precision and recall"),
      precision: tile("precision", "Precision", "flagged windows that were attacks"),
      recall: tile("recall", "Recall", "attack windows that were caught"),
      accuracy: tile("accuracy", "Accuracy", "misleading at this prevalence: flagging nothing already scores high", "lab-tile-muted"),
    };
    tweenNumber(refs.tiles.rocAuc.value, m.rocAuc, (v) => v.toFixed(4));
    const tiles = h("div", { class: "lab-tiles" }, ...Object.values(refs.tiles).map((t) => t.el));
    const kids = [hero, tiles];
    if (isLoro) {
      const sc = result.metricsScope;
      const excluded = sc?.excluded?.length ? sc.excluded : ls?.excludedLowSupport || [];
      kids.push(
        h(
          "p",
          { class: "lab-pooled-note small" },
          iconSpan("info"),
          h(
            "span",
            {},
            h("b", {}, "Two different summaries. "),
            `The headline is the mean of the per-fold AP over ${ls.ap.n} scoreable folds. The tiles, the threshold panel, the confusion matrix and the curves below instead pool ${isNum(sc?.n) ? `the ${fmt.int(sc.n)} test windows` : "the test windows"} of those folds, each scored by its own fold's model (${isNum(sc?.nModels) ? plural(sc.nModels, "model") : "one model per fold"}), so their numbers differ from the mean.`,
            excluded.length ? ` ${joinIds(excluded)} ${excluded.length === 1 ? "is" : "are"} left out of both (fewer than ${ls.minSupport ?? MIN_SUPPORT_FOR_RATES} attack windows).` : "",
          ),
        ),
      );
    }
    return h("div", { class: "lab-res-head" }, ...kids);
  }

  function comparisonLine(result) {
    const rr = realResults;
    const isLoro = result.mode === "loro";
    if (isLoro && rr?.loro?.summary) {
      const s0 = rr.loro.summary;
      const by = recordedBy(rr.loro.source);
      return h(
        "div",
        { class: "lab-compare" },
        h("span", { class: "tag-real" }, "Real"),
        h(
          "span",
          {},
          `Real leave-one-replica-out (${rr.loro.model || "LightGBM"}) on real windows: mean AP ${f4(s0.average_precision.mean)} ± ${f4(s0.average_precision.std)}`,
          s0.excluded_low_support?.length ? `, ${s0.excluded_low_support.join(", ")} left out` : "",
          ".",
        ),
        by ? h("span", { class: "tiny faint lab-prov", title: rr.loro.source }, `Recorded by ${by}`) : null,
      );
    }
    const ld = rr?.leakage_duplicates;
    if ((result.features || result.config.features) === "m2" && (result.mode === "random" || result.mode === "chronological") && ld && isNum(ld.r0_random_ap)) {
      const by = recordedBy(ld.source);
      return h(
        "div",
        { class: "lab-compare" },
        h("span", { class: "tag-real" }, "Real"),
        h(
          "span",
          {},
          `The closest real counterpart (${ld.replica} only, ${/LightGBM/.test(ld.source || "") ? "LightGBM" : "the recorded model"}, ${ld.features || "network metrics"}, chronological split with a 600 s gap): ${result.mode === "random" ? "random" : "chronological"} split AP ${f4(result.mode === "random" ? ld.r0_random_ap : ld.r1_chronological_ap)} (random ${f4(ld.r0_random_ap)} vs chronological ${f4(ld.r1_chronological_ap)}). Different model, replicas and split sizes: compare the direction of the gap, not the numbers.`,
        ),
        by ? h("span", { class: "tiny faint lab-prov", title: ld.source }, `Recorded by ${by}`) : null,
      );
    }
    if (isDefaultProtocol(result.config.split) && rr?.tessera_base) {
      const tb = rr.tessera_base;
      const by = recordedBy(tb.source);
      return h(
        "div",
        { class: "lab-compare" },
        h("span", { class: "tag-real" }, "Real"),
        h("span", {}, `For reference, real TESSERA-base on the real held-out ${tb.held_out} windows: AP ${f4(tb.tessera_ap)} (MCC ${f4(tb.tessera_mcc)}).`),
        by ? h("span", { class: "tiny faint lab-prov", title: tb.source }, `Recorded by ${by}`) : null,
      );
    }
    return h("div", { class: "lab-compare" }, iconSpan("info"), h("span", {}, "This split has no real-data counterpart. The real held-out numbers are in ", h("a", { href: "#results" }, "Real Results"), "."));
  }

  function resultsThreshold(result) {
    // the rows result.metrics counts (in LORO: the scoreable folds only), so the
    // threshold panel, the curves and the histogram all describe the same windows
    const test = scoredRows(result);
    const m = result.metrics;
    const pooled = pooledLabel(result);
    const wrap = h("section", { class: "lab-res-block" });
    const thrSlider = slider({
      id: "lab-threshold",
      label: "Decision threshold (flag a window as an attack when its score is at least this)",
      min: 0.01,
      max: 0.99,
      step: 0.01,
      value: S.threshold,
      format: (v) => v.toFixed(2),
      onInput: (v) => scheduleThreshold(v, "slider"),
    });
    refs.thrSlider = thrSlider;
    const resetThr = h("button", { type: "button", class: "btn sm ghost", onClick: () => scheduleThreshold(0.5, "reset") }, "Back to 0.50");
    refs.deploy = h("p", { class: "lab-deploy" });
    const top = h("div", { class: "lab-thr-row" }, thrSlider.el, resetThr);

    // confusion matrix + histogram
    const cmHost = h("div", {});
    refs.cm = confusionMatrix(cmHost, { tp: 0, fp: 0, tn: 0, fn: 0, labels: { pos: "Attack", neg: "Benign" }, ariaLabel: "Confusion matrix at the current threshold" });
    resultCharts.push(refs.cm);
    // the box already has a visible title, so the toggle's own title is kept for
    // the table caption only (lab-toggle-quiet hides it on screen)
    const histHost = h("div", { class: "lab-toggle-quiet" });
    const hist = m.histogram;
    const histToggle = withTableToggle(histHost, {
      title: "Score distribution by true class",
      render: (host) =>
        histogramChart(host, {
          edges: hist.edges,
          series: [
            { label: "Benign", color: "benign", counts: hist.benign },
            { label: "Attack", color: "attack", counts: hist.attack },
          ],
          threshold: S.threshold,
          onThreshold: (v) => scheduleThreshold(v, "hist"),
          xLabel: "Attack score",
          logY: true,
          height: 230,
          ariaLabel: "Histogram of attack scores for benign and attack test windows, with the draggable threshold",
        }),
      table: {
        columns: [
          { key: "bin", label: "Score bin" },
          { key: "benign", label: "Benign", num: true, format: (v) => fmt.int(v) },
          { key: "attack", label: "Attack", num: true, format: (v) => fmt.int(v) },
        ],
        rows: hist.benign.map((b, i) => ({ bin: `${hist.edges[i].toFixed(2)}–${hist.edges[i + 1].toFixed(2)}`, benign: b, attack: hist.attack[i] })),
      },
    });
    refs.hist = histToggle.chart;
    resultCharts.push(histToggle);

    // PR + ROC
    const prHost = h("div", {});
    const rocHost = h("div", {});
    const label = `${MODEL[result.model.kind]?.short || "Model"}${pooled ? " (pooled folds)" : ""}`;
    refs.pr = null;
    refs.roc = null;
    if (m.pr && m.roc) {
      refs.pr = curveChart(prHost, {
        kind: "pr",
        series: [{ label, color: "--series-1", x: Array.from(m.pr.recall), y: Array.from(m.pr.precision), auc: m.ap }],
        baseline: m.prevalence,
        height: 260,
        ariaLabel: "Precision-recall curve with the operating point at the current threshold",
      });
      refs.roc = curveChart(rocHost, {
        kind: "roc",
        series: [{ label, color: "--series-1", x: Array.from(m.roc.fpr), y: Array.from(m.roc.tpr), auc: m.rocAuc }],
        height: 260,
        ariaLabel: "ROC curve with the operating point at the current threshold",
      });
      resultCharts.push(refs.pr, refs.roc);
    } else {
      prHost.appendChild(h("p", { class: "muted small" }, "Curves need both classes in the test set."));
    }

    // In LORO the curve areas are pooled over the scoreable folds and differ
    // from the headline mean AP, so they never appear as a bare "AP".
    const prCaption = pooled ? `${pooled}: AP ${f4(m.ap)}` : `area = AP ${f4(m.ap)}`;
    const rocCaption = pooled ? `${pooled}: ROC-AUC ${f4(m.rocAuc)}` : `area = ROC-AUC ${f4(m.rocAuc)}`;
    wrap.append(
      h(
        "div",
        { class: "lab-sub-title" },
        "Choose where to draw the line",
        h("span", { class: "faint tiny lab-sub-note" }, pooled ? `everything below updates instantly · ${pooled}` : "everything below updates instantly"),
      ),
      top,
      refs.deploy,
      h(
        "div",
        { class: "grid-2 lab-res-grid" },
        h("div", { class: "lab-chart-box" }, h("div", { class: "lab-chart-title" }, "Confusion matrix", h("span", { class: "faint tiny" }, pooled ? "rows: actual · columns: predicted · pooled folds" : "rows: actual · columns: predicted")), cmHost),
        h("div", { class: "lab-chart-box" }, h("div", { class: "lab-chart-title" }, "Scores, benign vs attack", h("span", { class: "faint tiny" }, pooled ? "drag the line · pooled folds" : "drag the line")), histHost),
      ),
      h(
        "div",
        { class: "grid-2 lab-res-grid" },
        h("div", { class: "lab-chart-box" }, h("div", { class: "lab-chart-title" }, "Precision-recall curve", h("span", { class: "faint tiny" }, prCaption)), prHost),
        h("div", { class: "lab-chart-box" }, h("div", { class: "lab-chart-title" }, "ROC curve", h("span", { class: "faint tiny" }, rocCaption)), rocHost),
      ),
    );
    refs.thrTest = test;
    refs.thrPooled = pooled;
    return wrap;
  }

  function scheduleThreshold(v, source) {
    thrPending = { v, source };
    if (thrRaf) return;
    thrRaf = requestAnimationFrame(() => {
      thrRaf = 0;
      const p = thrPending;
      thrPending = null;
      if (p) applyThreshold(p.v, p.source);
    });
  }

  function applyThreshold(v, source) {
    const test = refs.thrTest;
    if (!test || !test.y || !test.scores) return;
    const thr = Math.round(clampNum(v, 0.01, 0.99) * 100) / 100;
    S.threshold = thr;
    if (source !== "slider" && refs.thrSlider) refs.thrSlider.set(thr);
    const mt = metricsAt(test.y, test.scores, thr);
    const t = refs.tiles;
    const dur = source === "init" ? 900 : 280;
    tweenNumber(t.mcc.value, mt.mcc, (x) => x.toFixed(3), dur);
    tweenNumber(t.f1.value, mt.f1, (x) => x.toFixed(3), dur);
    tweenNumber(t.precision.value, mt.precision, (x) => x.toFixed(3), dur);
    tweenNumber(t.recall.value, mt.recall, (x) => x.toFixed(3), dur);
    tweenNumber(t.accuracy.value, mt.accuracy, (x) => x.toFixed(3), dur);
    const c = mt.confusion;
    refs.cm.update({ tp: c.tp, fp: c.fp, tn: c.tn, fn: c.fn });
    const lab = `threshold ${thr.toFixed(2)}`;
    if (refs.pr) refs.pr.update({ point: { x: mt.recall, y: mt.precision, label: lab } });
    if (refs.roc) refs.roc.update({ point: { x: mt.fpr, y: mt.recall, label: lab } });
    if (source !== "hist" && refs.hist) refs.hist.update({ threshold: thr });
    // deployment consequence, with the counts behind the rate: the test set's
    // benign windows limit how small a false-alert rate it can resolve
    const fr = falseAlertRate(c.fp, c.fp + c.tn);
    const nPos = c.tp + c.fn;
    const perDay = `${fmt.int(WINDOWS_PER_DAY)} one-minute windows a day`;
    let alertPart;
    if (!fr.measurable) {
      alertPart = ["The test set has no benign windows, so the false-alert rate cannot be measured"];
    } else if (c.fp === 0) {
      alertPart = [
        "At this threshold, ",
        h("b", { class: "tabular" }, `0 of ${fmt.int(fr.benign)} benign test windows`),
        " were flagged: at most about ",
        h("b", { class: "tabular" }, `${alertCountText(fr.upper)} false alerts per host per day`),
        ` at 95% confidence (rule of three: 3 / ${fmt.int(fr.benign)} × ${perDay})`,
      ];
    } else {
      alertPart = [
        "At this threshold, a monitored host would raise about ",
        h("b", { class: "tabular" }, `${alertCountText(fr.rate)} false alert${alertCountText(fr.rate) === "1" ? "" : "s"} per day`),
        ` (${fmt.int(c.fp)} of ${fmt.int(fr.benign)} benign test windows flagged, scaled to ${perDay})`,
      ];
    }
    refs.deploy.replaceChildren(
      iconSpan("clock"),
      h(
        "span",
        {},
        ...alertPart,
        nPos > 0 ? [", and the detector would catch ", h("b", { class: "tabular" }, pct1(mt.recall)), ` of attack windows (${fmt.int(c.tp)} of ${fmt.int(nPos)}). `] : ". The test set has no attack windows, so recall is undefined. ",
        h(
          "span",
          { class: "faint" },
          `(Averaged over the test hosts${refs.thrPooled ? ` and ${refs.thrPooled}` : ""}. Synthetic benign behaviour: real false-alert rates can differ.)`,
        ),
      ),
    );
  }

  function resultsLoro(result) {
    const ls = result.loroSummary;
    const folds = result.folds || [];
    const wrap = h("section", { class: "lab-res-block" }, h("div", { class: "lab-sub-title" }, "Leave-one-replica-out folds", h("span", { class: "faint tiny lab-sub-note" }, "one experiment per held-out testbed")));
    const undefinedFolds = folds.filter((f) => f.ap == null).map((f) => f.heldOut);
    const stat = (label, value, sub, cls = "") =>
      h("div", { class: `stat ${cls}` }, h("div", { class: "stat-label" }, label), h("div", { class: "stat-value tabular" }, value), h("div", { class: "stat-sub" }, sub));
    wrap.appendChild(
      h(
        "div",
        { class: "grid-3 lab-loro-stats" },
        stat("Mean AP (honest)", `${f4(ls.ap.mean)} ± ${f4(ls.ap.std)}`, `${ls.ap.n} folds; low-support folds left out`, "primary"),
        stat("Mean MCC", `${f3(ls.mcc.mean)} ± ${f3(ls.mcc.std)}`, "same folds, threshold 0.5"),
        stat(
          "Naive mean AP, all folds",
          `${f4(ls.naiveAp.mean)} ± ${f4(ls.naiveAp.std)}`,
          undefinedFolds.length
            ? `what averaging in the low-support folds would read; ${undefinedFolds.join(", ")} ${undefinedFolds.length === 1 ? "has" : "have"} no attack windows at this data size, so AP is undefined there`
            : "what averaging in the low-support folds would read",
          "lab-tile-muted",
        ),
      ),
    );
    const chartHost = h("div", {});
    const toggle = withTableToggle(chartHost, {
      title: "AP per held-out replica",
      render: (host) =>
        barChart(host, {
          bars: folds.map((f) => ({ label: f.heldOut, value: f.ap, flagged: f.lowSupport, note: f.lowSupport ? "Low support" : "" })),
          yDomain: [0, 1],
          yFormat: (v) => v.toFixed(v === 0 || v === 1 ? 1 : 3),
          refLine: isNum(ls.ap.mean) ? { value: ls.ap.mean, label: "Mean (excl. low support)" } : null,
          height: 240,
          valueLabels: "ends",
          flaggedLabel: "Low support (fewer than 20 attack windows; left out of the mean)",
          ariaLabel: "Average precision per held-out replica",
        }),
      table: { columns: foldColumns(), rows: folds.map(foldRow) },
    });
    resultCharts.push(toggle);
    wrap.appendChild(chartHost);
    const tableHost = h("div", { class: "lab-table" });
    dataTable(tableHost, { columns: foldColumns(), rows: folds.map(foldRow), caption: "Leave-one-replica-out folds" });
    wrap.appendChild(tableHost);
    return wrap;
  }

  function resultsBreakdown(result) {
    const isLoro = result.mode === "loro";
    const wrap = h("section", { class: "lab-res-block" });
    const repHost = h("div", { class: "lab-table" });
    const hostHost = h("div", { class: "lab-table" });
    const lowNote = `Rows with fewer than ${MIN_SUPPORT_FOR_RATES} attack windows are flagged: their AP is noise.`;
    if (!isLoro) {
      dataTable(repHost, {
        columns: [
          { key: "id", label: "Replica" },
          { key: "n", label: "Test windows", num: true, format: (v) => fmt.int(v) },
          { key: "nPositive", label: "Attacks", num: true, format: (v) => fmt.int(v) },
          { key: "ap", label: "AP", num: true, format: (v) => (v == null ? "undefined" : f4(v)) },
          { key: "mcc", label: "MCC @ 0.5", num: true, format: f3 },
          { key: "note", label: "Note" },
        ],
        rows: (result.perReplica || []).map((r) => ({ ...r, note: r.lowSupport ? "Low support" : "", _flagged: r.lowSupport })),
        caption: "Results per test replica",
      });
    }
    dataTable(hostHost, {
      columns: [
        { key: "label", label: "Host" },
        { key: "n", label: "Test windows", num: true, format: (v) => fmt.int(v) },
        { key: "nPositive", label: "Attacks", num: true, format: (v) => fmt.int(v) },
        { key: "ap", label: "AP", num: true, format: (v) => (v == null ? "undefined" : f4(v)) },
        { key: "note", label: "Note" },
      ],
      rows: (result.perHost || []).map((r) => ({
        ...r,
        note: r.nPositive === 0 ? "No attacks: AP undefined" : r.nPositive < MIN_SUPPORT_FOR_RATES ? "Low support" : "",
        _flagged: r.nPositive < MIN_SUPPORT_FOR_RATES,
      })),
      caption: "Results per host",
    });
    wrap.append(
      h("div", { class: "lab-sub-title" }, "Where the score comes from", h("span", { class: "faint tiny lab-sub-note" }, lowNote)),
      h(
        "div",
        { class: isLoro ? "lab-breakdown" : "grid-2 lab-breakdown" },
        isLoro ? null : h("div", {}, h("div", { class: "lab-chart-title" }, "Per test replica"), repHost),
        h(
          "div",
          {},
          h("div", { class: "lab-chart-title" }, "Per host", h("span", { class: "faint tiny" }, isLoro ? `attacks are almost only on the firewall · ${pooledLabel(result)}` : "attacks are almost only on the firewall")),
          hostHost,
        ),
      ),
    );
    return wrap;
  }

  function resultsAttribution(result) {
    const wrap = h(
      "section",
      { class: "lab-res-block" },
      h(
        "div",
        { class: "lab-sub-title" },
        "Which telemetry the model leaned on",
        h("span", { class: "faint tiny lab-sub-note" }, result.mode === "loro" ? "mean gate weight per modality, over every fold's test windows" : "mean gate weight per modality"),
      ),
    );
    if (!result.attributionMean) {
      wrap.appendChild(
        h(
          "p",
          { class: "muted small" },
          result.model.kind === "logreg"
            ? "Logistic regression has no modality gates, so there is no attribution to show."
            : result.model.kind === "knn"
              ? "The memoriser has no modality gates: it compares whole windows, so there is no attribution to show."
              : "No attribution was produced for this run.",
        ),
      );
      return wrap;
    }
    // the section title above already names the chart: its toggle title is kept for the table caption only
    const host = h("div", { class: "lab-toggle-quiet" });
    const bars = MODALITIES.map((md, i) => ({ label: `${md.code} ${md.label}`, value: result.attributionMean[i], color: md.color }));
    const toggle = withTableToggle(host, {
      title: "Mean modality attribution",
      render: (el2) =>
        barChart(el2, {
          bars,
          yDomain: [0, 1],
          yFormat: (v) => `${Math.round(v * 100)}%`,
          height: 200,
          horizontal: true,
          valueLabels: "ends",
          ariaLabel: "Mean gate weight per modality on the test set",
        }),
      table: {
        columns: [
          { key: "label", label: "Modality" },
          { key: "value", label: "Mean gate weight", num: true, format: (v) => pct1(v) },
        ],
        rows: bars,
      },
    });
    resultCharts.push(toggle);
    const av = realResults?.attribution_vs_ablation;
    let caveat;
    if (av?.mean_attribution && Array.isArray(av.ablation)) {
      const without = av.ablation.find((a) => /without M1/i.test(a.check));
      const full = av.ablation.find((a) => /with M1/i.test(a.check));
      caveat = notice(
        "",
        "info",
        h("strong", {}, "Gate weights understate true importance. "),
        `On the real santos fold the gates gave M1 only ${Math.round(av.mean_attribution.m1_log * 100)}%`,
        without && full ? `, yet removing M1 dropped AP from ${f4(full.ap)} to ${f4(without.ap)}` : "",
        ". Treat these bars as a hint, not proof. ",
        h("span", { class: "tag-real" }, "Real"),
        " See ",
        h("a", { href: "#results" }, "Real Results"),
        ".",
      );
    } else {
      caveat = notice("", "info", h("strong", {}, "Gate weights understate true importance. "), "Treat these bars as a hint, not proof: the measured comparison with ablation is in ", h("a", { href: "#results" }, "Real Results"), ".");
    }
    wrap.append(host, caveat);
    if ((result.features || result.config?.features) === "m2") {
      wrap.appendChild(notice("", "info", "This model saw only the network metrics (M2). The log and graph sources were switched off, and the host-identity gate had zeroed inputs."));
    }
    return wrap;
  }

  function resultsChecks(result, entry) {
    const wrap = h("section", { class: "lab-res-block lab-checks" }, h("div", { class: "lab-sub-title" }, "Checks on this run"));
    const cert = result.certificate;
    if (cert) {
      const ov = OVERALL[cert.overall];
      const isSupport = (c) => (c.group ? c.group === "support" : c.id === "test-support");
      const leakChecks = cert.checks.filter((c) => !isSupport(c));
      const flagged = leakChecks.filter((c) => c.severity !== "ok");
      const sup = supportInfo(cert);
      const supCheck = cert.checks.find(isSupport);
      const m2 = (cert.featureSet || result.features || result.config?.features) === "m2";
      const cols = m2 ? "the 24 network-metric columns" : "all 42 features";
      const dupLine =
        isNum(cert.activeDuplicates) && isNum(cert.quietDuplicates)
          ? `Exact copies of a training or validation window, compared on ${cols} the model sees${result.mode === "loro" ? ", summed over the folds" : ""}: ${plural(cert.activeDuplicates, "active window")} (the kind that can leak) and ${plural(cert.quietDuplicates, "quiet window")} (no activity; identical on every testbed, so not a leak).`
          : null;
      wrap.appendChild(
        h(
          "div",
          { class: "lab-check-line" },
          iconSpan("shield"),
          h(
            "div",
            {},
            h("div", { class: "lab-check-title" }, "Leakage certificate ", pill(ov.cls, ov.icon, ov.word)),
            h(
              "div",
              { class: "small muted" },
              flagged.length ? flagged.map((c) => `${SEVERITY[c.severity].word}: ${c.label}`).join(" · ") : `All ${leakChecks.length} leakage checks passed on this split.`,
            ),
            dupLine ? h("div", { class: "small muted" }, dupLine) : null,
          ),
        ),
      );
      if (supCheck) {
        wrap.appendChild(
          h(
            "div",
            { class: "lab-check-line" },
            iconSpan("gauge"),
            h(
              "div",
              {},
              h("div", { class: "lab-check-title" }, "Test support ", sup ? pill(sup.cls, sup.icon, sup.word) : pill("good", "check", "Enough attacks to score")),
              h("div", { class: "small muted" }, supCheck.detail),
            ),
          ),
        );
      }
    }
    const fid = result.fidelity;
    wrap.appendChild(
      h(
        "div",
        { class: "lab-check-line" },
        iconSpan("gauge"),
        h(
          "div",
          {},
          h("div", { class: "lab-check-title" }, "Generator fidelity", fid && isNum(fid.pretrainedAp) ? h("span", { class: "badge tabular" }, `AP ${f4(fid.pretrainedAp)}`) : null),
          h(
            "div",
            { class: "small muted" },
            fid
              ? fid.note
              : result.mode === "loro"
                ? "Not computed for leave-one-replica-out runs."
                : "Not computed for this run: the real pretrained model's weights (weights.json) could not be loaded.",
          ),
        ),
      ),
    );
    const lg = result.ledger;
    const status = result.ledgerStatus || (lg ? "done" : "off");
    wrap.appendChild(
      h(
        "div",
        { class: "lab-check-line" },
        iconSpan("lock"),
        h(
          "div",
          {},
          h(
            "div",
            { class: "lab-check-title" },
            "Tamper-evident ledger",
            status === "skipped-no-crypto" ? pill("warn", "alert", "Ledger skipped") : status === "off" ? pill("neutral", "info", "Off") : null,
          ),
          lg
            ? h(
                "div",
                { class: "small muted" },
                `${fmt.int(lg.nLeaves)} verdicts committed to a Merkle log${lg.capped ? ` (the first ${fmt.int(lg.nLeaves)} of ${fmt.int(lg.nTest)} test windows)` : ""}, root `,
                h("code", { class: "hash", title: lg.rootHex }, `${lg.rootHex.slice(0, 16)}…`),
                ". Changing any one verdict changes the root. ",
                h("a", { href: "#detector" }, "Try the tamper demo in the Live Detector"),
                ".",
              )
            : h(
                "div",
                { class: "small muted" },
                status === "skipped-no-crypto"
                  ? "Skipped: the ledger hashes with Web Crypto, which browsers only provide on https:// or localhost pages. No verdicts were committed; every result above is unaffected."
                  : "The ledger was turned off for this run.",
              ),
        ),
      ),
    );
    return wrap;
  }

  function resultsActions(result, entry) {
    const hasWeights = result.model.kind === "tessera" && !!result.model.weights;
    const m2Only = (result.features || result.config?.features) === "m2";
    const canUse = hasWeights && !m2Only;
    const nFolds = result.folds ? result.folds.length : 8;
    // a disabled button cannot be focused, so the reason is visible text linked by aria-describedby
    const reason = canUse
      ? null
      : result.mode === "loro"
        ? `Leave-one-replica-out trains ${nFolds} models, one per fold, so there is no single model to use. Run a single split to get one.`
        : hasWeights && m2Only
          ? "This model was trained on the network metrics only, but the Live Detector feeds it all four sources, so it would be scored on inputs it never saw."
          : result.model.kind === "pretrained"
            ? "This is the pretrained model, which the Live Detector already uses by default."
            : `Only a TESSERA-base network trained here can run in the Live Detector; this run used the ${MODEL[result.model.kind]?.short || result.model.kind}.`;
    const reasonId = `lab-use-reason-${entry.k}`;
    const useBtn = h(
      "button",
      {
        type: "button",
        class: "btn primary",
        disabled: !canUse,
        "aria-describedby": reason ? reasonId : null,
        html: `${icon("detector")}<span>Use this model in the Live Detector</span>`,
        onClick: () => {
          const label = `Lab model #${entry.k}`;
          store.setLabModel({
            weights: result.model.weights,
            label,
            createdAt: new Date().toISOString(),
            summary: {
              ap: result.metrics.ap,
              mcc: result.metrics.atThreshold?.mcc ?? null,
              split: STRATEGY[entry.strategy]?.short || result.mode,
              trainTest: entry.trainTest,
              synthetic: true,
              runId: result.runId,
            },
          });
          toast(h("span", {}, `${label} is now active in the Live Detector. `, h("a", { href: "#detector" }, "Open it")), { kind: "good", timeout: 5200 });
        },
      },
    );
    const base = `tessera-lab-run-${entry.k}`;
    const dl = (label, onClick, disabled = false, title = null) =>
      h("button", { type: "button", class: "btn", disabled, title, html: `${icon("download")}<span>${escapeHtml(label)}</span>`, onClick });
    return h(
      "div",
      { class: "lab-actions" },
      useBtn,
      reason ? h("p", { class: "hint lab-use-reason", id: reasonId }, iconSpan("info"), h("span", {}, reason)) : null,
      dl("Download results (JSON)", () => downloadFile(`${base}-results.json`, JSON.stringify(resultsJson(result, entry), null, 2), "application/json")),
      dl("Download scores (CSV)", () => downloadFile(`${base}-scores.csv`, scoresCsv(result), "text/csv")),
      dl(
        "Download weights (JSON)",
        () => downloadFile(`${base}-weights.json`, JSON.stringify(result.model.weights), "application/json"),
        !hasWeights,
        hasWeights ? null : "Weights are only available for a TESSERA-base model trained on a single split",
      ),
    );
  }

  function resultsJson(result, entry) {
    const { pr, roc, ...metrics } = result.metrics || {};
    const found = [];
    const P = (v, path) => toPlain(v, path, found);
    const features = result.features || result.config?.features || "all";
    const body = {
      schema: "tessera-lab-run/v1",
      note: "Training Lab run on SYNTHETIC windows (tessera-synth/v1) calibrated from summary statistics of the real AIT replicas. Not a result on real data.",
      notes: [],
      exportedUtc: new Date().toISOString(),
      run: entry.k,
      runId: result.runId,
      strategy: STRATEGY[entry.strategy]?.short || result.mode,
      experiment: entry.exp || null,
      config: P(result.config, "config"),
      mode: result.mode,
      features,
      model: {
        kind: result.model.kind,
        label: MODEL[result.model.kind]?.label || result.model.kind,
        nParams: result.model.nParams,
        ...(result.model.kind === "knn" ? { k: result.model.k, nStored: result.model.nStored } : {}),
        trainMs: result.model.trainMs,
        earlyStopMetric: result.model.history?.earlyStopMetric ?? null,
        history: P(result.model.history, "model.history"),
      },
      corpus: P(result.corpus, "corpus"),
      metrics: P(metrics, "metrics"),
      curves: P({ pr: pr || null, roc: roc || null }, "curves"),
      thresholdShown: S.threshold,
      atShownThreshold: (() => {
        const rows = scoredRows(result);
        return rows ? P(metricsAt(rows.y, rows.scores, S.threshold), "atShownThreshold") : null;
      })(),
      metricsScope: P(result.metricsScope ?? null, "metricsScope"),
      ledgerStatus: result.ledgerStatus ?? (result.ledger ? "done" : "off"),
      perReplica: P(result.perReplica, "perReplica"),
      perHost: P(result.perHost, "perHost"),
      attributionMean: P(result.attributionMean, "attributionMean"),
      fidelity: P(result.fidelity, "fidelity"),
      certificate: P(result.certificate, "certificate"),
      splitSummary: P(result.splitSummary, "splitSummary"),
      ledger: P(result.ledger, "ledger"),
      folds: P(result.folds ? result.folds.map(({ certificate, splitSummary, history, ...f }) => ({ ...f, certificateOverall: certificate?.overall })) : null, "folds"),
      loroSummary: P(result.loroSummary, "loroSummary"),
      timings: P(result.timings, "timings"),
      elapsedMs: result.elapsedMs,
    };
    body.notes.push(body.note);
    body.notes.push(
      features === "m2"
        ? "The model saw only the 24 network-metric (M2) features; the leakage certificate compared rows on those columns."
        : "The model saw all 42 features (log templates, network metrics, host identity, graph structure).",
    );
    if (result.mode === "loro") {
      body.notes.push(
        `metrics, curves and atShownThreshold are ${pooledLabel(result)}: the test windows of the folds with at least ${MIN_SUPPORT_FOR_RATES} attack windows, each scored by its own fold's model. loroSummary.ap.mean (the headline) is the mean of the per-fold AP over the same folds.`,
      );
    }
    for (const n of result.notes || []) body.notes.push(n);
    if (found.length) {
      const shown = found.slice(0, 8).join(", ");
      body.notes.push(
        `JSON has no Infinity or NaN, so ${found.length === 1 ? "1 non-finite number is" : `${found.length} non-finite numbers are`} written as the strings "Infinity", "-Infinity" or "NaN" (for example the ROC curve's first threshold, which is +Infinity by convention): ${shown}${found.length > 8 ? `, and ${found.length - 8} more` : ""}.`,
      );
    } else {
      body.notes.push("Every number in this file is finite.");
    }
    return body;
  }

  function scoresCsv(result) {
    const t = result.test;
    const ids = t.replicaIds || replicaIds;
    const hosts = t.hostIds || stats.hosts.map((x) => x.id);
    // leave-one-replica-out: which fold scored each row, and whether it counts in the pooled metrics
    const loro = result.mode === "loro" && t.fold && t.included;
    const foldIds = loro ? Object.fromEntries((result.folds || []).map((f, i) => [f.foldIndex ?? i, f.heldOut])) : {};
    const head = ["row", "replica", "host", "label", "score", ...(loro ? ["fold_held_out", "in_pooled_metrics"] : []), ...(t.attribution ? MODALITIES.map((m) => `attr_${m.id}`) : [])];
    const lines = [head.join(",")];
    for (let i = 0; i < t.n; i++) {
      const row = [i, ids[t.replica[i]], hosts[t.host[i]], t.y[i], t.scores[i].toFixed(6)];
      if (loro) row.push(foldIds[t.fold[i]] ?? t.fold[i], t.included[i]);
      if (t.attribution) for (let m = 0; m < 4; m++) row.push(t.attribution[i * 4 + m].toFixed(6));
      lines.push(row.join(","));
    }
    return `${lines.join("\n")}\n`;
  }

  /* ================================================================ history */

  function buildHistoryCard() {
    const card = stepCard(null, "Run history", "This session's runs, side by side. Nothing leaves your browser.", { id: "lab-history", tour: "lab-history" });
    refs.historyBody = h("div", { class: "lab-history-body" });
    refs.gapBox = h("div", { class: "lab-gap", hidden: true });
    card.append(refs.historyBody, refs.gapBox);
    renderHistory();
    return card;
  }

  function addHistory(run, result) {
    const isLoro = result.mode === "loro";
    const entry = {
      k: run.k,
      runId: run.id,
      strategy: run.strategy,
      mode: result.mode,
      trainTest: trainTestText(result.config.split),
      model: result.model.kind,
      features: result.features || result.config.features || "all",
      scale: result.config.dataset.scale,
      seed: result.config.dataset.seed,
      replicaKey: replicaSetOf(result.config.split).join(","),
      prevalence: result.metrics?.prevalence ?? null,
      nPositive: result.metrics?.nPositive ?? null,
      exp: run.exp ? run.exp.label : null,
      ap: isLoro ? result.loroSummary?.ap?.mean : result.metrics.ap,
      mcc: isLoro ? result.loroSummary?.mcc?.mean : result.metrics.atThreshold?.mcc,
      elapsedMs: result.elapsedMs,
      cert: result.certificate?.overall || null,
      replicaDisjoint: result.certificate?.checks?.find((c) => c.id === "replica-disjoint")?.severity === "ok",
      support: supportInfo(result.certificate), // null when the test set had enough attacks
    };
    S.history.push(entry);
    S.freshK = entry.k;
    S.results.set(entry.k, result);
    for (const k of [...S.results.keys()]) if (S.results.size > KEEP_RESULTS) S.results.delete(k);
    return entry;
  }

  /** The replicas a split draws on (training and test sides together), sorted. */
  function replicaSetOf(split) {
    if (!split) return [];
    const ids = split.mode === "replica" ? [...(split.train || []), ...(split.test || [])] : split.replicas || replicaIds;
    return [...new Set(ids)].sort();
  }

  function renderHistory() {
    if (!refs.historyBody) return;
    refs.historyBody.replaceChildren();
    if (!S.history.length) {
      refs.historyBody.appendChild(h("div", { class: "lab-empty lab-empty-sm" }, iconSpan("table", "lab-empty-ico"), h("p", {}, "No runs yet. Each run you make appears here so you can compare them.")));
      refs.gapBox.hidden = true;
      return;
    }
    const table = h("table", { class: "table" });
    table.appendChild(h("caption", { class: "sr-only" }, "Runs in this session"));
    table.appendChild(
      h(
        "thead",
        {},
        h(
          "tr",
          {},
          ...["#", "Strategy", "Train → test", "Model", "Features", "AP", "MCC", "Duration", "Leakage", ""].map((t, i) => h("th", { scope: "col", class: i >= 5 && i <= 7 ? "num" : "" }, t)),
        ),
      ),
    );
    const tbody = h("tbody");
    const shownK = S.lastShown?.entry?.k;
    // only the run that was just added fades in; re-renders leave the other rows still
    const freshK = S.freshK;
    S.freshK = null;
    for (const e of [...S.history].reverse()) {
      const ov = e.cert ? OVERALL[e.cert] : null;
      const viewable = S.results.has(e.k);
      const tr = h(
        "tr",
        { class: `lab-hist-row${e.k === freshK ? " is-new" : ""}${e.k === shownK ? " lab-row-current" : ""}` },
        h("td", { class: "tabular" }, String(e.k)),
        h("td", {}, STRATEGY[e.strategy]?.short || e.mode, " ", pill(STRATEGY[e.strategy]?.honesty.cls || "neutral", STRATEGY[e.strategy]?.honesty.icon || "info", STRATEGY[e.strategy]?.honesty.text || "")),
        h("td", { class: "lab-td-wrap" }, e.trainTest),
        h("td", {}, MODEL[e.model]?.short || e.model, e.exp ? h("span", { class: "tiny faint lab-quiet-note" }, "split experiment") : null),
        h("td", {}, FEATURE[e.features]?.table || e.features),
        h("td", { class: "num" }, f4(e.ap)),
        h("td", { class: "num" }, f3(e.mcc)),
        h("td", { class: "num" }, fmt.ms(e.elapsedMs)),
        h(
          "td",
          {},
          ov ? pill(ov.cls, ov.icon, ov.word) : "—",
          e.support ? h("span", { class: "tiny faint lab-quiet-note", title: `Test support: ${e.support.word}. This is not a leak.` }, `Test support: ${e.support.short}`) : null,
        ),
        h(
          "td",
          {},
          h(
            "button",
            {
              type: "button",
              class: "btn sm ghost",
              disabled: !viewable || e.k === shownK,
              title: viewable ? null : "Only the last 8 runs are kept in memory",
              onClick: () => {
                const r = S.results.get(e.k);
                if (!r) return;
                renderResults(r, e, { reveal: true });
                renderHistory();
                // this button was just replaced by renderHistory: move focus to what it opened
                scrollToCard("lab-results");
              },
            },
            e.k === shownK ? "Showing" : "View",
          ),
        ),
      );
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    refs.historyBody.appendChild(h("div", { class: "table-wrap" }, table));
    renderGap();
  }

  /** Which side of the leakage comparison a history entry is on (null: neither). */
  function gapKind(e) {
    if (e.mode === "random") return "random";
    if (e.mode === "chronological") return "chrono";
    if ((e.mode === "replica" || e.mode === "loro") && e.replicaDisjoint) return "holdout";
    return null;
  }

  function groupTitle(e) {
    const sc = SCALES.find((x) => x.scale === e.scale);
    const reps = e.replicaKey.split(",").length;
    return `${MODEL[e.model]?.short || e.model} · ${FEATURE[e.features]?.table || e.features} · ${sc ? `${sc.label} data` : `scale ${e.scale}`}, seed ${e.seed} · ${reps === replicaIds.length ? `all ${reps} replicas` : plural(reps, "replica")}`;
  }

  // Like with like: runs are compared only inside a group with the same model,
  // features, data size and seed, and replica set.
  function renderGap() {
    refs.gapBox.replaceChildren();
    const groups = new Map();
    for (const e of S.history) {
      const kind = gapKind(e);
      if (!kind || !isNum(e.ap)) continue;
      const key = [e.model, e.features, e.scale, e.seed, e.replicaKey].join("|");
      let g = groups.get(key);
      if (!g) groups.set(key, (g = { key, last: 0, latest: {}, sample: e }));
      g.latest[kind] = e; // history is in run order, so this keeps the latest
      g.last = Math.max(g.last, e.k);
    }
    const all = [...groups.values()];
    const ready = all.filter((g) => g.latest.random && (g.latest.chrono || g.latest.holdout)).sort((a, b) => b.last - a.last);
    const anyRandom = all.some((g) => g.latest.random);
    const anyHonest = all.some((g) => g.latest.chrono || g.latest.holdout);
    if (!ready.length) {
      refs.gapBox.hidden = !(anyRandom && anyHonest);
      if (!refs.gapBox.hidden) {
        refs.gapBox.append(
          h("div", { class: "lab-sub-title" }, "Leakage gap", h("span", { class: "faint tiny lab-sub-note" }, "like with like only")),
          h(
            "p",
            { class: "lab-takeaway" },
            iconSpan("info"),
            h("span", {}, "Your random and honest runs differ in model, features, data size, seed or replicas, so their scores are not compared here. Run both splits with the same settings, or use the split experiment at the top of the page, which does exactly that."),
          ),
        );
      }
      return;
    }
    refs.gapBox.hidden = false;
    refs.gapBox.append(h("div", { class: "lab-sub-title" }, "Leakage gap", h("span", { class: "faint tiny lab-sub-note" }, "random vs honest splits, same model, features, data and replicas")));
    for (const g of ready.slice(0, 3)) {
      const R = g.latest.random;
      const rows = [
        { label: `Random split (run #${R.k})`, ap: R.ap, base: R.prevalence, color: "--series-2", tag: pill("bad", "alert", "Leaky") },
      ];
      if (g.latest.chrono) rows.push({ label: `Chronological (run #${g.latest.chrono.k})`, ap: g.latest.chrono.ap, base: g.latest.chrono.prevalence, color: "--series-1" });
      if (g.latest.holdout) {
        const Hn = g.latest.holdout;
        rows.push({ label: `${Hn.mode === "loro" ? "LORO mean" : "Replica hold-out"} (run #${Hn.k})`, ap: Hn.ap, base: Hn.mode === "loro" ? null : Hn.prevalence, color: "--series-3", tag: pill("good", "check", "Honest") });
      }
      const parts = [];
      if (g.latest.chrono) parts.push(`${signedPts(R.ap - g.latest.chrono.ap)} against the chronological split`);
      if (g.latest.holdout) parts.push(`${signedPts(R.ap - g.latest.holdout.ap)} against the replica hold-out`);
      const worse = [g.latest.chrono, g.latest.holdout].filter(Boolean).every((x) => R.ap - x.ap < 0.02);
      const text = worse
        ? `The random split is ${parts.join(" and ")}: in this group it did not come out clearly ahead, so it shows no inflation here. The honest runs are marked on different test windows, and the certificate still flags the random split.`
        : `The random split reads ${parts.join(" and ")}. Same model, same features, same windows: what differs is only how they were split. Compare each bar with its own no-skill line, since the splits' test sets hold different shares of attacks.`;
      refs.gapBox.append(
        h(
          "div",
          { class: "lab-gap-group" },
          h("div", { class: "lab-gap-group-title small" }, groupTitle(g.sample)),
          compareBars(rows, { ariaLabel: `Average precision by split for ${groupTitle(g.sample)}` }),
          h("p", { class: "lab-takeaway" }, iconSpan("sparkles"), h("span", {}, text)),
        ),
      );
    }
    if (ready.length > 3) refs.gapBox.append(h("p", { class: "tiny faint" }, `Showing the 3 most recent of ${ready.length} comparable groups.`));
    const ld = realResults?.leakage_duplicates;
    if (ld && isNum(ld.r0_random_ap) && isNum(ld.r1_chronological_ap)) {
      refs.gapBox.appendChild(
        h(
          "p",
          { class: "small muted lab-gap-real" },
          h("span", { class: "tag-real" }, "Real"),
          ` The closest real counterpart (${ld.replica} only, ${/LightGBM/.test(ld.source || "") ? "LightGBM" : "the recorded model"}, ${ld.features}): the random split read AP ${f4(ld.r0_random_ap)} against ${f4(ld.r1_chronological_ap)} for a chronological split with a 600 s gap.`,
          recordedBy(ld.source) ? h("span", { class: "tiny faint lab-prov", title: ld.source }, ` Recorded by ${recordedBy(ld.source)}`) : null,
        ),
      );
    }
    refs.gapBox.classList.remove("reveal");
    void refs.gapBox.offsetWidth;
    refs.gapBox.classList.add("reveal");
  }
}
