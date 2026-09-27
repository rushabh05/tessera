// Overview tab: the landing page an examiner sees first.
//
// Where the numbers come from. Every figure on this page is either
//   - read from a file the Python exporter wrote from the real cache
//     (data/real_results.json, data/design_notes.json, data/replica_stats.json), or
//   - counted or measured live in this browser (parameters from
//     data/weights.json; the parity self-check against data/golden.json and
//     merkle_vectors.json, which the real PyTorch model and Python ledger wrote).
// Each one carries a tag and a source line saying which. None is typed in here.

import { h, s, countUp, stagger, revealOnScroll, reducedMotion, fmt, debounce, escapeHtml } from "../ui/dom.js";
import { icon } from "../ui/icons.js";
import { showTooltip, hideTooltip } from "../ui/charts.js";
import { forward } from "../forward.js";
import { MerkleLog, setSha256, verifyInclusion, bytesToHex, canonicalBytes } from "../merkle.js";

/* ------------------------------------------------------------------ constants */

// Fixed role colours for the four modalities (the same order the detector's
// attribution bars and the brand mark use).
const MODS = [
  { id: "m1_log", code: "M1", name: "Log templates", short: "Logs", color: "var(--series-1)", nFeat: 8 },
  { id: "m2_metrics", code: "M2", name: "Network metrics", short: "Network", color: "var(--series-2)", nFeat: 24 },
  { id: "m3_identity", code: "M3", name: "Host identity", short: "Identity", color: "var(--series-3)", nFeat: 2 },
  { id: "m4_graph", code: "M4", name: "Graph structure", short: "Graph", color: "var(--series-7)", nFeat: 8 },
];

const TAB_NAMES = {
  overview: "Overview",
  detector: "Live detector",
  lab: "Training Lab",
  results: "Real results",
  design: "Design decisions",
  about: "About",
};

let uid = 0;

/* ------------------------------------------------------------------ small helpers */

function setVars(el, vars) {
  for (const [k, v] of Object.entries(vars)) el.style.setProperty(k, String(v));
  return el;
}

const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const ap3 = (v) => (isNum(v) ? v.toFixed(3) : "—");
const ap4 = (v) => (isNum(v) ? v.toFixed(4) : "—");
const pct0 = (v) => (isNum(v) ? `${Math.round(Math.min(1, Math.max(0, v)) * 100)}%` : "—");

/** "200-tree " from loro.model ("LightGBM, 200 trees, seed 0, ..."), or "". */
function treeWord(model) {
  const m = /(\d[\d,]*)\s*trees/i.exec(String(model || ""));
  return m ? `${m[1]}-tree ` : "";
}

function tag(kind, title) {
  if (kind === "real") return h("span", { class: "tag-real", title: title || "Measured on real AIT data" }, "Real");
  if (kind === "measured") return h("span", { class: "tag-real", title: title || "Measured in a recorded run" }, "Measured");
  if (kind === "live") return h("span", { class: "tag-real", title: title || "Measured just now, in this browser" }, "Live check");
  if (kind === "synthetic") return h("span", { class: "tag-synthetic", title: title || "Synthetic data" }, "Synthetic");
  return null;
}

function tabLink(tab, text, cls = "ov-link") {
  return h("a", { class: cls, href: `#${tab}`, html: `<span>${escapeHtml(text)}</span>${icon("arrowRight")}` });
}

function inView(el, fn, { once = true, margin = "0px 0px -10% 0px" } = {}) {
  if (typeof IntersectionObserver !== "function") {
    fn();
    return () => {};
  }
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          fn();
          if (once) io.disconnect();
        }
      }
    },
    { rootMargin: margin }
  );
  io.observe(el);
  return () => io.disconnect();
}

/* ------------------------------------------------------------------ data */

async function loadData(store) {
  const settle = (p) => p.then((v) => v, () => null);
  const [real, design, stats, weights] = await Promise.all([
    settle(store.loadJSON("data/real_results.json")),
    settle(store.loadJSON("data/design_notes.json")),
    settle(store.loadJSON("data/replica_stats.json")),
    settle(store.loadPretrainedWeights()),
  ]);
  return { real, design, stats, weights };
}

function countParams(w) {
  try {
    const lin = (l) => l.weight.flat().length + l.bias.length;
    let n = 0;
    for (const e of w.encoders) n += lin(e.linear0) + lin(e.linear1) + e.groupnorm.weight.length + e.groupnorm.bias.length;
    n += lin(w.fusion.gate_linear0) + lin(w.fusion.gate_linear1) + lin(w.head.linear0) + lin(w.head.linear1);
    return n;
  } catch {
    return null;
  }
}

/** Derive every displayed figure from the loaded files. Missing pieces stay undefined. */
function derive({ real, design, stats, weights }) {
  const D = { ok: { real: !!real, design: !!design, stats: !!stats, weights: !!weights } };

  const loro = real?.loro;
  if (loro?.summary?.average_precision && isNum(loro.summary.average_precision.mean)) {
    D.ap = loro.summary.average_precision;
    D.mcc = loro.summary.mcc;
    D.naive = loro.summary.naive_all_folds_ap;
    D.excluded = loro.summary.excluded_low_support || [];
    D.minSupport = loro.summary.min_support ?? 20;
    D.loroModel = loro.model;
    D.lowFolds = (loro.folds || []).filter((f) => f.low_support);
  }

  const ds = real?.dataset;
  if (ds?.replicas?.length) {
    D.nReplicas = ds.replicas.length;
    D.nWindows = ds.totals?.n_windows;
    D.nPositive = ds.totals?.n_positive;
    const spans = ds.replicas.map((r) => r.span_hours).filter(isNum);
    if (spans.length) D.spanRange = [Math.min(...spans), Math.max(...spans)];
    D.licence = ds.licence;
    D.zenodo = ds.zenodo_record;
  }

  D.leak = real?.leakage_duplicates || null;
  D.calendar = real?.calendar_leak?.rows?.length >= 2 ? real.calendar_leak : null;
  D.allMod = real?.all_modalities_r0_r1 || null;
  D.tb = real?.tessera_base || null;
  // The shortcut floor: the same LORO folds, a model that sees only which sources are present.
  const fl = real?.mask_only_floor;
  D.floor = fl?.summary?.average_precision && isNum(fl.summary.average_precision.mean) ? fl : null;

  const abl = real?.attribution_vs_ablation;
  if (abl?.ablation) {
    const find = (re) => abl.ablation.find((r) => re.test(r.check));
    D.abl = {
      m1Alone: find(/M1 alone/i),
      m3Alone: find(/M3 alone/i),
      hostAlone: find(/host_bucket alone/i),
      withM1: find(/with M1/i),
      withoutM1: find(/without M1/i),
      attr: abl.mean_attribution || null,
    };
  }

  D.params = countParams(weights) ?? real?.tessera_base?.n_parameters ?? null;
  D.paramsCounted = countParams(weights) != null;

  // Availability (a modality counts as present if any feature in its slice is
  // non-zero), pooled over every replica: aggregates of tens of thousands of
  // real windows, far above min_support.
  const gp = stats?.global_pooled;
  const minSup = stats?.min_support ?? 20;
  const availOf = (cls) => {
    const c = gp?.[cls];
    if (!c || !(c.n >= minSup) || !Array.isArray(c.patterns)) return null;
    const a = [0, 0, 0];
    for (const p of c.patterns) for (let k = 0; k < 3; k++) a[k] += (p.a?.[k] ? 1 : 0) * p.p;
    return { m1: Math.min(1, a[0]), m2: Math.min(1, a[1]), m4: Math.min(1, a[2]), n: c.n };
  };
  D.avail = gp ? { benign: availOf("benign"), attack: availOf("attack"), minSupport: minSup } : null;

  // Per-host attack share, pooled over the replicas. A host with fewer than
  // min_support attack windows is reported as a count, never as a rate.
  if (stats?.replicas && stats?.hosts) {
    D.hosts = stats.hosts.map((host) => {
      let n = 0;
      let p = 0;
      for (const r of stats.replicas) {
        const hh = r.hosts?.[host.id];
        if (hh) {
          n += hh.n_windows || 0;
          p += hh.n_positive || 0;
        }
      }
      return { id: host.id, label: host.label, n, p, rate: n ? p / n : null, asCount: p < minSup };
    });
  }

  const opt = design?.optimiser_comparison?.results;
  if (opt) {
    const find = (re) => opt.find((r) => re.test(r.optimiser));
    D.opt = { eho: find(/EHO/), random: find(/random/i), tpe: find(/TPE|optuna/i), all: opt };
  }
  D.fh = design?.fh_invariance || null;
  return D;
}

function hostPhrase(hh) {
  if (!hh) return "";
  return hh.asCount ? `${fmt.int(hh.p)} of ${fmt.int(hh.n)}` : fmt.pct(hh.rate, hh.rate < 0.01 ? 2 : 1);
}

/** "0.08% of Intranet Server windows" or, below min_support, "9 of 66,519 VPN Gateway windows". */
function hostWindows(hh) {
  if (!hh) return "";
  return hh.asCount ? `${hostPhrase(hh)} ${hh.label} windows` : `${hostPhrase(hh)} of ${hh.label} windows`;
}

/** Parse one SVG/HTML markup string into an element. */
function fromMarkup(markup) {
  const t = document.createElement("template");
  t.innerHTML = markup.trim();
  return t.content.firstElementChild;
}

/* ------------------------------------------------------------------ live self-check */

/**
 * Re-run the two parity checks the repo's Node tests run, here in the browser:
 * the JS forward pass against PyTorch's outputs (data/golden.json), and the JS
 * ledger against the Python ledger's root and proofs (merkle_vectors.json),
 * then a tamper check. Returns a result object; never throws.
 */
async function runSelfCheck(store, onUpdate) {
  const R = { model: { state: "running" }, ledger: { state: "pending" }, tamper: { state: "pending" } };
  const push = () => onUpdate({ model: { ...R.model }, ledger: { ...R.ledger }, tamper: { ...R.tamper } });
  const pace = () => new Promise((r) => setTimeout(r, reducedMotion() ? 0 : 420));
  push();
  await pace();

  try {
    const t0 = performance.now();
    const [weights, golden] = await Promise.all([store.loadPretrainedWeights(), store.loadJSON("data/golden.json")]);
    let pass = 0;
    let maxDelta = 0;
    let maxAttr = 0;
    let masked = 0;
    for (const v of golden.vectors) {
      const { score, attribution } = forward(v.input, v.availability, weights);
      const d = Math.abs(score - v.expected_score);
      let ad = 0;
      for (let m = 0; m < attribution.length; m++) ad = Math.max(ad, Math.abs(attribution[m] - v.expected_attribution[m]));
      maxDelta = Math.max(maxDelta, d);
      maxAttr = Math.max(maxAttr, ad);
      if (d < 1e-4 && ad < 1e-4) pass++;
      if (v.availability.some((a) => a < 0.5)) masked++;
    }
    const n = golden.vectors.length;
    R.model = { state: pass === n && n > 0 ? "done" : "error", pass, n, maxDelta, maxAttr, masked, ms: performance.now() - t0 };
  } catch (err) {
    R.model = { state: "error", message: err.message };
  }
  R.ledger = { state: "running" };
  push();
  await pace();

  let log = null;
  let vec = null;
  try {
    if (!globalThis.crypto?.subtle) throw new Error("Web Crypto is unavailable here (it needs https or localhost)");
    setSha256(async (bytes) => new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)));
    const t0 = performance.now();
    vec = await store.loadJSON("merkle_vectors.json");
    log = new MerkleLog();
    for (const v of vec.verdicts) await log.appendJson(v);
    const root = await log.root();
    const rootHex = bytesToHex(root);
    const rootOk = rootHex === vec.root_hex;
    let ok = 0;
    for (const pv of vec.proofs) {
      const p = await log.inclusionProof(pv.leaf_index, pv.tree_size);
      const leafMatch = bytesToHex(p.leaf) === pv.leaf_hex;
      const pathMatch = JSON.stringify(p.path.map(bytesToHex)) === JSON.stringify(pv.path_hex);
      const verifies = await verifyInclusion(p.leaf, pv.leaf_index, pv.tree_size, p.path, root);
      if (leafMatch && pathMatch && verifies) ok++;
    }
    R.ledger = {
      state: rootOk && ok === vec.proofs.length ? "done" : "error",
      rootOk,
      ok,
      n: vec.proofs.length,
      nVerdicts: vec.verdicts.length,
      rootHex,
      root,
      ms: performance.now() - t0,
    };
  } catch (err) {
    R.ledger = { state: "error", message: err.message };
    log = null;
  }
  R.tamper = { state: log ? "running" : "error", message: log ? undefined : "needs the ledger check to pass first" };
  push();
  if (!log) return R;
  await pace();

  try {
    const i = Math.min(3, vec.verdicts.length - 1);
    const original = vec.verdicts[i];
    const before = await log.inclusionProof(i);
    const rootBefore = R.ledger.root;
    const forged = { ...original, verdict: !original.verdict, score: original.verdict ? "0.000000" : "0.999999" };
    await log.tamper(i, canonicalBytes(forged));
    const rootAfter = await log.root();
    const changed = bytesToHex(rootBefore) !== bytesToHex(rootAfter);
    const oldStill = await verifyInclusion(before.leaf, i, before.treeSize, before.path, rootBefore);
    const staleOnNew = await verifyInclusion(before.leaf, i, before.treeSize, before.path, rootAfter);
    R.tamper = {
      state: changed && oldStill && !staleOnNew ? "done" : "error",
      index: i,
      changed,
      oldStill,
      staleOnNew,
      before: bytesToHex(rootBefore),
      after: bytesToHex(rootAfter),
    };
  } catch (err) {
    R.tamper = { state: "error", message: err.message };
  }
  push();
  return R;
}

/* ------------------------------------------------------------------ hero */

function buildHero(ctx) {
  const startTour = async () => {
    try {
      const mod = await import("./tour.js");
      mod.startTour(ctx);
    } catch (err) {
      console.error(err);
      ctx.toast(`The guided tour failed to load: ${err.message}`, { kind: "bad" });
    }
  };

  const title = h("h1", { class: "ov-title", id: "ov-title" }, h("span", { class: "sr-only" }, "TESSERA"));
  const letters = h("span", { class: "ov-title-letters", "aria-hidden": "true" });
  [..."TESSERA"].forEach((ch, i) => letters.appendChild(setVars(h("span", { class: "ov-letter" }, ch), { "--i": i })));
  title.appendChild(letters);

  const mark = h("div", { class: "ov-mark", "aria-hidden": "true" });
  const dirs = [
    [-1, -1],
    [1, -1],
    [-1, 1],
    [1, 1],
  ];
  dirs.forEach(([dx, dy], i) => mark.appendChild(setVars(h("span", { class: `ov-tile ov-tile-${i + 1}` }), { "--i": i, "--dx": dx, "--dy": dy })));

  const legend = h("ul", { class: "ov-mods", "aria-label": "The four feature groups (modalities)" });
  MODS.forEach((m, i) => {
    const li = h(
      "li",
      { class: "ov-mod" },
      setVars(h("span", { class: "swatch" }), { background: m.color }),
      h("span", { class: "ov-mod-code" }, m.code),
      h("span", {}, m.name)
    );
    li.addEventListener("pointerenter", () => (mark.dataset.hi = String(i + 1)));
    li.addEventListener("pointerleave", () => delete mark.dataset.hi);
    legend.appendChild(li);
  });

  const art = h(
    "div",
    { class: "ov-hero-art" },
    h("div", { class: "ov-mark-wrap" }, mark),
    h("div", { class: "ov-mods-cap tiny faint" }, "Four tiles, four feature groups from two telemetry streams"),
    legend
  );
  art.addEventListener("pointerenter", () => mark.classList.add("is-spread"));
  art.addEventListener("pointerleave", () => mark.classList.remove("is-spread"));

  const cta = h(
    "div",
    { class: "ov-cta" },
    h("a", { class: "btn lg primary", href: "#detector", html: `${icon("detector")}<span>Try the live detector</span>` }),
    h("a", { class: "btn lg", href: "#lab", html: `${icon("lab")}<span>Open the Training Lab</span>` }),
    h("a", { class: "btn lg", href: "#results", html: `${icon("results")}<span>See real results</span>` })
  );

  const tourRow = h(
    "div",
    { class: "ov-tour-row" },
    h("button", { class: "btn ghost ov-tour-btn", type: "button", html: `${icon("compass")}<span>Take the guided tour</span>`, onClick: startTour }),
    h("span", { class: "small faint" }, "A short walk through every tab.", h("span", { class: "ov-keys-hint" }, " Use the arrow keys to move."))
  );

  return h(
    "section",
    { class: "ov-hero", "data-tour": "overview-hero", "aria-labelledby": "ov-title" },
    h(
      "div",
      { class: "ov-hero-text" },
      h("div", { class: "eyebrow" }, "Final-year engineering project"),
      title,
      h("p", { class: "ov-tagline" }, "Honest, multimodal cloud anomaly detection with a tamper-evident verdict ledger"),
      h(
        "p",
        { class: "ov-intro" },
        "TESSERA is a multimodal cloud anomaly detector built around leakage-checked evaluation and a tamper-evident verdict ledger. It fuses four feature groups (modalities) from two telemetry streams, log files and Suricata network events, seen on the same host in the same minute. It measures itself on splits designed to limit leakage, each with a leakage certificate, and commits every verdict to a Merkle log that holds hashes, never an IP address, username, URL or request body."
      ),
      cta,
      tourRow
    ),
    art
  );
}

/* ------------------------------------------------------------------ headline numbers */

function buildStats() {
  const grid = h("div", { class: "grid-4 ov-stat-grid" });
  const defs = [
    { key: "ap", label: "Mean cross-replica AP (LightGBM baseline)", primary: true },
    { key: "replicas", label: "Replicas of the AIT testbed" },
    { key: "params", label: "TESSERA-base parameters" },
    { key: "servers", label: "Servers" },
  ];
  const cells = {};
  for (const d of defs) {
    const value = h("div", { class: "stat-value ov-stat-value tabular" }, h("span", { class: "skeleton ov-skel" }));
    const sub = h("div", { class: "stat-sub ov-stat-sub" }, h("span", { class: "skeleton ov-skel-line" }));
    const foot = h("div", { class: "ov-stat-foot" });
    const card = h("div", { class: `stat ov-stat${d.primary ? " primary" : ""}` }, h("div", { class: "stat-label" }, d.label), value, sub, foot);
    cells[d.key] = { card, value, sub, foot };
    grid.appendChild(card);
  }

  const notice = h("div", { class: "notice ov-honest" });
  const legend = h(
    "div",
    { class: "ov-legend small muted" },
    h("span", { class: "tag-real" }, "Real"),
    h("span", {}, "measured on real, held-out AIT data or counted from the shipped files"),
    h("span", { class: "tag-synthetic" }, "Synthetic"),
    h("span", {}, "generated in your browser, never evidence of accuracy")
  );

  const section = h(
    "section",
    { class: "ov-section ov-stats", "aria-labelledby": "ov-stats-h" },
    h(
      "div",
      { class: "ov-section-head" },
      h("div", {}, h("div", { class: "eyebrow" }, "At a glance"), h("h2", { id: "ov-stats-h" }, "The headline numbers")),
      legend
    ),
    grid,
    notice
  );

  // fill() lays out the final text straight away (so nothing below shifts when
  // the numbers animate); play() runs the count-up once the row is on screen.
  const counters = [];
  let played = false;
  function fill(D) {
    counters.length = 0;
    const put = (key, { to, format, sub, foot, from = 0, pop = false }) => {
      const c = cells[key];
      c.value.replaceChildren();
      const num = h("span", { class: "ov-num" });
      c.value.appendChild(num);
      if (isNum(to)) {
        num.textContent = played || reducedMotion() ? format(to) : format(from);
        counters.push({ num, to, from, format, pop });
      } else num.textContent = "—";
      c.sub.replaceChildren(sub);
      c.foot.replaceChildren(...foot);
    };

    put("ap", {
      to: D.ap?.mean,
      format: (v) => v.toFixed(3),
      sub: D.ap
        ? h(
            "span",
            {},
            `Train on 7 replicas, test on the 8th · ${D.ap.n} folds, ± ${ap3(D.ap.std)}`,
            D.tb ? h("span", { class: "ov-stat-pair" }, `TESSERA-base: ${ap4(D.tb.tessera_ap)} on one held-out fold (${D.tb.held_out || "santos"})`) : null
          )
        : "Real results file not available",
      foot: D.ap
        ? [
            tag("real", "Recomputed by the exporter from the real AIT cache; the run tests/test_loro_real.py asserts on"),
            h("span", { class: "tiny faint" }, `A default-configured ${treeWord(D.loroModel)}LightGBM; ${D.excluded.join(", ") || "no"} fold excluded (${D.lowFolds.map((f) => `${fmt.int(f.n_pos_test)} attack windows`).join(", ") || "low support"})`),
          ]
        : [],
    });
    put("replicas", {
      to: D.nReplicas,
      format: (v) => String(Math.round(v)),
      sub: isNum(D.nWindows) ? `${fmt.int(D.nWindows)} real 60-second windows, ${fmt.int(D.nPositive)} of them attacks` : "AIT Log Data Set V2.1",
      foot: D.nReplicas ? [tag("real", "Counted from the real cache by the exporter"), h("span", { class: "tiny faint" }, "AIT Log Data Set V2.1")] : [],
    });
    put("params", {
      to: D.params,
      format: (v) => fmt.int(v),
      sub: "Four small encoders, a gate and a head. Small enough to train in your browser.",
      foot: D.params
        ? [tag("real", D.paramsCounted ? "Counted just now from data/weights.json" : "From real_results.json"), h("span", { class: "tiny faint" }, D.paramsCounted ? "Counted from the shipped weights" : "TESSERA-base")]
        : [],
    });
    put("servers", {
      to: 0,
      format: (v) => String(Math.round(v)),
      pop: true,
      sub: "Static files only. The model, the ledger and training all run on your device.",
      foot: [h("span", { class: "pill neutral", html: `${icon("lock")}By design` })],
    });

    notice.replaceChildren(
      fromMarkup(icon("info", { cls: "notice-icon" })),
      h(
        "div",
        {},
        h("strong", {}, "Why average precision, not accuracy. "),
        D.leak
          ? `Accuracy on a random split looks excellent when a large benign majority dominates the data, even over pooled corpora that contain duplicate rows. We report average precision (AP), which that majority cannot inflate the same way, on splits designed to limit leakage, each with a leakage certificate. The effect is real: on the same data our own random split scores AP ${ap3(D.leak.r0_random_ap)}, a chronological split ${ap3(D.leak.r1_chronological_ap)}. `
          : "Accuracy on a random split looks excellent when a large benign majority dominates the data. We report average precision (AP) on splits designed to limit leakage, each with a leakage certificate. ",
        h("a", { href: "#results" }, "Real results"),
        " has every number and its source; ",
        h("a", { href: "#design" }, "See the design decisions behind this"),
        "."
      )
    );
  }

  function play() {
    if (played) return;
    played = true;
    for (const c of counters) {
      if (c.pop) {
        c.num.textContent = c.format(c.to);
        c.num.classList.add("pop");
      } else countUp(c.num, c.to, { from: c.from, duration: 1100, format: c.format });
    }
  }

  return { el: section, fill, play, grid };
}

/* ------------------------------------------------------------------ pipeline */

const PIPE = [
  { id: "raw", stage: 1, kind: "single", icon: "database", title: "Raw logs", sub: "AIT V2.1 · 8 replicas", cTitle: "Raw logs", cSub: "AIT V2.1, 8 replicas" },
  { id: "win", stage: 2, kind: "single", icon: "clock", title: "60-second windows", sub: "one sample per host per minute", cTitle: "60-second windows", cSub: "per host" },
  ...MODS.map((m, k) => ({ id: `m${k + 1}`, stage: 3, kind: "mod", m: k, title: m.name, sub: `${m.nFeat} features`, cTitle: m.code, cSub: m.short })),
  ...MODS.map((m, k) => ({ id: `e${k + 1}`, stage: 4, kind: "enc", m: k, title: "Encoder", sub: `${m.nFeat} → 32 → 16`, cTitle: "Enc", cSub: "→ 16" })),
  { id: "fuse", stage: 5, kind: "fuse", title: "Gated fusion", sub: "masks missing sources", cTitle: "Gated fusion", cSub: "masks missing sources" },
  { id: "score", stage: 6, kind: "single", icon: "gauge", title: "Attack score", sub: "a probability, 0 to 1", cTitle: "Attack score", cSub: "0 to 1" },
  { id: "ledger", stage: 7, kind: "single", icon: "lock", title: "Merkle ledger", sub: "stores hashes only", cTitle: "Merkle ledger", cSub: "hashes only" },
];
const PIPE_BY_ID = Object.fromEntries(PIPE.map((n) => [n.id, n]));
const N_STAGES = 7;
const SEQ = ["raw", "win", "m1", "m2", "m3", "m4", "e1", "fuse", "score", "ledger"];
const FOCUS_ORDER = PIPE.map((n) => n.id);
const EDGES = [
  ["raw", "win", null],
  ...[0, 1, 2, 3].map((k) => ["win", `m${k + 1}`, k]),
  ...[0, 1, 2, 3].map((k) => [`m${k + 1}`, `e${k + 1}`, k]),
  ...[0, 1, 2, 3].map((k) => [`e${k + 1}`, "fuse", k]),
  ["fuse", "score", null],
  ["score", "ledger", null],
];
const ARROW_GAP = 3;

function computeLayout(W) {
  // Horizontal needs ~120 px per column; below that the vertical flow reads better.
  const L = { W, horiz: W >= 960, nodes: {}, H: 0, compact: false };
  if (L.horiz) {
    const cols = 7;
    const gap = Math.round(Math.min(44, Math.max(26, W * 0.034)));
    const nw = (W - gap * (cols - 1)) / cols;
    const qh = 63;
    const qg = 8;
    const pad = 6;
    const H = 4 * qh + 3 * qg + pad * 2;
    const cy = H / 2;
    const cx = (c) => c * (nw + gap);
    const single = (id, c, hh) => (L.nodes[id] = { x: cx(c), y: cy - hh / 2, w: nw, h: hh, col: c });
    single("raw", 0, 132);
    single("win", 1, 132);
    for (let k = 0; k < 4; k++) {
      L.nodes[`m${k + 1}`] = { x: cx(2), y: pad + k * (qh + qg), w: nw, h: qh, col: 2 };
      L.nodes[`e${k + 1}`] = { x: cx(3), y: pad + k * (qh + qg), w: nw, h: qh, col: 3 };
    }
    single("fuse", 4, 176);
    single("score", 5, 112);
    single("ledger", 6, 132);
    L.H = H;
    L.compact = nw < 104;
  } else {
    const gapY = 34;
    const tg = 8;
    const sw = Math.min(W, 440);
    const sx = (W - sw) / 2;
    const tw = (W - tg * 3) / 4;
    let y = 0;
    let row = 0;
    const single = (id, hh) => {
      L.nodes[id] = { x: sx, y, w: sw, h: hh, col: row++ };
      y += hh + gapY;
    };
    single("raw", 58);
    single("win", 58);
    for (let k = 0; k < 4; k++) L.nodes[`m${k + 1}`] = { x: k * (tw + tg), y, w: tw, h: 64, col: row };
    row++;
    y += 64 + gapY;
    for (let k = 0; k < 4; k++) L.nodes[`e${k + 1}`] = { x: k * (tw + tg), y, w: tw, h: 54, col: row };
    row++;
    y += 54 + gapY;
    single("fuse", 92);
    single("score", 58);
    single("ledger", 58);
    L.H = y - gapY;
    L.compact = tw < 132;
  }
  return L;
}

function edgePath(L, from, to, k) {
  const a = L.nodes[from];
  const b = L.nodes[to];
  const r = (v) => Math.round(v * 10) / 10;
  if (L.horiz) {
    let y1 = a.y + a.h / 2;
    let y2 = b.y + b.h / 2;
    if (from === "win") y1 += (k - 1.5) * 14;
    if (to === "fuse") y2 += (k - 1.5) * 24;
    const x1 = a.x + a.w;
    const x2 = b.x - ARROW_GAP;
    const mx = (x1 + x2) / 2;
    return `M${r(x1)} ${r(y1)} C${r(mx)} ${r(y1)} ${r(mx)} ${r(y2)} ${r(x2)} ${r(y2)}`;
  }
  let x1 = a.x + a.w / 2;
  let x2 = b.x + b.w / 2;
  if (from === "win") x1 += (k - 1.5) * Math.min(26, a.w / 8);
  if (to === "fuse") x2 += (k - 1.5) * Math.min(40, b.w / 6);
  const y1 = a.y + a.h;
  const y2 = b.y - ARROW_GAP;
  const my = (y1 + y2) / 2;
  return `M${r(x1)} ${r(y1)} C${r(x1)} ${r(my)} ${r(x2)} ${r(my)} ${r(x2)} ${r(y2)}`;
}

/** The explanation shown for a pipeline node, built from the loaded data. */
function explain(id, D, C) {
  const n = PIPE_BY_ID[id];
  const facts = [];
  const fact = (value, text, kind, source) => facts.push({ value, text, kind, source });
  const note = (text) => facts.push({ text });
  let what = "";
  let tab = "results";
  let tabText = "See the real results";
  let title = n.kind === "mod" ? `${MODS[n.m].code} · ${MODS[n.m].name}` : n.title;
  const benign = D.avail?.benign;
  const attack = D.avail?.attack;
  const availSrc = "data/replica_stats.json: aggregate availability pooled over all 8 real replicas";

  switch (id) {
    case "raw":
      what =
        "The AIT Log Data Set V2.1 (Landauer et al.): eight replicas of one simulated enterprise network, each with labelled multi-step attacks. TESSERA reads three hosts (a VPN gateway, an intranet server and an internet firewall): four log files (auth.log, audit.log, openvpn.log, dnsmasq.log) plus the Suricata network sensor's events.";
      if (isNum(D.nWindows)) fact(fmt.int(D.nWindows), `real windows across ${D.nReplicas} replicas, ${fmt.int(D.nPositive)} of them attacks`, "real", "Counted from the real cache by the exporter (real_results.json, dataset)");
      if (D.spanRange) fact(`${Math.round(D.spanRange[0])}–${Math.round(D.spanRange[1])} h`, `of capture per replica (${(D.spanRange[0] / 24).toFixed(1)} to ${(D.spanRange[1] / 24).toFixed(1)} days)`, "real", "real_results.json, dataset.replicas[].span_hours");
      note(`Only aggregate statistics reach this site. That is the project's data policy (LICENSE-DATA), not a licence ban: the dataset is ${D.licence || "CC BY-NC-SA 4.0"}, and shipping derived rows would bring ShareAlike obligations with them.`);
      tab = "results";
      tabText = "See the dataset in Real results";
      break;
    case "win": {
      what =
        "Each host's activity is cut into 60-second windows, and one window is one sample. A window is labelled an attack if any labelled log line from any source on that host falls inside it.";
      const fw = D.hosts?.find((x) => x.id === "inet-firewall");
      const is = D.hosts?.find((x) => x.id === "intranet_server");
      const vpn = D.hosts?.find((x) => x.id === "vpn");
      if (fw && is && vpn)
        fact(
          hostPhrase(fw),
          `${fw.asCount ? `${fw.label} windows are` : `of ${fw.label} windows are`} attacks, against ${hostWindows(is)} and ${hostWindows(vpn)}`,
          "real",
          "Aggregate counts in data/replica_stats.json, all 8 replicas (a host with fewer than 20 attack windows is shown as a count, not a rate)"
        );
      if (D.leak)
        fact(
          fmt.pct(D.leak.exact_duplicate_rate, 0),
          `of windows exactly duplicate another (${D.leak.replica}, network metrics only), so a random split puts copies of test rows into training`,
          "real",
          "Recomputed by the exporter with the protocol of tests/test_pipeline_real_data.py"
        );
      tab = "lab";
      tabText = "Split windows yourself in the Training Lab";
      break;
    }
    case "m1":
      what =
        "Drain3 mines one shared vocabulary of log templates from the message bodies (timestamps stripped first). Each window becomes 8 statistics: event count, unique templates, template entropy, the dominant template and its share, new templates, and line lengths.";
      if (D.abl?.m1Alone) fact(ap4(D.abl.m1Alone.ap), "AP for a gradient-boosted model using log templates alone (held-out santos)", "real", "RESULTS.md finding 4 (one-off ablation)");
      if (benign && attack) fact(pct0(benign.m1), `of benign windows have any log-template activity, against ${pct0(attack.m1)} of attack windows`, "real", availSrc);
      note("That gap is partly by construction: the AIT labels exist only for these log files, so every attack window has log activity. The About FAQ ('Is log-template presence just the label?') gives the real floor this sets.");
      tab = "detector";
      tabText = "Switch it off in the Live detector";
      break;
    case "m2":
      what =
        "24 aggregates of the Suricata network sensor's events per window: flow, alert, DNS, HTTP and TLS counts, packet and byte totals and 90th percentiles, flow durations, destination-port spread and protocol mix.";
      if (D.leak) fact(`${ap3(D.leak.r0_random_ap)} → ${ap3(D.leak.r1_chronological_ap)}`, "AP on network metrics alone: a random split, then a chronological split of the same data", "real", "Recomputed by the exporter (RESULTS.md finding 1)");
      if (benign) fact(pct0(benign.m2), "of benign windows have any network-sensor activity", "real", availSrc);
      tab = "results";
      tabText = "See the leakage finding";
      break;
    case "m3":
      what =
        "Two derived features, not a telemetry stream of their own: a hashed host bucket and the number of feature groups (M1, M2, M4) present in the window. Calendar features (hour of day, day of week) were removed after they turned out to leak.";
      if (D.calendar) {
        const [withCal, without] = D.calendar.rows;
        fact(`${withCal.gap.toFixed(3)} → ${without.gap.toFixed(3)}`, "gap between random-split and chronological AP, with calendar features and then without them", "real", "Transcribed from RESULTS.md finding 2 (same-seed ablation)");
      }
      if (D.abl?.m3Alone) fact(ap4(D.abl.m3Alone.ap), "AP for a gradient-boosted model on host identity alone", "real", "RESULTS.md finding 4 (one-off ablation)");
      tab = "results";
      tabText = "See the calendar leak";
      break;
    case "m4":
      what =
        "Read from the same Suricata flow records as M2, in time order: each window builds a host-to-destination-subnet graph with memory across windows (how many peers are new, how often old peers come back, how concentrated the traffic is). 8 features. Destination IPs are cut to /24 subnets before anything else sees them.";
      if (D.allMod) fact(ap3(D.allMod.r1_ap), `chronological-split AP with all four feature groups together (MCC ${ap3(D.allMod.r1_mcc)}, ${D.allMod.replica})`, "real", "Recomputed by the exporter (RESULTS.md finding 2)");
      if (benign) fact(pct0(benign.m4), "of benign windows have any graph activity", "real", availSrc);
      tab = "results";
      tabText = "See the real results";
      break;
    case "e1":
    case "e2":
    case "e3":
    case "e4": {
      const m = MODS[n.m];
      title = `Encoder for ${m.code} · ${m.name}`;
      what = `One tiny network per source: Linear → GroupNorm → GELU → Linear, turning ${m.name.toLowerCase()}'s ${m.nFeat} features into 16 numbers. Features go in raw, and GroupNorm copes with their very different scales. All four encoders feed the gate.`;
      if (D.params) fact(fmt.int(D.params), "parameters in the whole model", "real", D.paramsCounted ? "Counted just now from data/weights.json" : "real_results.json, tessera_base");
      if (C?.model?.state === "done")
        fact(`${C.model.pass}/${C.model.n}`, `PyTorch reference outputs reproduced in your browser, just now (largest score difference ${C.model.maxDelta.toExponential(1)})`, "live", "Checked live against data/golden.json, written by the real trained PyTorch model");
      tab = "lab";
      tabText = "Train one in the Training Lab";
      break;
    }
    case "fuse":
      what =
        "A small gate network reads all four embeddings plus which sources are present, and gives each source a softmax weight. A missing source is masked out of the softmax entirely, so its zeros are never read as evidence.";
      if (D.abl?.withM1 && D.abl?.withoutM1) fact(`${ap4(D.abl.withM1.ap)} → ${ap4(D.abl.withoutM1.ap)}`, "AP when log templates are removed honestly (zeroed and marked missing)", "real", "RESULTS.md finding 4; tests/test_tessera_base.py");
      if (D.abl?.attr) fact(`${pct0(D.abl.attr.m1_log)} / ${pct0(D.abl.attr.m3_identity)}`, "mean gate weight on M1 and on M3. They understate M1, so read them as a hint, not proof", "real", "RESULTS.md finding 4 (NEGATIVE_RESULTS.md F9)");
      tab = "detector";
      tabText = "Switch a source off in the Live detector";
      break;
    case "score":
      what =
        "A small head turns the fused embedding into a probability between 0 and 1. Ranking quality is measured by average precision, which needs no threshold; MCC uses a fixed 0.5 cut-off.";
      if (D.tb)
        fact(
          ap4(D.tb.tessera_ap),
          `AP for TESSERA-base on one held-out fold (${D.tb.held_out}, MCC ${ap4(D.tb.tessera_mcc)}); a default-configured LightGBM on the same fold scores ${ap4(D.tb.lightgbm_ap)}. One fold, one run: an 8-fold paired comparison is future work (NEGATIVE_RESULTS C2)`,
          "real",
          "Transcribed from RESULTS.md finding 4; tests/test_tessera_base.py"
        );
      if (D.ap) fact(`${ap3(D.ap.mean)} ± ${ap3(D.ap.std)}`, `mean AP over ${D.ap.n} leave-one-replica-out folds (LightGBM baseline; ${D.excluded.join(", ")} kept out as low-support)`, "real", "Recomputed by the exporter; tests/test_loro_real.py");
      if (D.floor)
        fact(
          `${ap3(D.floor.summary.average_precision.mean)} ± ${ap3(D.floor.summary.average_precision.std)}`,
          "mean AP on the same folds for a model that sees only which feature groups are present: the floor the full model must beat",
          "real",
          "Recomputed by the exporter (real_results.json, mask_only_floor)"
        );
      tab = "results";
      tabText = "See every fold in Real results";
      break;
    case "ledger":
      what =
        "Each verdict is appended to an RFC 6962 Merkle log, the construction Certificate Transparency uses. A leaf commits to a hash of {window id, host hash, time bucket, verdict, score, model version}: never an IP, username or URL. Anyone holding an old root can detect a retroactive edit.";
      if (C?.ledger?.state === "done")
        fact(`${C.ledger.ok}/${C.ledger.n}`, "inclusion proofs written by the Python ledger verify in your browser, and the root matches exactly", "live", "Checked live against merkle_vectors.json, written by tessera/ledger/merkle.py");
      note("Building this port caught a real cross-language hashing bug (Python wrote a score as 0.0, JavaScript as 0). It was fixed in the production ledger, not worked around.");
      tab = "detector";
      tabText = "Tamper with it in the Live detector";
      break;
    default:
      break;
  }
  return { title, what, facts, tab, tabText, stage: n.stage };
}

function buildPipeline(getData) {
  const id = ++uid;
  const panelId = `ov-pipe-panel-${id}`;
  const arrowId = `ov-arrow-${id}`;
  const arrowHotId = `ov-arrow-hot-${id}`;
  const canvas = h("div", { class: "ov-canvas", role: "group", "aria-label": "The TESSERA pipeline in seven stages. Select a stage to see what it does." });
  const svg = s("svg", { class: "ov-wires", "aria-hidden": "true", focusable: "false" });
  canvas.appendChild(svg);

  const status = h("div", { class: "sr-only", "aria-live": "polite" });
  const panel = h("div", { class: "ov-panel", id: panelId });
  const section = h(
    "section",
    { class: "card ov-pipe", "data-tour": "overview-pipeline", "aria-labelledby": `ov-pipe-h-${id}` },
    h(
      "div",
      { class: "card-head" },
      h(
        "div",
        {},
        h("div", { class: "eyebrow" }, "How it works"),
        h("h2", { id: `ov-pipe-h-${id}` }, "From raw logs to a verdict you can audit"),
        h("div", { class: "card-sub" }, "Click any stage to see what it does, the real numbers behind it and which tab shows it.")
      ),
      h(
        "ul",
        { class: "ov-pipe-key tiny muted", "aria-label": "Colour key" },
        ...MODS.map((m) => {
          const li = h("li", {}, h("span", { class: "swatch" }), m.code);
          li.firstChild.style.background = m.color;
          return li;
        })
      )
    ),
    canvas,
    panel,
    status
  );

  const buttons = {};
  let prevBtnRef = null;
  let nextBtnRef = null;
  let L = null;
  let selected = "raw";
  let hover = null;
  let drawn = false;

  for (const n of PIPE) {
    const b = h("button", {
      class: `ov-node ov-node-${n.kind}`,
      type: "button",
      "data-node": n.id,
      "aria-pressed": "false",
      "aria-controls": panelId,
      "aria-label": `${n.kind === "mod" ? `${MODS[n.m].code} ${n.title}` : n.kind === "enc" ? `Encoder for ${MODS[n.m].code}` : n.title}: ${n.sub}. Stage ${n.stage} of ${N_STAGES}.`,
    });
    if (n.m != null) b.style.setProperty("--mc", MODS[n.m].color);
    b.style.setProperty("--i", String(n.stage - 1));

    const full = h("span", { class: "ov-node-full", "aria-hidden": "true" });
    const compact = h("span", { class: "ov-node-compact", "aria-hidden": "true" });
    if (n.kind === "mod") {
      full.append(
        h("span", { class: "ov-node-top" }, h("span", { class: "ov-node-code" }, MODS[n.m].code), h("span", { class: "ov-node-meta" }, n.sub)),
        h("span", { class: "ov-node-title" }, n.title)
      );
      compact.append(h("span", { class: "ov-node-code" }, n.cTitle), h("span", { class: "ov-node-sub" }, n.cSub));
    } else if (n.kind === "enc") {
      full.append(h("span", { class: "ov-node-title" }, n.title), h("span", { class: "ov-node-sub mono" }, n.sub));
      compact.append(h("span", { class: "ov-node-title" }, n.cTitle), h("span", { class: "ov-node-sub" }, n.cSub));
    } else if (n.kind === "fuse") {
      const gates = h("span", { class: "ov-gates" });
      MODS.forEach((m, k) => {
        const g = h("span", { class: `ov-gate ov-gate-${k + 1}` });
        g.style.setProperty("--mc", m.color);
        g.style.setProperty("--k", String(k));
        gates.appendChild(g);
      });
      full.append(h("span", { class: "ov-node-title" }, n.title), gates, h("span", { class: "ov-node-sub" }, n.sub));
      compact.append(h("span", { class: "ov-node-title" }, n.cTitle), gates.cloneNode(true), h("span", { class: "ov-node-sub" }, n.cSub));
    } else {
      full.append(h("span", { class: "ov-node-ico", html: icon(n.icon) }), h("span", { class: "ov-node-title" }, n.title), h("span", { class: "ov-node-sub" }, n.sub));
      compact.append(h("span", { class: "ov-node-ico", html: icon(n.icon) }), h("span", { class: "ov-node-title" }, n.cTitle), h("span", { class: "ov-node-sub" }, n.cSub));
    }
    b.append(full, compact);
    b.addEventListener("click", () => select(n.id, { announce: true }));
    b.addEventListener("pointerenter", () => setHover(n.id));
    b.addEventListener("pointerleave", () => setHover(null));
    b.addEventListener("focus", () => setHover(n.id));
    b.addEventListener("blur", () => setHover(null));
    buttons[n.id] = b;
    canvas.appendChild(b);
  }

  canvas.addEventListener("keydown", (e) => {
    const cur = e.target.closest?.(".ov-node")?.dataset.node;
    if (!cur) return;
    const i = FOCUS_ORDER.indexOf(cur);
    let j = -1;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") j = Math.min(FOCUS_ORDER.length - 1, i + 1);
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = Math.max(0, i - 1);
    else if (e.key === "Home") j = 0;
    else if (e.key === "End") j = FOCUS_ORDER.length - 1;
    if (j < 0) return;
    e.preventDefault();
    buttons[FOCUS_ORDER[j]].focus();
  });

  function related(idOrNull) {
    const set = new Set();
    if (!idOrNull) return set;
    set.add(idOrNull);
    for (const [a, b] of EDGES) {
      if (a === idOrNull) set.add(b);
      if (b === idOrNull) set.add(a);
    }
    return set;
  }

  function paintState() {
    const hot = new Set([selected, hover].filter(Boolean));
    const rel = related(hover);
    canvas.classList.toggle("has-hover", !!hover);
    for (const n of PIPE) {
      const b = buttons[n.id];
      b.setAttribute("aria-pressed", n.id === selected ? "true" : "false");
      b.classList.toggle("is-related", rel.has(n.id));
      b.classList.toggle("is-stage", n.id !== selected && PIPE_BY_ID[selected]?.stage === n.stage && n.stage === 4);
    }
    svg.querySelectorAll(".ov-edge").forEach((g) => {
      const on = hot.has(g.dataset.from) || hot.has(g.dataset.to);
      const onHover = hover && (g.dataset.from === hover || g.dataset.to === hover);
      g.classList.toggle("is-hot", on);
      g.classList.toggle("is-dim", !!hover && !onHover);
      const wire = g.querySelector(".ov-wire");
      if (wire) wire.setAttribute("marker-end", `url(#${on ? arrowHotId : arrowId})`);
    });
  }

  function setHover(idOrNull) {
    hover = idOrNull;
    paintState();
  }

  function layout() {
    const W = Math.round(canvas.clientWidth);
    if (W < 40) return;
    // Only the width drives the layout; setting our own height re-fires the observer.
    if (L && L.W === W) return;
    L = computeLayout(W);
    canvas.style.height = `${Math.round(L.H)}px`;
    canvas.classList.toggle("is-vertical", !L.horiz);
    canvas.classList.toggle("is-compact", L.compact);
    for (const n of PIPE) {
      const r = L.nodes[n.id];
      const b = buttons[n.id];
      b.style.left = `${Math.round(r.x)}px`;
      b.style.top = `${Math.round(r.y)}px`;
      b.style.width = `${Math.round(r.w)}px`;
      b.style.height = `${Math.round(r.h)}px`;
    }
    renderWires();
    paintState();
    if (!drawn) {
      drawn = true;
      setTimeout(() => section.classList.add("is-drawn"), reducedMotion() ? 0 : 1900);
    }
  }

  function renderWires() {
    svg.setAttribute("viewBox", `0 0 ${L.W} ${Math.round(L.H)}`);
    svg.setAttribute("width", String(L.W));
    svg.setAttribute("height", String(Math.round(L.H)));
    const marker = (mid, cls) =>
      s(
        "marker",
        { id: mid, viewBox: "0 0 10 10", refX: "8.6", refY: "5", markerWidth: "9", markerHeight: "9", markerUnits: "userSpaceOnUse", orient: "auto" },
        s("path", { class: cls, d: "M2 1.6 L8.6 5 L2 8.4" })
      );
    const defs = s("defs", {}, marker(arrowId, "ov-arrowhead"), marker(arrowHotId, "ov-arrowhead is-hot"));
    svg.replaceChildren(defs);
    const flowOn = !reducedMotion();
    for (const [from, to, k] of EDGES) {
      const d = edgePath(L, from, to, k);
      const g = s("g", { class: "ov-edge", "data-from": from, "data-to": to });
      const wire = s("path", { class: "ov-wire", d, pathLength: "1", "marker-end": `url(#${arrowId})` });
      wire.style.setProperty("--i", String(L.nodes[from].col));
      g.appendChild(wire);
      if (flowOn) {
        const flow = s("path", { class: "ov-flow", d });
        flow.style.stroke = k == null ? "var(--accent)" : MODS[k].color;
        flow.style.setProperty("--i", String(L.nodes[from].col));
        g.appendChild(flow);
      }
      svg.appendChild(g);
    }
  }

  function renderPanel({ swap = true } = {}) {
    const { D, C } = getData();
    const ex = explain(selected, D || {}, C);
    const seqIdx = SEQ.indexOf(/^e[1-4]$/.test(selected) ? "e1" : selected);
    const prevBtn = h("button", {
      class: "icon-btn ov-step-btn",
      type: "button",
      "aria-label": "Previous stage",
      title: "Previous stage",
      html: icon("arrowLeft"),
      disabled: seqIdx <= 0,
      onClick: () => select(SEQ[Math.max(0, seqIdx - 1)], { announce: true }),
    });
    const nextBtn = h("button", {
      class: "icon-btn ov-step-btn",
      type: "button",
      "aria-label": "Next stage",
      title: "Next stage",
      html: icon("arrowRight"),
      disabled: seqIdx >= SEQ.length - 1,
      onClick: () => select(SEQ[Math.min(SEQ.length - 1, seqIdx + 1)], { announce: true }),
    });

    const factsList = h("ul", { class: "ov-facts" });
    for (const f of ex.facts) {
      if (f.value == null) {
        factsList.appendChild(h("li", { class: "ov-fact ov-fact-note" }, h("span", { class: "ov-fact-dot", html: icon("info") }), h("span", { class: "ov-fact-text" }, f.text)));
        continue;
      }
      factsList.appendChild(
        h(
          "li",
          { class: "ov-fact" },
          h("span", { class: "ov-fact-value tabular" }, f.value),
          h(
            "span",
            { class: "ov-fact-body" },
            h("span", { class: "ov-fact-text" }, f.text),
            h("span", { class: "ov-fact-src" }, tag(f.kind, f.source), h("span", { class: "tiny faint" }, f.source))
          )
        )
      );
    }
    if (!ex.facts.length) factsList.appendChild(h("li", { class: "small faint" }, "Loading the recorded numbers…"));

    const n = PIPE_BY_ID[selected];
    const badge = h("span", { class: "ov-panel-badge", html: n.icon ? icon(n.icon) : n.kind === "fuse" ? icon("layers") : n.kind === "enc" ? icon("cpu") : icon("file") });
    if (n.m != null) badge.style.setProperty("--mc", MODS[n.m].color);
    if (n.m != null) badge.classList.add("is-mod");

    const inner = h(
      "div",
      { class: "ov-panel-inner" },
      h(
        "div",
        { class: "ov-panel-head" },
        h("div", { class: "ov-panel-title" }, badge, h("div", {}, h("div", { class: "eyebrow" }, `Stage ${ex.stage} of ${N_STAGES}`), h("h3", {}, ex.title))),
        h("div", { class: "ov-panel-nav" }, prevBtn, nextBtn)
      ),
      h(
        "div",
        { class: "ov-panel-body" },
        h("div", { class: "ov-panel-what" }, h("div", { class: "label" }, "What it does"), h("p", {}, ex.what)),
        h("div", { class: "ov-panel-nums" }, h("div", { class: "label" }, "The numbers behind it"), factsList)
      ),
      h("div", { class: "ov-panel-foot" }, tabLink(ex.tab, ex.tabText, "btn sm"))
    );
    if (swap && !reducedMotion()) inner.classList.add("ov-swap");
    // The step buttons are rebuilt with the panel; keep keyboard focus on the one
    // that was pressed (or its partner, once it becomes disabled at either end).
    const active = document.activeElement;
    const hadFocus = active === prevBtnRef || active === nextBtnRef ? (active === prevBtnRef ? "prev" : "next") : null;
    panel.replaceChildren(inner);
    prevBtnRef = prevBtn;
    nextBtnRef = nextBtn;
    if (hadFocus) {
      const want = hadFocus === "prev" ? prevBtn : nextBtn;
      const other = hadFocus === "prev" ? nextBtn : prevBtn;
      (want.disabled ? other : want).focus({ preventScroll: true });
    }
    return ex;
  }

  function select(nodeId, { announce = false } = {}) {
    selected = nodeId;
    paintState();
    const ex = renderPanel();
    if (announce) status.textContent = `Stage ${ex.stage} of ${N_STAGES}: ${ex.title}`;
  }

  if (typeof ResizeObserver === "function") new ResizeObserver(debounce(() => layout(), 60)).observe(canvas);
  else window.addEventListener("resize", debounce(() => layout(), 120));

  return {
    el: section,
    layout,
    refresh: () => renderPanel({ swap: false }),
    init() {
      layout();
      paintState();
      renderPanel({ swap: false });
    },
    pause(v) {
      section.classList.toggle("is-paused", !!v);
    },
  };
}

/* ------------------------------------------------------------------ what makes it different */

function miniBars(rows, { ariaLabel, domain = 1, format = (v) => v.toFixed(3), barColor = "var(--series-1)" } = {}) {
  const box = h("div", { class: "ov-mini", role: "img", "aria-label": ariaLabel });
  for (const r of rows) {
    const v = Math.max(0, r.value) / domain;
    const w = v > 0 ? `max(2px, ${(Math.min(1, v) * 100).toFixed(2)}%)` : "0%";
    const fill = h("div", { class: "ov-mini-fill" });
    fill.style.setProperty("--w", w);
    fill.style.setProperty("--bar", r.color || barColor);
    const row = h(
      "div",
      { class: "ov-mini-row" },
      h("span", { class: "ov-mini-label" }, r.label),
      h("div", { class: "ov-mini-track" }, fill),
      h("span", { class: "ov-mini-val tabular" }, r.display ?? format(r.value))
    );
    row.addEventListener("pointermove", (e) =>
      showTooltip({ title: r.label, rows: [{ label: r.metric || "Value", value: r.display ?? format(r.value), color: r.color || barColor, shape: "rect" }], note: r.note }, e.clientX, e.clientY, box)
    );
    row.addEventListener("pointerleave", () => hideTooltip(box));
    box.appendChild(row);
  }
  inView(box, () => requestAnimationFrame(() => box.classList.add("is-in")));
  return box;
}

function buildDiff(D, C) {
  const cards = [];

  // 1. leakage
  {
    const proof = h("div", { class: "ov-proof" });
    if (D.leak) {
      proof.append(
        miniBars(
          [
            { label: "Random split", value: D.leak.r0_random_ap, metric: "Average precision", note: "Leaky: duplicate rows sit on both sides" },
            { label: "Chronological", value: D.leak.r1_chronological_ap, metric: "Average precision", note: "Honest: train on the past, test on the future" },
          ],
          { ariaLabel: `Average precision on the same data: random split ${ap3(D.leak.r0_random_ap)}, chronological split ${ap3(D.leak.r1_chronological_ap)}` }
        ),
        h(
          "div",
          { class: "ov-proof-src" },
          tag("real", "Recomputed by the exporter with the protocol of tests/test_pipeline_real_data.py"),
          h("span", { class: "tiny faint" }, `AP, ${D.leak.replica}, network metrics only · ${fmt.int(D.leak.test_rows_identical_to_train)} test rows identical to a training row`)
        )
      );
    }
    cards.push({
      icon: "split",
      title: "Leakage-instrumented evaluation",
      text: "A random split lets near-copies of test windows sit in training, so scores inflate. TESSERA holds out whole replicas or cuts by time, and every Lab run prints a leakage certificate before it trains.",
      proof,
      tab: "lab",
      link: "Try the splits in the Training Lab",
    });
  }

  // 2. missing source
  {
    const proof = h("div", { class: "ov-proof" });
    const b = D.avail?.benign;
    const a = D.avail?.attack;
    if (b && a) {
      proof.append(
        miniBars(
          [
            { label: "Benign windows", value: b.m1, display: pct0(b.m1), color: "var(--c-benign)", metric: "Log templates present" },
            { label: "Attack windows", value: a.m1, display: pct0(a.m1), color: "var(--c-attack)", metric: "Log templates present" },
          ],
          { ariaLabel: `Log templates are present in ${pct0(b.m1)} of benign windows and ${pct0(a.m1)} of attack windows` }
        ),
        h("div", { class: "ov-proof-src" }, tag("real", "data/replica_stats.json: aggregate availability pooled over all 8 real replicas"), h("span", { class: "tiny faint" }, "Share of windows with any log-template activity"))
      );
    }
    if (D.floor)
      proof.append(
        h(
          "div",
          { class: "ov-proof-line" },
          h("span", { class: "ov-proof-num tabular" }, ap3(D.floor.summary.average_precision.mean)),
          h("span", { class: "small muted" }, "The same gap is a shortcut: labels come only from the logs. A model that sees nothing but which sources are present scores this mean AP across held-out replicas."),
          tag("real", "Recomputed by the exporter (real_results.json, mask_only_floor)")
        )
      );
    if (D.abl?.withM1 && D.abl?.withoutM1)
      proof.append(
        h(
          "div",
          { class: "ov-proof-line" },
          h("span", { class: "ov-proof-num tabular" }, `${ap4(D.abl.withM1.ap)} → ${ap4(D.abl.withoutM1.ap)}`),
          h("span", { class: "small muted" }, "AP when log templates are removed honestly (zeroed and marked missing)"),
          tag("real", "RESULTS.md finding 4; tests/test_tessera_base.py")
        )
      );
    cards.push({
      icon: "layers",
      title: "Handles a missing feature group",
      text: "Real sources go quiet: most benign windows have no log activity at all. The fusion gate drops a missing group from its softmax instead of reading its zeros as evidence.",
      proof,
      tab: "detector",
      link: "Switch a source off in the Live detector",
    });
  }

  // 3. ledger
  {
    const proof = h("div", { class: "ov-proof ov-proof-ledger" });
    const line = h("div", { class: "ov-proof-line" });
    const paint = (CC) => {
      line.replaceChildren();
      if (CC?.ledger?.state === "done" && CC?.tamper?.state === "done") {
        line.append(
          h("span", { class: "ov-proof-num tabular" }, `${CC.ledger.ok}/${CC.ledger.n}`),
          h("span", { class: "small muted" }, "Python ledger proofs verified in your browser; an edit to entry " + (CC.tamper.index + 1) + " changed the root"),
          tag("live", "Checked just now against merkle_vectors.json (written by tessera/ledger/merkle.py)")
        );
      } else if (CC?.ledger?.state === "error") {
        line.append(h("span", { class: "small bad-ink", html: `${icon("alert")} The live ledger check could not run: ${escapeHtml(CC.ledger.message || "failed")}` }));
      } else {
        line.append(h("span", { class: "small faint", html: `<span class="ov-spin" aria-hidden="true"></span> Checking the ledger in your browser…` }));
      }
    };
    paint(C);
    proof.append(
      h(
        "ul",
        { class: "ov-ticks small" },
        h("li", { html: `${icon("check")}<span>Only a hash of {window id, host hash, time bucket, verdict, score, model version}</span>` }),
        h("li", { html: `${icon("x")}<span>Never an IP address, username, URL or request body</span>` })
      ),
      line
    );
    cards.push({
      icon: "lock",
      title: "Tamper-evident verdicts, no raw telemetry",
      text: "Every verdict becomes a leaf in an RFC 6962 Merkle log. Change any past verdict and the root changes, so anyone who kept the old root can detect the edit.",
      proof,
      tab: "detector",
      link: "Try to tamper with it",
      update: paint,
    });
  }

  // 4. audit
  {
    const proof = h("div", { class: "ov-proof" });
    if (D.opt?.eho && D.opt?.random) {
      const rows = [D.opt.eho, D.opt.random, D.opt.tpe].filter(Boolean).map((r) => ({
        label: /EHO/.test(r.optimiser) ? "Paper's EHO" : /random/i.test(r.optimiser) ? "Random search" : "Optuna TPE",
        value: r.optimality_gap_pct,
        // Same format as the Design Decisions tab, so both show the identical figure.
        display: r.optimality_gap_pct === 0 ? "0%" : r.optimality_gap_pct < 0.1 ? `${r.optimality_gap_pct.toFixed(3)}%` : `${Number(r.optimality_gap_pct.toFixed(1))}%`,
        metric: "Optimality gap",
        note: `Best segment length ${fmt.int(r.best_segment_length)}, same budget`,
      }));
      const max = Math.max(...rows.map((r) => r.value));
      proof.append(
        miniBars(rows, { ariaLabel: `Optimality gap on the same budget: ${rows.map((r) => `${r.label} ${r.display}`).join(", ")}`, domain: max > 0 ? max : 1 }),
        h("div", { class: "ov-proof-src" }, tag("measured", "Recomputed by the exporter: tessera.chainsim.benchmark.run (the calls tessera.chainsim.report makes)"), h("span", { class: "tiny faint" }, "Optimality gap vs. exhaustive search, lower is better"))
      );
    }
    if (D.fh && isNum(D.fh.absolute_spread))
      proof.append(
        h(
          "div",
          { class: "ov-proof-line" },
          h("span", { class: "ov-proof-num tabular" }, String(D.fh.absolute_spread)),
          h("span", { class: "small muted" }, `spread of the paper's objective across ${D.fh.nsc_values?.length ?? "all"} values of the variable it optimises`),
          tag("measured", "Recomputed by the exporter: tessera.chainsim.segment_objective.demonstrate_objective_invariance")
        )
      );
    cards.push({
      icon: "design",
      title: "Every design choice has a receipt",
      text: "Every non-obvious decision — the metric, the split protocol, the optimiser, the ledger's privacy design — is backed by a real measurement, not intuition. Mistakes are logged the same way as findings.",
      proof,
      tab: "design",
      link: "See the design decisions",
    });
  }

  const grid = h("div", { class: "grid-2 ov-diff-grid" });
  const updaters = [];
  cards.forEach((c, i) => {
    const card = h(
      "article",
      { class: "card lift ov-diff" },
      h("div", { class: "ov-diff-head" }, h("span", { class: "ov-diff-icon", html: icon(c.icon) }), h("h3", {}, c.title)),
      h("p", { class: "ov-diff-text" }, c.text),
      c.proof,
      h("div", { class: "ov-diff-foot" }, tabLink(c.tab, c.link))
    );
    card.style.setProperty("--i", String(i));
    grid.appendChild(card);
    if (c.update) updaters.push(c.update);
  });
  stagger(grid);

  const el = h(
    "section",
    { class: "ov-section reveal-on-scroll", "aria-labelledby": "ov-diff-h" },
    h("div", { class: "ov-section-head" }, h("div", {}, h("div", { class: "eyebrow" }, "Why it's different"), h("h2", { id: "ov-diff-h" }, "Four things this project does differently"))),
    grid
  );
  return { el, update: (CC) => updaters.forEach((u) => u(CC)) };
}

/* ------------------------------------------------------------------ live self-check card */

function buildSelfCheck(store) {
  const rows = {
    model: { label: "The model in this page matches PyTorch", idle: "Re-scores 40 reference windows the real PyTorch model scored" },
    ledger: { label: "The ledger in this page matches Python", idle: "Rebuilds a log the Python ledger wrote and checks every proof" },
    tamper: { label: "Editing history is detected", idle: "Forges one past verdict and checks the proofs catch it" },
  };
  const stepper = h("div", { class: "stepper ov-check", role: "list" });
  const els = {};
  for (const [key, r] of Object.entries(rows)) {
    const iconEl = h("span", { class: "step-icon", "aria-hidden": "true" });
    const label = h("div", { class: "step-label" }, r.label);
    const detail = h("div", { class: "step-detail" }, r.idle);
    const pctEl = h("span", { class: "step-pct" });
    const step = h("div", { class: "step", role: "listitem", "data-state": "pending" }, iconEl, h("div", { class: "step-body" }, label, detail), pctEl);
    els[key] = { step, iconEl, detail, pctEl, idle: r.idle };
    stepper.appendChild(step);
  }
  const live = h("div", { class: "sr-only", "aria-live": "polite" });
  const rerun = h("button", { class: "btn sm", type: "button", html: `${icon("reset")}<span>Run again</span>`, disabled: true });
  const summary = h("div", { class: "ov-check-summary small muted" }, "Waiting to run…");

  let last = null;
  let running = false;
  const listeners = [];

  function paint(R) {
    last = R;
    const set = (key, st, detailText) => {
      const e = els[key];
      e.step.dataset.state = st;
      e.iconEl.innerHTML = st === "done" ? icon("check") : st === "error" ? icon("x") : st === "running" ? '<span class="ov-spin" aria-hidden="true"></span>' : "";
      e.detail.textContent = detailText;
      e.pctEl.replaceChildren(
        st === "done"
          ? h("span", { class: "pill good", html: `${icon("check")}Pass` })
          : st === "error"
            ? h("span", { class: "pill bad", html: `${icon("x")}Fail` })
            : st === "running"
              ? h("span", { class: "tiny faint" }, "Running")
              : h("span", { class: "tiny faint" }, "Queued")
      );
    };
    const m = R.model;
    set(
      "model",
      m.state,
      m.state === "done" || (m.state === "error" && m.n)
        ? `${m.pass}/${m.n} reference windows match (tolerance 1e-4), ${m.masked} of them with a source switched off. Largest score difference ${m.maxDelta.toExponential(1)}.`
        : m.state === "error"
          ? `Could not run: ${m.message}`
          : els.model.idle
    );
    const l = R.ledger;
    set(
      "ledger",
      l.state,
      l.state === "done" || (l.state === "error" && l.n)
        ? `Root ${l.rootOk ? "matches" : "does not match"} Python exactly; ${l.ok}/${l.n} inclusion proofs match and verify. Root ${l.rootHex.slice(0, 12)}…`
        : l.state === "error"
          ? `Could not run: ${l.message}`
          : els.ledger.idle
    );
    const t = R.tamper;
    set(
      "tamper",
      t.state,
      t.state === "done" || (t.state === "error" && t.before)
        ? `Forged entry ${t.index + 1}: root ${t.before.slice(0, 8)}… became ${t.after.slice(0, 8)}…; the old proof ${t.staleOnNew ? "still verifies (bad)" : "no longer verifies against the new root"}, and still verifies against the root an auditor kept.`
        : t.state === "error"
          ? `Could not run: ${t.message}`
          : els.tamper.idle
    );
    const states = [m.state, l.state, t.state];
    if (states.every((x) => x === "done")) {
      summary.replaceChildren(h("span", { class: "pill good", html: `${icon("check")}All three checks passed` }), h("span", {}, " in this browser, just now. Nothing was sent anywhere."));
      live.textContent = "All three self-checks passed.";
    } else if (states.includes("error")) {
      summary.replaceChildren(h("span", { class: "pill bad", html: `${icon("alert")}A check did not pass` }), h("span", {}, " The details are above."));
      live.textContent = "A self-check did not pass.";
    } else {
      summary.textContent = "Running in your browser…";
    }
    listeners.forEach((fn) => fn(R));
  }

  async function run() {
    if (running) return;
    running = true;
    rerun.disabled = true;
    for (const key of Object.keys(els)) {
      els[key].step.dataset.state = "pending";
    }
    await runSelfCheck(store, paint);
    running = false;
    rerun.disabled = false;
  }
  rerun.addEventListener("click", run);

  const el = h(
    "section",
    { class: "card ov-section-card reveal-on-scroll", "aria-labelledby": "ov-check-h" },
    h(
      "div",
      { class: "card-head" },
      h(
        "div",
        {},
        h("div", { class: "eyebrow" }, "Don't take our word for it"),
        h("h2", { id: "ov-check-h" }, "Checked in your browser, just now"),
        h("div", { class: "card-sub" }, "The same parity checks the repository's tests run, re-run here against the outputs the real PyTorch model and Python ledger wrote.")
      ),
      rerun
    ),
    stepper,
    h("div", { class: "card-foot ov-check-foot" }, summary, tag("live", "Measured in this browser"), live)
  );

  return {
    el,
    /** Run once `trigger` (default: this card) scrolls into view. */
    start: (trigger = el) => inView(trigger, run),
    onResult: (fn) => {
      listeners.push(fn);
      if (last) fn(last);
    },
    get last() {
      return last;
    },
  };
}

/* ------------------------------------------------------------------ explore strip */

function buildExplore() {
  const items = [
    { tab: "detector", title: "Live detector", text: "Score a window, switch a feature group off, then try to tamper with the ledger." },
    { tab: "lab", title: "Training Lab", text: "Generate synthetic data, pick a split, train the model in your browser and read its leakage certificate." },
    { tab: "results", title: "Real results", text: "The measured numbers on real, held-out AIT data, with where each one came from." },
    { tab: "design", title: "Design decisions", text: "Every non-obvious design choice, next to the evidence behind it." },
    { tab: "about", title: "About", text: "Model card, how we evaluate, commands to reproduce it all, and a viva FAQ." },
  ];
  const list = h("ol", { class: "ov-explore" });
  items.forEach((it, i) => {
    const a = h(
      "a",
      { class: "ov-explore-item", href: `#${it.tab}` },
      h("span", { class: "ov-explore-num", "aria-hidden": "true" }, String(i + 1)),
      h("span", { class: "ov-explore-ico", html: icon(it.tab) }),
      h("span", { class: "ov-explore-title" }, it.title),
      h("span", { class: "ov-explore-text" }, it.text),
      h("span", { class: "ov-explore-go", html: `<span>Open</span>${icon("arrowRight")}` })
    );
    const li = h("li", {}, a);
    li.style.setProperty("--i", String(i));
    list.appendChild(li);
  });
  stagger(list);
  return h(
    "section",
    { class: "ov-section reveal-on-scroll", "aria-labelledby": "ov-explore-h" },
    h("div", { class: "ov-section-head" }, h("div", {}, h("div", { class: "eyebrow" }, "Where next"), h("h2", { id: "ov-explore-h" }, "How to explore this site"))),
    list
  );
}

/* ------------------------------------------------------------------ mount */

export async function mount(el, ctx) {
  const { store } = ctx;
  el.replaceChildren();
  const root = h("div", { class: "ov" });
  el.appendChild(root);

  const state = { D: {}, C: null };
  const hero = buildHero(ctx);
  const stats = buildStats();
  const pipe = buildPipeline(() => state);
  root.append(hero, stats.el, pipe.el);
  stagger(root);
  // Lay the diagram out now so it never flashes unpositioned; its numbers fill in below.
  pipe.init();

  // Everything numeric waits for the data files; the hero is interactive immediately.
  const data = await loadData(store);
  state.D = derive(data);
  pipe.refresh();

  const diff = buildDiff(state.D, null);
  const check = buildSelfCheck(store);
  const explore = buildExplore();
  root.append(diff.el, check.el, explore);

  if (!state.D.ok.real) {
    root.insertBefore(
      h("div", { class: "notice warn", html: `${icon("alert", { cls: "notice-icon" })}<div><strong>The real-results file did not load.</strong> Headline numbers are shown as dashes rather than guessed. Serve the site over http (see About) and reload.</div>` }),
      stats.el.nextSibling
    );
  }

  check.onResult((C) => {
    state.C = C;
    diff.update(C);
    pipe.refresh();
  });

  stats.fill(state.D);
  inView(stats.grid, () => stats.play());
  revealOnScroll(root);
  // The self-check is cheap; start it as soon as the section above it comes
  // into view so its three steps visibly tick over as the viewer scrolls.
  check.start(diff.el);

  return {
    onShow() {
      pipe.pause(false);
      pipe.layout();
    },
    onHide() {
      pipe.pause(true);
      hideTooltip();
    },
  };
}
