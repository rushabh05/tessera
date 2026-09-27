// Real Results tab: what the detector does on REAL, held-out AIT data.
//
// Every number on this page is read from data/real_results.json. The Python
// exporter (uv run python -m tessera.demo.export_web_data) either recomputes a
// block from the local real-data cache or transcribes it from RESULTS.md, and
// each block carries a `source` naming the test or command that produced it.
// This module never computes anything from real rows (there are none on the
// site); it only formats, compares and draws those exported aggregates. The one
// thing on the page that is NOT data is the small split sketch in finding 1,
// which is labelled as an illustration.
//
// Motion: the head and hero stagger in, cards reveal as they scroll into view,
// and every chart / count-up is created when it becomes visible so its draw-in
// animation is actually seen. All of it is skipped under prefers-reduced-motion.

import { h, s, fmt, countUp, stagger, revealOnScroll, reducedMotion, copyText, escapeHtml, toast as domToast } from "../ui/dom.js";
import { icon } from "../ui/icons.js";
import { barChart, withTableToggle, hideTooltip } from "../ui/charts.js";

const MINUS = "−";

const MODALITIES = [
  { id: "m1_log", code: "M1", label: "Log templates", color: "--series-1" },
  { id: "m2_metrics", code: "M2", label: "Network metrics", color: "--series-2" },
  { id: "m3_identity", code: "M3", label: "Host identity", color: "--series-3" },
  { id: "m4_graph", code: "M4", label: "Graph structure", color: "--series-7" },
];

// The reproduce commands recorded in RESULTS.md (and the exporter that wrote this page's data).
const CMD = {
  leakage: "uv run pytest tests/test_pipeline_real_data.py tests/test_cross_replica.py -q -v",
  loro: "uv run pytest tests/test_loro_real.py -q -v",
  neural: "uv run pytest tests/test_tessera_base.py -q -v -m slow",
  export: "uv run python -m tessera.demo.export_web_data",
  floor: "uv run pytest tests/test_web_exports.py -q -k mask_only_floor",
  parity: "cd web && npm test",
};

const COPY_ICON =
  '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 012-2h9"/></svg>';

/* ------------------------------------------------------------------ formatting */

const num = (v, d = 3) => fmt.num(v, d);
const int = (v) => fmt.int(v);
const pct = (v, d = 1) => fmt.pct(v, d);
/** Axis-friendly AP formatter: round ticks print short (0.2, 1.0), data values keep `dp` decimals. */
const apFormat = (dp) => (v) => {
  if (!Number.isFinite(v)) return "—";
  return Math.abs(v * 10 - Math.round(v * 10)) < 1e-9 ? v.toFixed(1) : v.toFixed(dp);
};
/** A gain in AP points, always signed: "+29", or "−3" if the random split scored lower. */
const signedPts = (v) => (Number.isFinite(v) ? `${v < 0 ? MINUS : "+"}${Math.round(Math.abs(v))}` : "—");
/** "200-tree" from loro.model ("LightGBM, 200 trees, seed 0, ..."), or null. */
function treeCount(data) {
  const m = /(\d[\d,]*)\s*trees/i.exec(String(data.loro?.model || ""));
  return m ? m[1] : null;
}

/* ------------------------------------------------------------------ lifecycle helpers */

/**
 * Deferred "when visible" hooks. Builders register callbacks while the page is
 * still detached; start() runs them once the page is in the document. Under
 * reduced motion (or without IntersectionObserver) callbacks run at start().
 */
function makeEnv(notify) {
  const queue = [];
  const observers = [];
  let live = false;
  function observe(el, fn) {
    if (reducedMotion() || typeof IntersectionObserver !== "function") {
      fn();
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          io.disconnect();
          fn();
        }
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0 }
    );
    io.observe(el);
    observers.push(io);
  }
  return {
    notify,
    onVisible(el, fn) {
      if (live) observe(el, fn);
      else queue.push([el, fn]);
    },
    start() {
      live = true;
      for (const [el, fn] of queue.splice(0)) observe(el, fn);
    },
  };
}

/** A chart that is created the first time its host scrolls into view. Same {update, destroy} API. */
function lazyChart(env, host, factory, opts, minHeight = 0) {
  let chart = null;
  let pending = { ...opts };
  let dead = false;
  if (minHeight) host.style.minHeight = `${minHeight}px`;
  env.onVisible(host, () => {
    if (dead) return;
    chart = factory(host, pending);
    host.style.minHeight = "";
  });
  return {
    update(next = {}) {
      if (chart) chart.update(next);
      else pending = { ...pending, ...next };
    },
    destroy() {
      dead = true;
      if (chart) chart.destroy();
    },
  };
}

/** Put the final value in the element now (screen readers, no-JS), count up when it is seen. */
function countIn(env, el, to, format, duration = 1000) {
  el.textContent = format(to);
  if (!Number.isFinite(to)) return el;
  env.onVisible(el, () => countUp(el, to, { from: 0, duration, format }));
  return el;
}

function section(name, build) {
  try {
    return build();
  } catch (err) {
    console.error(`results tab: the "${name}" section failed`, err);
    return h("div", { class: "mount-error" }, `The "${name}" section could not be drawn: ${err.message}. The rest of the page still works.`);
  }
}

function scrollToCard(root, id) {
  const target = root.querySelector(`#${id}`);
  if (!target) return;
  target.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
  target.focus({ preventScroll: true });
}

/* ------------------------------------------------------------------ small components */

/** The visible marker for a transcribed one-off ablation that no test re-runs. */
function notRerunPill() {
  return h("span", { class: "pill warn res-notrerun", html: `${icon("alert")}<span>Not re-run by any test</span>` });
}

const ico = (name, cls = "") => h("span", { class: `res-ico ${cls}`.trim(), html: icon(name) });

function pill(kind, iconName, text) {
  return h("span", { class: `pill ${kind}`, html: `${icon(iconName)}<span>${escapeHtml(text)}</span>` });
}

function takeaway(text) {
  return h("p", { class: "res-takeaway" }, text);
}

function whyBlock(title, ...paras) {
  return h(
    "div",
    { class: "res-why" },
    h("div", { class: "res-why-title" }, ico("info"), h("span", {}, title)),
    ...paras.map((p) => h("p", {}, p))
  );
}

function cmdLine(env, cmd, label = "Reproduce") {
  const btn = h("button", { class: "btn sm ghost res-copy", type: "button", "aria-label": `Copy the command: ${cmd}`, html: `${COPY_ICON}<span>Copy</span>` });
  let timer = 0;
  btn.addEventListener("click", async () => {
    const ok = await copyText(cmd);
    clearTimeout(timer);
    if (ok) {
      btn.classList.add("is-copied");
      btn.innerHTML = `${icon("check")}<span>Copied</span>`;
      timer = setTimeout(() => {
        btn.classList.remove("is-copied");
        btn.innerHTML = `${COPY_ICON}<span>Copy</span>`;
      }, 1600);
    }
    env.notify(ok ? "Command copied to the clipboard" : "Could not copy automatically: select the command and copy it by hand", { kind: ok ? "good" : "warn" });
  });
  return h("div", { class: "res-cmd" }, label ? h("span", { class: "res-cmd-label" }, label) : null, h("code", { class: "res-cmd-code", tabindex: "0" }, cmd), btn);
}

/** items: [{provenance: 'recomputed'|'transcribed', source, label?}] */
function provenanceFoot(env, { items, command }) {
  const kinds = [...new Set(items.map((it) => (it.provenance === "transcribed" ? "transcribed" : "recomputed")))];
  const pills = kinds.map((k) =>
    k === "transcribed" ? pill("neutral", "file", "Transcribed from RESULTS.md") : pill("info", "database", "Recomputed from the real data cache")
  );
  return h(
    "footer",
    { class: "card-foot res-foot" },
    h("div", { class: "res-foot-row" }, h("span", { class: "res-foot-label" }, "Provenance"), h("span", { class: "tag-real" }, "Real"), ...pills),
    command ? cmdLine(env, command) : null,
    h(
      "details",
      { class: "res-src" },
      h("summary", {}, items.length > 1 ? `Where these numbers come from (${items.length} sources)` : "Where these numbers come from"),
      ...items.map((it) => h("p", {}, it.label ? h("strong", {}, `${it.label}: `) : null, it.source || "No source recorded."))
    )
  );
}

function findingCard({ n, id, tour, title }) {
  const card = h("section", { class: "card res-card reveal-on-scroll", id, tabindex: "-1", "aria-labelledby": `${id}-title`, "data-tour": tour || null });
  card.append(
    h(
      "div",
      { class: "res-card-head" },
      h("div", { class: "res-kicker" }, h("span", { class: "res-num" }, n ? `Finding ${n}` : "Reference"), h("span", { class: "tag-real" }, "Real data")),
      h("h2", { class: "res-card-title", id: `${id}-title` }, title)
    )
  );
  return card;
}

function chip(env, { value, format, label, sub }) {
  const v = h("span", { class: "res-chip-value" });
  countIn(env, v, value, format, 900);
  return h("div", { class: "res-chip" }, v, h("span", { class: "res-chip-label" }, label), sub ? h("span", { class: "res-chip-sub" }, sub) : null);
}

function legend(items) {
  return h(
    "div",
    { class: "viz-legend res-legend" },
    ...items.map((it) =>
      h("span", { class: "item" }, h("span", { class: `viz-key ${it.shape || "rect"}`, style: it.shape === "ref" ? {} : { background: `var(${it.color})` } }), h("span", {}, it.label))
    )
  );
}

function panel(title, sub, ...children) {
  return h("div", { class: "res-panel" }, h("div", { class: "res-panel-title" }, title), sub ? h("div", { class: "res-panel-sub" }, sub) : null, ...children);
}

/* ------------------------------------------------------------------ page head + reading guide + hero */

function pageHead(data) {
  const ds = data.dataset || {};
  const nRep = Array.isArray(ds.replicas) ? ds.replicas.length : null;
  const nWin = ds.totals?.n_windows;
  const scope = nRep && Number.isFinite(nWin) ? ` (${nRep} replicas, ${int(nWin)} one-minute windows)` : "";
  return h(
    "header",
    { class: "page-head res-head" },
    h(
      "div",
      {},
      h("div", { class: "eyebrow" }, "Real results"),
      h("h1", {}, "What the detector does on real, held-out data"),
      h(
        "p",
        { class: "lede" },
        `These numbers come from recorded runs on the real AIT Log Data Set${scope}, not from this browser. Most findings name the test that re-runs them; transcribed one-off ablations are marked "Not re-run by any test".`
      ),
      h(
        "div",
        { class: "res-tags" },
        h("span", { class: "tag-real" }, "Real data"),
        h("span", { class: "badge" }, h("span", { class: "dot", style: { color: "var(--text-3)" } }), "Held-out test sets only"),
        h("span", { class: "badge" }, h("span", { class: "dot", style: { color: "var(--text-3)" } }), "Aggregates only, no raw rows on this site")
      )
    )
  );
}

function readingGuide(data) {
  const santos = data.loro?.folds?.find((f) => f.held_out === "santos");
  const prev = santos?.test_prevalence;
  const accText = Number.isFinite(prev)
    ? `On held-out santos ${pct(1 - prev, 1)} of windows are benign, so a detector that never raises an alarm already scores ${pct(1 - prev, 1)} accuracy. We report it nowhere as a headline.`
    : "When most windows are benign, a detector that never raises an alarm still looks accurate, so accuracy is never the headline here.";
  const item = (iconName, title, tagEl, text, muted = false) =>
    h("div", { class: `res-guide-item${muted ? " is-muted" : ""}` }, h("h3", {}, ico(iconName), h("span", {}, title), tagEl), h("p", {}, text));
  return h(
    "div",
    { class: "res-guide", role: "note", "aria-label": "How to read the numbers on this page" },
    item("gauge", "Average precision (AP)", h("span", { class: "pill info" }, "Headline"), "How well attacks are ranked above benign windows, across every threshold. 1.0 is perfect; guessing scores about the share of attack windows."),
    item("check", "Matthews correlation (MCC)", h("span", { class: "pill neutral" }, "Second"), "How good the yes/no calls are at a fixed 0.5 threshold, from −1 to 1. Zero means no better than chance."),
    item("alert", "Accuracy", h("span", { class: "pill neutral" }, "De-emphasised"), accText, true)
  );
}

function heroStat(env, root, { label, iconName, value, format, suffix, sub, target, primary = false, aria }) {
  const val = h("span", { class: "res-cu" });
  countIn(env, val, value, format, 1100);
  return h(
    "button",
    { type: "button", class: `stat res-stat${primary ? " primary" : ""}`, "aria-label": aria, onClick: () => scrollToCard(root, target) },
    h("span", { class: "stat-label" }, ico(iconName, "res-stat-ico"), h("span", {}, label)),
    h("span", { class: "stat-value tabular" }, val, suffix ? h("span", { class: "res-pm" }, suffix) : null),
    h("span", { class: "stat-sub" }, sub),
    h("span", { class: "res-stat-go" }, h("span", {}, "See the evidence"), ico("arrowRight"))
  );
}

function hero(env, root, data) {
  const sum = data.loro?.summary || {};
  const ap = sum.average_precision || {};
  const excluded = sum.excluded_low_support || [];
  const lk = data.leakage_duplicates || {};
  const tb = data.tessera_base || {};
  const drop = (lk.r0_random_ap - lk.r1_chronological_ap) * 100;
  const grid = h("div", { class: "grid-4 res-hero", "data-tour": "results-hero" });
  grid.append(
    heroStat(env, root, {
      primary: true,
      label: "Mean AP on held-out replicas (LightGBM baseline)",
      iconName: "gauge",
      value: ap.mean,
      format: (v) => num(v, 3),
      suffix: ` ± ${num(ap.std, 3)}`,
      sub: `${ap.n} reliably testable replicas${excluded.length ? `; ${excluded.join(", ")} excluded (too few attacks to score)` : ""}`,
      target: "res-loro",
      aria: `LightGBM baseline: mean average precision ${num(ap.mean, 3)} plus or minus ${num(ap.std, 3)} across ${ap.n} held-out replicas. Jump to finding 3.`,
    }),
    heroStat(env, root, {
      label: "Replicas tested",
      iconName: "layers",
      value: sum.n_folds,
      format: (v) => int(v),
      sub: Number.isFinite(sum.n_folds) ? `Each one held out in turn: train on the other ${sum.n_folds - 1}, test on it` : "Each one held out in turn",
      target: "res-loro",
      aria: `${sum.n_folds} replicas tested, each held out once. Jump to finding 3.`,
    }),
    heroStat(env, root, {
      label: "AP gained by a random split",
      iconName: "split",
      value: drop,
      format: signedPts,
      suffix: " points",
      sub: `AP ${num(lk.r0_random_ap, 3)} on a random split vs ${num(lk.r1_chronological_ap, 3)} chronological (${lk.replica || "one replica"}, network metrics only)`,
      target: "res-leakage",
      aria: `A random split gains ${signedPts(drop)} AP points: ${num(lk.r0_random_ap, 3)} versus ${num(lk.r1_chronological_ap, 3)} on a chronological split. Jump to finding 1.`,
    }),
    heroStat(env, root, {
      label: "TESSERA-base vs LightGBM, one fold",
      iconName: "cpu",
      value: tb.tessera_ap,
      format: (v) => num(v, 4),
      suffix: ` vs ${num(tb.lightgbm_ap, 4)}`,
      sub: `AP on one held-out fold (${tb.held_out || "santos"}), one run; ${int(tb.n_parameters)} parameters vs ${treeCount(data) ? `a default-configured ${treeCount(data)}-tree LightGBM` : "a default-configured LightGBM"}`,
      target: "res-neural",
      aria: `On one held-out fold, ${tb.held_out || "santos"}, TESSERA-base scores AP ${num(tb.tessera_ap, 4)} against ${num(tb.lightgbm_ap, 4)} for LightGBM. Jump to finding 4.`,
    })
  );
  return grid;
}

/* ------------------------------------------------------------------ finding 1: leakage */

/** The split sketch: an illustration of the mechanism, not data. */
function leakSketch() {
  const GROUPS = [3, 2, 4, 2, 3, 3, 2, 4, 3, 2, 2]; // runs of near-identical neighbouring windows
  const RANDOM_TEST = new Set([1, 5, 8, 11, 14, 15, 19, 22, 26, 28]);
  const N = GROUPS.reduce((a, b) => a + b, 0);
  const CUT = 19;
  const BUFFER = 2;
  const CELL = 20;
  const W = 16;
  const TOP = 4;
  const CH = 24;
  const roleOf = (mode, i) => (mode === "random" ? (RANDOM_TEST.has(i) ? "test" : "train") : i < CUT ? "train" : i < CUT + BUFFER ? "gap" : "test");

  const title = s("title", {});
  const svg = s("svg", { class: "res-strip-svg", viewBox: `0 0 ${N * CELL} 52`, role: "img" }, title);
  const cells = [];
  for (let i = 0; i < N; i++) {
    const r = s("rect", { class: "res-cell is-train", x: i * CELL + 2, y: TOP, width: W, height: CH, rx: 4 });
    r.style.transitionDelay = `${i * 14}ms`;
    svg.appendChild(r);
    cells.push(r);
  }
  const brackets = [];
  let start = 0;
  for (const len of GROUPS) {
    const x0 = start * CELL + 3;
    const x1 = (start + len - 1) * CELL + 2 + W - 1;
    const y0 = TOP + CH + 5;
    const p = s("path", { class: "res-bracket", d: `M${x0} ${y0}V${y0 + 5}H${x1}V${y0}` });
    const dot = s("circle", { class: "res-leak-dot", cx: (x0 + x1) / 2, cy: y0 + 12, r: 3 });
    svg.append(p, dot);
    brackets.push({ p, dot, from: start, len });
    start += len;
  }

  const caption = h("p", { class: "res-strip-caption", "aria-live": "polite" });
  const btnRandom = h("button", { type: "button", "aria-pressed": "true" }, "Random split");
  const btnChrono = h("button", { type: "button", "aria-pressed": "false" }, "Chronological split");
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Split type shown in the sketch" }, btnRandom, btnChrono);

  function apply(mode) {
    cells.forEach((c, i) => c.setAttribute("class", `res-cell is-${roleOf(mode, i)}`));
    let leaks = 0;
    for (const b of brackets) {
      const roles = new Set();
      for (let k = 0; k < b.len; k++) roles.add(roleOf(mode, b.from + k));
      const leak = roles.has("train") && roles.has("test");
      b.p.classList.toggle("is-leak", leak);
      b.dot.classList.toggle("is-leak", leak);
      if (leak) leaks++;
    }
    btnRandom.setAttribute("aria-pressed", mode === "random" ? "true" : "false");
    btnChrono.setAttribute("aria-pressed", mode === "random" ? "false" : "true");
    const text =
      mode === "random"
        ? `Random split: near-copies of the same quiet minutes land on both sides, so the test set holds rows the model has effectively already seen. In this sketch ${leaks} of ${GROUPS.length} groups straddle the split.`
        : `Chronological split: training is everything before the cut, testing everything after, and a short buffer at the cut is dropped. In this sketch ${leaks} of ${GROUPS.length} groups straddle the split.`;
    caption.textContent = text;
    title.textContent = `Illustration of a ${mode === "random" ? "random" : "chronological"} split over 30 windows in time order. ${text}`;
  }
  btnRandom.addEventListener("click", () => apply("random"));
  btnChrono.addEventListener("click", () => apply("chrono"));
  apply("random");

  return h(
    "div",
    { class: "res-strip" },
    h(
      "div",
      { class: "res-strip-head" },
      h("div", {}, h("div", { class: "res-panel-title" }, "Why a random split leaks"), h("div", { class: "res-panel-sub" }, "Thirty one-minute windows in time order; brackets join runs of near-identical neighbours.")),
      h("div", { class: "row" }, h("span", { class: "pill neutral" }, "Illustration, not data"), seg)
    ),
    svg,
    h("div", { class: "res-strip-axis", "aria-hidden": "true" }, h("span", {}, "Earlier"), h("span", {}, "Time"), h("span", {}, "Later")),
    legend([
      { color: "--c-train", label: "Training" },
      { color: "--c-test", label: "Test" },
      { color: "--c-unused", label: "Buffer, dropped" },
      { color: "--critical", label: "Near-copies on both sides (leak)", shape: "line" },
    ]),
    caption
  );
}

function buildLeakage(env, data) {
  const d = data.leakage_duplicates;
  if (!d) throw new Error("leakage_duplicates is missing from real_results.json");
  const card = findingCard({ n: 1, id: "res-leakage", tour: "results-leakage", title: "Random splits flatter the model" });
  const drop = Math.round((d.r0_random_ap - d.r1_chronological_ap) * 100);
  card.append(
    takeaway(
      `Shuffling windows before splitting made the detector look ${drop} AP points better than a chronological split of the same windows measures: AP ${num(d.r0_random_ap)} on a random split, but ${num(d.r1_chronological_ap)} when every test window comes after every training window.`
    ),
    h("p", { class: "res-context" }, `Replica ${d.replica} · ${d.features} · ${int(d.n_windows)} windows, ${int(d.n_test_rows)} in the test set · LightGBM`)
  );

  const chartHost = h("div", { class: "res-chart" });
  lazyChart(
    env,
    chartHost,
    barChart,
    {
      horizontal: true,
      height: 124,
      yDomain: [0, 1],
      yFormat: apFormat(3),
      valueName: "Average precision",
      bars: [
        { id: "r0", label: "Random split", value: d.r0_random_ap, color: "--series-1", note: "R0: rows shuffled, then split" },
        { id: "r1", label: "Chronological split", value: d.r1_chronological_ap, color: "--series-2", note: "R1: split in time order, 600 s buffer" },
      ],
      ariaLabel: `Average precision: random split ${num(d.r0_random_ap)}, chronological split ${num(d.r1_chronological_ap)}.`,
    },
    124
  );

  const nearest = d.test_rows_identical_to_train;
  const chips = h(
    "div",
    { class: "res-chips", role: "list", "aria-label": "Leakage certificate" },
    chip(env, { value: d.exact_duplicate_rate, format: (v) => pct(v, 1), label: "of rows are exact duplicates of another row" }),
    chip(env, { value: d.near_duplicate_rate, format: (v) => pct(v, 1), label: "are near-duplicates", sub: d.near_duplicate_method ? `method: ${d.near_duplicate_method}` : null }),
    chip(env, { value: nearest, format: (v) => int(v), label: "test rows identical to a training row", sub: Number.isFinite(d.n_test_rows) ? `out of ${int(d.n_test_rows)} test rows` : null })
  );
  [...chips.children].forEach((c) => c.setAttribute("role", "listitem"));

  card.append(
    h(
      "div",
      { class: "res-split" },
      panel("Average precision on the same data, two ways to split it", "Higher is better; 1.0 is perfect ranking", chartHost),
      panel("The leakage certificate", "Measured on the random split's rows", chips)
    ),
    leakSketch(),
    whyBlock(
      "Why this happens",
      "Neighbouring one-minute windows are often identical or nearly so: a quiet minute produces the same all-zero vector as the next. A random split scatters these near-copies across training and test, so the model is graded on rows it has effectively seen. A chronological split tests on the future instead, which is what a deployed detector faces.",
      h("span", {}, "This exact shuffled-split protocol is what makes an evaluation look far stronger than it is — see the ", h("a", { href: "#design" }, "design decisions"), " behind our own protocol choice. A permutation control on our pipeline scores at chance, so the inflation is in the split, not a bug.")
    ),
    provenanceFoot(env, { items: [{ provenance: d.provenance, source: d.source }], command: CMD.leakage })
  );
  return card;
}

/* ------------------------------------------------------------------ finding 2: calendar leak */

function miniSplitChart(env, { title, row, leak }) {
  const host = h("div", { class: "res-chart" });
  lazyChart(
    env,
    host,
    barChart,
    {
      height: 190,
      yDomain: [0, 1],
      yFormat: apFormat(4),
      valueName: "Average precision",
      bars: [
        { id: "r0", label: "Random", value: row.r0_ap, color: "--series-1" },
        { id: "r1", label: "Chronological", value: row.r1_ap, color: "--series-2" },
      ],
      ariaLabel: `${title}: random split AP ${num(row.r0_ap, 4)}, chronological split AP ${num(row.r1_ap, 4)}, gap ${num(row.gap, 3)}.`,
    },
    190
  );
  const gapPill = leak ? pill("bad", "alert", `Gap ${num(row.gap, 3)} · leaking`) : pill("good", "check", `Gap ${num(row.gap, 3)} · closed`);
  return h(
    "div",
    { class: "res-mini" },
    h("div", { class: "res-mini-head" }, h("div", {}, h("div", { class: "res-mini-title" }, title), h("div", { class: "res-mini-sub" }, row.features)), gapPill),
    host
  );
}

function buildCalendar(env, data) {
  const c = data.calendar_leak;
  if (!c || !Array.isArray(c.rows) || c.rows.length < 2) throw new Error("calendar_leak rows are missing");
  const withCal = c.rows.find((r) => /with calendar/i.test(r.features)) || c.rows[0];
  const without = c.rows.find((r) => /without calendar/i.test(r.features)) || c.rows[1];
  const card = findingCard({ n: 2, id: "res-calendar", title: "A leak we found in our own features" });
  const spans = (data.dataset?.replicas || []).map((r) => r.span_hours).filter(Number.isFinite);
  const spanText = spans.length ? `${(Math.min(...spans) / 24).toFixed(1)} to ${(Math.max(...spans) / 24).toFixed(1)} day` : "4 to 6 day";
  card.append(
    takeaway(
      `Our first host-identity features included the hour of day and the day of the week. They let the model memorise when this capture's attacks happened: the gap between a random and a chronological split was ${num(withCal.gap, 3)}. Deleting them shrank it to ${num(without.gap, 3)}.`
    ),
    h(
      "p",
      { class: "res-context res-context-pill" },
      h("span", {}, `Replica ${c.replica} · same seed, same split, features in versus out · LightGBM · recorded before the calendar features were deleted, so these two rows are transcribed`),
      notRerunPill()
    ),
    h("div", { class: "grid-2 res-minis" }, miniSplitChart(env, { title: "With calendar features", row: withCal, leak: true }), miniSplitChart(env, { title: "Calendar features deleted", row: without, leak: false })),
    legend([
      { color: "--series-1", label: "Random split (R0)" },
      { color: "--series-2", label: "Chronological split (R1)" },
    ])
  );

  const all = data.all_modalities_r0_r1;
  if (all) {
    card.append(
      panel(
        "After the fix, with all four feature groups",
        `${all.features} · ${all.replica}`,
        h(
          "div",
          { class: "res-chips res-chips-row" },
          chip(env, { value: all.r0_ap, format: (v) => num(v, 3), label: "AP, random split" }),
          chip(env, { value: all.r1_ap, format: (v) => num(v, 3), label: "AP, chronological split" }),
          chip(env, { value: all.r1_mcc, format: (v) => num(v, 3), label: "MCC, chronological split" })
        )
      )
    );
  }
  const re = c.recheck_without_calendar;
  if (re) {
    card.append(
      h(
        "div",
        { class: "notice res-notice" },
        h("span", { class: "notice-icon", html: icon("reset") }),
        h(
          "div",
          {},
          h("strong", {}, "Re-checked at export. "),
          `The "calendar features deleted" row re-run on today's cache: random ${num(re.r0_ap, 4)}, chronological ${num(re.r1_ap, 4)}, gap ${num(re.gap, 3)}. `,
          re.note || ""
        )
      )
    );
  }
  card.append(
    whyBlock(
      "Why this happens",
      `In one ${spanText} capture, "hour 3 on 24 January" happens exactly once. A random split puts the exact calendar position of every held-out attack window into training, so the model learns "attacks happen around hour X in this capture", which transfers to nothing.${c.top_feature ? ` The most important feature was ${c.top_feature}.` : ""}`,
      "Fixed at the source: the calendar features were deleted from m3_identity.py, and a test fails if a calendar-shaped feature ever comes back."
    ),
    provenanceFoot(env, {
      items: [
        { provenance: c.provenance, source: c.source, label: "Calendar ablation" },
        ...(re ? [{ provenance: re.provenance, source: re.source, label: "Re-check" }] : []),
        ...(all ? [{ provenance: all.provenance, source: all.source, label: "All four feature groups" }] : []),
      ],
      command: CMD.leakage,
    })
  );
  return card;
}

/* ------------------------------------------------------------------ finding 3: leave-one-replica-out */

function buildLoro(env, data) {
  const loro = data.loro;
  if (!loro || !Array.isArray(loro.folds)) throw new Error("loro.folds is missing");
  const sum = loro.summary || {};
  const minSupport = sum.min_support ?? 20;
  const order = (data.dataset?.replicas || []).map((r) => r.id);
  const folds = [...loro.folds].sort((a, b) => {
    const ia = order.indexOf(a.held_out);
    const ib = order.indexOf(b.held_out);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  const ap = sum.average_precision || {};
  const card = findingCard({
    n: 3,
    id: "res-loro",
    tour: "results-loro",
    title: Number.isFinite(ap.n) ? `It generalises across the ${ap.n} replicas with enough attacks to measure` : "It generalises across the replicas with enough attacks to measure",
  });
  const floor = data.mask_only_floor;
  const floorAp = floor?.summary?.average_precision || null;
  const floorByRep = Object.fromEntries((floor?.folds || []).map((f) => [f.held_out, f.average_precision]));
  const mcc = sum.mcc || {};
  const naive = sum.naive_all_folds_ap || {};
  card.append(
    takeaway(
      `Trained on ${folds.length - 1} replicas and tested on the one left out, repeated for every replica, the detector averages AP ${num(ap.mean)} ± ${num(ap.std)} over the ${ap.n} replicas with enough attacks to measure.`
    ),
    h("p", { class: "res-context" }, `Leave-one-replica-out · ${loro.model || "LightGBM"} (default settings) · all four feature groups · MCC at threshold 0.5`)
  );

  const METRICS = {
    ap: { key: "average_precision", name: "Average precision", title: "Average precision per held-out replica", mean: ap.mean },
    mcc: { key: "mcc", name: "MCC at threshold 0.5", title: "MCC per held-out replica (threshold 0.5)", mean: mcc.mean },
  };
  const excludedNote = (f) => `Only ${int(f.n_pos_test)} attack windows in ${int(f.n_test)} (fewer than ${minSupport}): reported, excluded from the mean`;
  const chartOpts = (metric) => {
    const M = METRICS[metric];
    const vals = folds.map((f) => f[M.key]).filter(Number.isFinite);
    return {
      horizontal: true,
      height: folds.length * 30 + 36,
      yDomain: [Math.min(0, ...vals), 1],
      yFormat: apFormat(3),
      valueLabels: "flagged",
      valueName: M.name,
      flaggedLabel: `Low support (fewer than ${minSupport} attack windows): shown, excluded from the mean`,
      refLine: Number.isFinite(M.mean) ? { value: M.mean, label: `Mean of the ${ap.n ?? folds.length} included folds (${num(M.mean)})` } : null,
      bars: folds.map((f) => ({
        id: f.held_out,
        label: f.held_out,
        value: f[M.key],
        color: "--series-1",
        flagged: !!f.low_support,
        note: f.low_support ? excludedNote(f) : `${int(f.n_pos_test)} attack windows in ${int(f.n_test)} test windows`,
      })),
      ariaLabel: `${M.title}. ${folds.map((f) => `${f.held_out} ${num(f[M.key])}${f.low_support ? " (low support, excluded)" : ""}`).join(", ")}.`,
    };
  };

  const toggleHost = h("div", { class: "res-loro-chart" });
  const columns = [
    { key: "held_out", label: "Held-out replica" },
    { key: "n_test", label: "Test windows", num: true, format: (v) => int(v) },
    { key: "n_pos_test", label: "Attack windows", num: true, format: (v) => int(v) },
    { key: "test_prevalence", label: "Prevalence", num: true, format: (v) => pct(v, v < 0.01 ? 2 : 1) },
    { key: "average_precision", label: "AP", num: true, format: (v) => num(v, 4) },
    { key: "mcc", label: "MCC", num: true, format: (v) => num(v, 3) },
    ...(floor ? [{ key: "floor_ap", label: "Mask-only AP (floor)", num: true, format: (v) => (Number.isFinite(v) ? num(v, 4) : "—") }] : []),
    { key: "status", label: "In the mean?" },
  ];
  const rows = folds.map((f) => ({ ...f, floor_ap: floorByRep[f.held_out], status: f.low_support ? `No: low support (< ${minSupport} attacks)` : "Yes", _flagged: !!f.low_support }));
  const tt = withTableToggle(toggleHost, {
    title: METRICS.ap.title,
    render: (host) => lazyChart(env, host, barChart, chartOpts("ap"), folds.length * 30 + 70),
    table: { columns, rows },
  });

  const btnAp = h("button", { type: "button", "aria-pressed": "true" }, "Average precision");
  const btnMcc = h("button", { type: "button", "aria-pressed": "false" }, "MCC");
  const setMetric = (m) => {
    btnAp.setAttribute("aria-pressed", m === "ap" ? "true" : "false");
    btnMcc.setAttribute("aria-pressed", m === "mcc" ? "true" : "false");
    tt.chart.update(chartOpts(m));
    const t = toggleHost.querySelector(".viz-toggle-title");
    if (t) t.textContent = METRICS[m].title;
  };
  btnAp.addEventListener("click", () => setMetric("ap"));
  btnMcc.addEventListener("click", () => setMetric("mcc"));

  const tiles = h(
    "div",
    { class: `${floorAp ? "grid-4" : "grid-3"} res-tiles` },
    summaryTile(env, { primary: true, label: `Mean AP · ${ap.n} included folds`, mean: ap.mean, std: ap.std, sub: `min ${num(ap.min)}, max ${num(ap.max)}` }),
    floorAp
      ? summaryTile(env, {
          label: `Mask-only floor · same ${floorAp.n} folds`,
          mean: floorAp.mean,
          std: floorAp.std,
          sub: "The floor the full model must beat: LightGBM on which feature groups are present, no values",
          cls: "res-floor-tile",
        })
      : null,
    summaryTile(env, { label: `Naive mean AP · all ${naive.n ?? folds.length} folds`, mean: naive.mean, std: naive.std, sub: "What averaging the low-support fold in would read" }),
    summaryTile(env, { label: `Mean MCC · ${mcc.n} included folds`, mean: mcc.mean, std: mcc.std, sub: `min ${num(mcc.min)}, max ${num(mcc.max)} at threshold 0.5` })
  );

  const notes = h("ul", { class: "res-notes" });
  for (const f of folds.filter((x) => x.low_support)) {
    notes.append(
      h(
        "li",
        {},
        ico("alert", "res-note-ico warn"),
        h(
          "div",
          {},
          h("strong", {}, `${f.held_out} is shown but not averaged in. `),
          `It has only ${int(f.n_pos_test)} attack windows out of ${int(f.n_test)} (prevalence ${pct(f.test_prevalence, 2)}), all in one short episode of its capture. That is below the ${minSupport}-positive minimum for a stable rate, so its AP of ${num(f.average_precision)} is reported in full and kept out of the summary.`
        )
      )
    );
  }
  const gapFold = folds
    .filter((f) => !f.low_support && Number.isFinite(f.average_precision) && Number.isFinite(f.mcc))
    .sort((a, b) => b.average_precision - b.mcc - (a.average_precision - a.mcc))[0];
  if (gapFold && gapFold.average_precision - gapFold.mcc > 0.3) {
    notes.append(
      h(
        "li",
        {},
        ico("info", "res-note-ico info"),
        h(
          "div",
          {},
          h("strong", {}, `${gapFold.held_out}: high AP, low MCC. `),
          `It ranks attacks well (AP ${num(gapFold.average_precision)}) yet scores MCC ${num(gapFold.mcc)} at the fixed 0.5 threshold, which usually means its scores are shifted for that replica rather than the model failing to learn. Switch the chart to MCC to see it. AP is threshold-free, which is why it is the headline.`
        )
      )
    );
  }
  if (floorAp) {
    const sh = floor.attack_share || {};
    const hostAp = floor.host_only?.summary?.average_precision;
    const both = floor.mask_and_host?.summary?.average_precision;
    const elsewhere = Number.isFinite(sh.n_attack_windows) && Number.isFinite(sh.n_attack_on_host) ? sh.n_attack_windows - sh.n_attack_on_host : null;
    notes.append(
      h(
        "li",
        { class: "res-floor-note" },
        ico("alert", "res-note-ico warn"),
        h(
          "div",
          {},
          h("strong", {}, "The floor the full model must beat. "),
          `A LightGBM that sees only which feature groups are present in each window, and no feature values, already scores mean AP ${num(floorAp.mean)} ± ${num(floorAp.std)} on these folds${Number.isFinite(hostAp?.mean) ? `; host identity alone scores ${num(hostAp.mean)}` : ""}${Number.isFinite(both?.mean) ? `, and the two together ${num(both.mean)}` : ""}. That is by construction: labels exist only for the log files M1 reads, so ${Number.isFinite(sh.m1_present_rate_attack) ? pct(sh.m1_present_rate_attack, 1) : "every"} of attack windows have M1 activity against ${Number.isFinite(sh.m1_present_rate_benign) ? pct(sh.m1_present_rate_benign, 1) : "a minority"} of benign ones, and ${Number.isFinite(sh.share_of_attacks_on_host) ? pct(sh.share_of_attacks_on_host, 1) : "nearly all"} of attack windows are on ${sh.host || "the firewall host"}. The full model's ${num(ap.mean)} clears the floor by ${Number.isFinite(ap.mean) ? Math.round((ap.mean - floorAp.mean) * 100) : "—"} AP points, but cross-replica AP is dominated by detecting the attack on the firewall host${elsewhere != null ? `; the other two hosts contribute only ${int(elsewhere)} attack windows in total` : ""}.`
        )
      )
    );
  }
  notes.append(
    h(
      "li",
      {},
      ico("compass", "res-note-ico info"),
      h(
        "div",
        {},
        h("strong", {}, "Cross-replica, not cross-organisation. "),
        "The 8 replicas are randomised runs of the same scenario and attack repertoire, so this measures robustness to that randomisation, not transfer to a different network or attack type."
      )
    )
  );

  card.append(
    h("div", { class: "res-loro-controls" }, h("span", { class: "label", id: "res-loro-metric" }, "Metric"), h("div", { class: "seg", role: "group", "aria-labelledby": "res-loro-metric" }, btnAp, btnMcc)),
    toggleHost,
    tiles,
    notes,
    floor ? cmdLine(env, CMD.floor, "Floor") : null,
    provenanceFoot(env, {
      items: [
        { provenance: loro.provenance, source: loro.source, label: "Leave-one-replica-out" },
        ...(floor ? [{ provenance: floor.provenance, source: floor.source, label: "Mask-only floor" }] : []),
      ],
      command: CMD.loro,
    })
  );
  return card;
}

function summaryTile(env, { label, mean, std, sub, primary = false, cls = "" }) {
  const v = h("span", {});
  countIn(env, v, mean, (x) => num(x, 3), 1000);
  return h(
    "div",
    { class: `stat${primary ? " primary" : ""}${cls ? ` ${cls}` : ""}` },
    h("span", { class: "stat-label" }, label),
    h("span", { class: "stat-value tabular" }, v, Number.isFinite(std) ? h("span", { class: "res-pm" }, ` ± ${num(std, 3)}`) : null),
    h("span", { class: "stat-sub" }, sub)
  );
}

/* ------------------------------------------------------------------ finding 4: neural model + attribution */

function ablationStyle(check) {
  const c = String(check).toLowerCase();
  if (c.startsWith("gbdt on m1")) return { label: "GBDT · M1 only", color: "--series-1" };
  if (c.includes("host_bucket")) return { label: "GBDT · host_bucket only", color: "--series-3" };
  if (c.startsWith("gbdt on m3")) return { label: "GBDT · M3 only", color: "--series-3" };
  if (c.includes("without m1")) return { label: "TESSERA-base · no M1", color: "--series-5" };
  if (c.startsWith("tessera-base")) return { label: "TESSERA-base · all four", color: "--series-5" };
  return { label: String(check), color: "--series-5" };
}

function buildNeural(env, data) {
  const tb = data.tessera_base;
  const av = data.attribution_vs_ablation;
  if (!tb || !av) throw new Error("tessera_base or attribution_vs_ablation is missing");
  const card = findingCard({ n: 4, id: "res-neural", title: "The neural model, and why we don't trust its explanations" });
  const abl = av.ablation || [];
  const withM1 = abl.find((r) => /tessera-base with m1/i.test(r.check));
  const noM1 = abl.find((r) => /without m1/i.test(r.check));
  const m1Alone = abl.find((r) => /gbdt on m1/i.test(r.check));
  const gateM1 = av.mean_attribution?.m1_log;
  card.append(
    takeaway(
      `Our ${int(tb.n_parameters)}-parameter neural model matches a default-configured ${treeCount(data) ? `${treeCount(data)}-tree ` : ""}LightGBM on one held-out fold (${tb.held_out}, single run): AP ${num(tb.tessera_ap, 4)} vs ${num(tb.lightgbm_ap, 4)}. But its built-in explanation says the log-template group barely matters, and an ablation shows it matters a lot.`
    ),
    h("p", { class: "res-context" }, `Held-out replica ${tb.held_out} · ${int(tb.n_train_windows)} training windows from the other 7 replicas · ${tb.epochs} epochs with early stopping · ${tb.train_seconds} s on ${tb.device}`)
  );

  const stat = (primary, label, apV, mccV, sub) => {
    const v = h("span", {});
    countIn(env, v, apV, (x) => num(x, 4), 1100);
    return h(
      "div",
      { class: `stat${primary ? " primary" : ""}` },
      h("span", { class: "stat-label" }, label),
      h("span", { class: "stat-value tabular" }, h("span", { class: "res-unit" }, "AP "), v),
      h("span", { class: "stat-sub" }, `MCC ${num(mccV, 4)} · ${sub}`)
    );
  };
  card.append(
    h(
      "div",
      { class: "grid-2 res-pair" },
      stat(true, `TESSERA-base · neural, ${int(tb.n_parameters)} parameters`, tb.tessera_ap, tb.tessera_mcc, "four small encoders, a gated fusion unit and one head"),
      stat(false, `LightGBM · ${treeCount(data) ? `${treeCount(data)} boosted trees` : "gradient-boosted trees"}, default settings`, tb.lightgbm_ap, tb.lightgbm_mcc, "the same held-out fold, from finding 3")
    ),
    h(
      "p",
      { class: "res-context res-context-pill" },
      pill("warn", "alert", "One fold, one seed"),
      h("span", {}, `The neural-vs-LightGBM comparison is a single held-out fold from a single training run. An 8-fold paired comparison is future work (NEGATIVE_RESULTS C2), so read the ${num(Math.abs(tb.tessera_ap - tb.lightgbm_ap), 4)} AP difference as a tie, not a win.`)
    ),
    h("h3", { class: "res-subhead" }, "Can we trust the model's own explanation?")
  );

  const attrHost = h("div", { class: "res-chart" });
  lazyChart(
    env,
    attrHost,
    barChart,
    {
      horizontal: true,
      height: 4 * 32 + 36,
      yDomain: [0, 1],
      yFormat: (v) => `${Math.round(v * 100)}%`,
      valueName: "Mean gate weight",
      bars: MODALITIES.map((m) => ({ id: m.id, label: `${m.label} (${m.code})`, value: av.mean_attribution?.[m.id], color: m.color })),
      ariaLabel: `Mean gate weight per feature group: ${MODALITIES.map((m) => `${m.label} ${Math.round((av.mean_attribution?.[m.id] ?? 0) * 100)}%`).join(", ")}.`,
    },
    4 * 32 + 36
  );

  const santosPrev = data.loro?.folds?.find((f) => f.held_out === tb.held_out)?.test_prevalence;
  const ablBars = abl.map((r, i) => {
    const st = ablationStyle(r.check);
    return { id: `abl${i}`, label: st.label, value: r.ap, color: st.color, note: [r.check, r.note].filter(Boolean).join(" · ") };
  });
  const ablTable = {
    columns: [
      { key: "check", label: "Check" },
      { key: "ap", label: "AP", num: true, format: (v) => num(v, 4) },
      { key: "note", label: "Note", format: (v) => v || "—" },
    ],
    rows: abl,
  };
  const ablToggle = h("div", {});
  withTableToggle(ablToggle, {
    render: (host) =>
      lazyChart(
        env,
        host,
        barChart,
        {
          horizontal: true,
          height: ablBars.length * 32 + 36,
          yDomain: [0, 1],
          yFormat: apFormat(4),
          valueName: "Average precision",
          refLine: Number.isFinite(santosPrev) ? { value: santosPrev, label: `No-skill AP (prevalence ${num(santosPrev, 3)})` } : null,
          bars: ablBars,
          ariaLabel: `Ablation average precision: ${abl.map((r) => `${r.check} ${num(r.ap, 4)}`).join("; ")}.`,
        },
        ablBars.length * 32 + 64
      ),
    table: ablTable,
  });

  card.append(
    h(
      "div",
      { class: "grid-2 res-attr" },
      panel("What the gates say", "Mean gate weight per feature group on the test set (the GMU's own explanation)", attrHost),
      panel(
        "What removing a source actually does",
        "AP of each ablation on the same held-out fold",
        legend([
          { color: "--series-1", label: "Log templates (M1) only" },
          { color: "--series-3", label: "Host identity (M3) only" },
          { color: "--series-5", label: "TESSERA-base" },
        ]),
        ablToggle,
        gbdtRowsNote(abl, data)
      )
    )
  );

  if (Number.isFinite(gateM1) && withM1 && noM1) {
    const gateEl = h("span", { class: "res-big" });
    countIn(env, gateEl, gateM1, (v) => `${Math.round(v * 100)}%`, 900);
    card.append(
      h(
        "div",
        { class: "res-contrast", role: "group", "aria-label": "Gate weight compared with ablation" },
        h("div", { class: "res-contrast-item" }, h("span", { class: "res-contrast-k" }, "The gate says log templates (M1) get"), gateEl, h("span", { class: "res-contrast-s" }, "of the weight, on average")),
        h("span", { class: "res-contrast-vs", "aria-hidden": "true" }, "but"),
        h(
          "div",
          { class: "res-contrast-item" },
          h("span", { class: "res-contrast-k" }, "Genuinely removing M1 drops AP"),
          h("span", { class: "res-big tabular" }, `${num(withM1.ap, 4)} → ${num(noM1.ap, 4)}`),
          h("span", { class: "res-contrast-s" }, m1Alone ? `and M1 on its own scores ${num(m1Alone.ap, 4)}` : "zeroed and marked unavailable")
        )
      )
    );
  }

  card.append(
    whyBlock(
      "What we conclude",
      av.finding || "Gate values are a signal, never ground truth for what the model used.",
      h("span", {}, "So the ", h("a", { href: "#detector" }, "live detector"), " shows gate weights as a hint with this caveat beside them, and per-source ablation is the importance measure this project cites.")
    ),
    provenanceFoot(env, {
      items: [
        { provenance: tb.provenance, source: tb.source, label: "TESSERA-base vs LightGBM" },
        { provenance: av.provenance, source: av.source, label: "Attribution and ablation" },
      ],
      command: CMD.neural,
    })
  );
  return card;
}

/** Marks the three GBDT single-group rows as a transcribed one-off, and says which one
 *  the exporter's host-only floor now re-checks independently. */
function gbdtRowsNote(abl, data) {
  const gbdt = abl.filter((r) => /^gbdt/i.test(String(r.check)));
  if (!gbdt.length) return null;
  const host = gbdt.find((r) => /host_bucket/i.test(r.check));
  const floorSantos = (data.mask_only_floor?.host_only?.folds || []).find((f) => f.held_out === data.tessera_base?.held_out)?.average_precision;
  const agrees = host && Number.isFinite(floorSantos) && num(floorSantos, 4) === num(host.ap, 4);
  return h(
    "p",
    { class: "res-context res-context-pill" },
    notRerunPill(),
    h(
      "span",
      {},
      `The ${gbdt.length} GBDT single-group rows are a one-off ablation on this fold, transcribed from RESULTS.md.`,
      agrees ? ` The host_bucket-only row (${num(host.ap, 4)}) is independently reproduced by the exporter's host-only floor on the same fold.` : ""
    )
  );
}

/* ------------------------------------------------------------------ dataset + reproduce */

function buildDataset(env, data) {
  const ds = data.dataset;
  if (!ds || !Array.isArray(ds.replicas)) throw new Error("dataset.replicas is missing");
  const minSupport = data.loro?.summary?.min_support ?? 20;
  const card = findingCard({ n: 0, id: "res-dataset", title: "The data behind every number" });
  const maxPrev = Math.max(...ds.replicas.map((r) => r.prevalence).filter(Number.isFinite), 1e-9);
  const tbody = h("tbody", {});
  ds.replicas.forEach((r, i) => {
    const low = r.n_positive < minSupport;
    tbody.append(
      h(
        "tr",
        { class: low ? "is-flagged" : "" },
        h("td", {}, h("span", { class: "res-rep" }, r.id, low ? h("span", { class: "pill warn", html: `${icon("alert")}<span>Low support</span>` }) : null)),
        h("td", { class: "num" }, int(r.n_windows)),
        h("td", { class: "num" }, int(r.n_positive)),
        h(
          "td",
          { class: "num" },
          h(
            "span",
            { class: "res-prev" },
            h("span", {}, pct(r.prevalence, r.prevalence < 0.01 ? 2 : 1)),
            h("span", { class: "res-prev-bar", "aria-hidden": "true" }, h("span", { style: { width: `${Math.max(2, (r.prevalence / maxPrev) * 100)}%`, animationDelay: `${i * 60}ms` } }))
          )
        ),
        h("td", { class: "num" }, num(r.span_hours, 1))
      )
    );
  });
  const t = ds.totals || {};
  const tfoot = h(
    "tfoot",
    {},
    h(
      "tr",
      { class: "res-total" },
      h("td", {}, "All replicas"),
      h("td", { class: "num" }, int(t.n_windows)),
      h("td", { class: "num" }, int(t.n_positive)),
      h("td", { class: "num" }, Number.isFinite(t.n_windows) && t.n_windows > 0 ? pct(t.n_positive / t.n_windows, 1) : "—"),
      h("td", { class: "num" }, "—")
    )
  );
  const table = h(
    "table",
    { class: "table" },
    h("caption", { class: "sr-only" }, "Per-replica window counts, attack windows, prevalence and capture span"),
    h(
      "thead",
      {},
      h("tr", {}, h("th", { scope: "col" }, "Replica"), h("th", { class: "num", scope: "col" }, "Windows"), h("th", { class: "num", scope: "col" }, "Attack windows"), h("th", { class: "num", scope: "col" }, "Prevalence"), h("th", { class: "num", scope: "col" }, "Span (hours)"))
    ),
    tbody,
    tfoot
  );
  card.append(
    h("p", { class: "res-takeaway res-takeaway-soft" }, `${ds.name}${ds.authors ? ` (${ds.authors})` : ""}: ${ds.subset || ""}`),
    h("div", { class: "table-wrap" }, table),
    h(
      "div",
      { class: "notice res-notice" },
      h("span", { class: "notice-icon", html: icon("lock") }),
      h(
        "div",
        {},
        h("strong", {}, "Counts only. "),
        `The dataset is licensed ${ds.licence || "CC BY-NC-SA 4.0"} (Zenodo record `,
        h("a", { href: `https://zenodo.org/records/${ds.zenodo_record}`, target: "_blank", rel: "noopener noreferrer" }, String(ds.zenodo_record)),
        "). By the project's data policy (LICENSE-DATA in the repository), this site never ships a row of it, only aggregate counts and summary statistics; the Training Lab runs on synthetic data calibrated from those statistics."
      )
    ),
    provenanceFoot(env, { items: [{ provenance: ds.provenance, source: ds.source }], command: null })
  );
  return card;
}

function buildReproduce(env, data) {
  const card = h("section", { class: "card res-card reveal-on-scroll", id: "res-reproduce", tabindex: "-1", "aria-labelledby": "res-reproduce-title" });
  const steps = [
    { label: "Findings 1 and 2 · leakage (fast)", cmd: CMD.leakage },
    { label: "Finding 3 · all 8 replicas (fast, uses the cache)", cmd: CMD.loro },
    { label: "Finding 3 · the mask-only floor (uses the cache)", cmd: CMD.floor },
    { label: "Finding 4 · neural model and attribution check (slow, about 5 minutes)", cmd: CMD.neural },
    { label: "Re-export every number on this page", cmd: data.generated_by || CMD.export },
    { label: "Check the in-browser model and ledger against Python", cmd: CMD.parity },
  ];
  const list = h("ol", { class: "res-steps" }, ...steps.map((st, i) => h("li", {}, h("span", { class: "res-step-n", "aria-hidden": "true" }, String(i + 1)), h("div", { class: "res-step-body" }, h("div", { class: "res-step-label" }, st.label), cmdLine(env, st.cmd, null)))));
  const when = data.generated_utc ? new Date(data.generated_utc) : null;
  const whenText = when && !Number.isNaN(when.getTime()) ? `${when.toISOString().slice(0, 16).replace("T", " ")} UTC` : null;
  card.append(
    h(
      "div",
      { class: "res-card-head" },
      h("div", { class: "res-kicker" }, h("span", { class: "res-num" }, "Reproduce"), h("span", { class: "tag-real" }, "Real data")),
      h("h2", { class: "res-card-title", id: "res-reproduce-title" }, "Re-run the numbers yourself")
    ),
    h("p", { class: "res-context" }, "Run from the repository root. The tests need the AIT data downloaded and processed locally, which this project deliberately does not redistribute (LICENSE-DATA). Transcribed one-off ablations have no command; they are marked where they appear."),
    list,
    h("p", { class: "res-context res-context-end" }, whenText ? `Page data exported ${whenText} by ${data.generated_by || CMD.export}.` : `Page data exported by ${data.generated_by || CMD.export}.`),
    data.provenance_note ? h("details", { class: "res-src" }, h("summary", {}, "How this page's numbers were exported"), h("p", {}, data.provenance_note)) : null
  );
  return card;
}

/* ------------------------------------------------------------------ mount */

function skeleton() {
  return h(
    "div",
    { class: "res res-loading", "aria-busy": "true", "aria-label": "Loading real results" },
    h("div", { class: "skeleton", style: { height: "22px", width: "140px" } }),
    h("div", { class: "skeleton", style: { height: "40px", width: "min(560px, 100%)" } }),
    h("div", { class: "grid-4" }, ...[0, 1, 2, 3].map(() => h("div", { class: "skeleton", style: { height: "118px" } }))),
    h("div", { class: "skeleton", style: { height: "320px" } })
  );
}

export async function mount(el, ctx = {}) {
  const store = ctx.store || (await import("../store.js")).store;
  const env = makeEnv(ctx.toast || domToast);
  el.replaceChildren(skeleton());
  let data;
  try {
    data = await store.loadJSON("data/real_results.json");
    if (!data || typeof data !== "object") throw new Error("the file is empty");
  } catch (err) {
    el.replaceChildren(
      h("div", { class: "mount-error", role: "alert" }, `The real results could not be loaded (${err.message}). They live in data/real_results.json, written by ${CMD.export}.`)
    );
    return null;
  }

  const root = h("div", { class: "res" });
  const top = h("div", { class: "res-top" }, pageHead(data), section("reading guide", () => readingGuide(data)), section("headline numbers", () => hero(env, root, data)));
  root.append(
    top,
    section("finding 1", () => buildLeakage(env, data)),
    section("finding 2", () => buildCalendar(env, data)),
    section("finding 3", () => buildLoro(env, data)),
    section("finding 4", () => buildNeural(env, data)),
    h("div", { class: "grid-2 res-closing" }, section("dataset", () => buildDataset(env, data)), section("reproduce", () => buildReproduce(env, data)))
  );
  stagger(top);
  el.replaceChildren(root);
  revealOnScroll(root);
  env.start();

  return {
    onShow() {},
    onHide() {
      hideTooltip();
    },
  };
}
