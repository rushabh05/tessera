// Design Decisions tab: TESSERA's own internal record of why it is built the
// way it is.
//
// Every number here is read from data/design_notes.json, which the Python
// exporter (uv run python -m tessera.demo.export_web_data) recomputes with the
// same calls tessera.chainsim.report makes (calibrate, benchmark.run,
// eho.demonstrate_structural_unreachability, segment_objective.demonstrate_objective_invariance).
// design_decisions paraphrase RESULTS.md / NEGATIVE_RESULTS.md / THREAT_MODEL.md
// and name the file each decision comes from. This module draws them; it
// computes nothing new beyond simple comparisons of the exported values.
//
// NOTE ON FIELD NAMES: this file was written before data/design_notes.json
// existed. It expects `optimiser_comparison`, `eho_unreachability`, an
// objective-invariance block (tried under `fh_invariance` first, with a couple
// of fallback names), and `design_decisions: [{decision, why, evidence,
// source_file}]`. If the exporter lands with different field names, update the
// small set of lookups in `mount()` and `buildDecisions()` below.
//
// Motion: head and hero stagger in, cards reveal on scroll, charts and count-ups
// start when they become visible. All of it is skipped under prefers-reduced-motion.

import { h, fmt, countUp, stagger, revealOnScroll, reducedMotion, copyText, escapeHtml, toast as domToast } from "../ui/dom.js";
import { icon } from "../ui/icons.js";
import { barChart, lineChart, rangeChart, withTableToggle, hideTooltip } from "../ui/charts.js";

const CMD = {
  report: "uv run python -m tessera.chainsim.report",
  export: "uv run python -m tessera.demo.export_web_data",
};

const COPY_ICON =
  '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 012-2h9"/></svg>';

/* ------------------------------------------------------------------ formatting */

const num = (v, d = 3) => fmt.num(v, d);
const int = (v) => fmt.int(v);
const sInt = (v) => (Number.isFinite(v) ? v.toLocaleString("en-US", { maximumFractionDigits: 1 }) : "—");
/** Optimality-gap percentages: 0%, 0.005%, 27.6%. */
const gapPct = (v) => {
  if (!Number.isFinite(v)) return "—";
  if (v === 0) return "0%";
  if (Math.abs(v) < 0.1) return `${v.toFixed(3)}%`;
  return `${Number(v.toFixed(1))}%`;
};
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const superscript = (n) => String(n).split("").map((c) => SUP[c] ?? c).join("");

/** Returns the first present, non-null value among `data[key]` for each key in `names`. */
function pick(data, names) {
  for (const name of names) {
    if (data && data[name] != null) return data[name];
  }
  return undefined;
}

/* ------------------------------------------------------------------ lifecycle helpers */

/** Deferred "when visible" hooks; start() runs them once the page is attached. */
function makeEnv(ctx) {
  const queue = [];
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
  }
  return {
    notify: ctx.toast || domToast,
    navigate: ctx.navigate || ((id) => (location.hash = `#${id}`)),
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

function countIn(env, el, to, format, duration = 1000) {
  el.textContent = format(to);
  if (!Number.isFinite(to) || to === 0) return el;
  env.onVisible(el, () => countUp(el, to, { from: 0, duration, format }));
  return el;
}

function section(name, build) {
  try {
    return build();
  } catch (err) {
    console.error(`design tab: the "${name}" section failed`, err);
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

const ico = (name, cls = "") => h("span", { class: `dsg-ico ${cls}`.trim(), html: icon(name) });

function pill(kind, iconName, text) {
  return h("span", { class: `pill ${kind}`, html: `${icon(iconName)}<span>${escapeHtml(text)}</span>` });
}

function legend(items) {
  return h(
    "div",
    { class: "viz-legend dsg-legend" },
    ...items.map((it) => h("span", { class: "item" }, h("span", { class: `viz-key ${it.shape || "rect"}`, style: { background: `var(${it.color})` } }), h("span", {}, it.label)))
  );
}

function panel(title, sub, ...children) {
  return h("div", { class: "dsg-panel" }, h("div", { class: "dsg-panel-title" }, title), sub ? h("div", { class: "dsg-panel-sub" }, sub) : null, ...children);
}

function cmdLine(env, cmd, label = "Reproduce") {
  const btn = h("button", { class: "btn sm ghost dsg-copy", type: "button", "aria-label": `Copy the command: ${cmd}`, html: `${COPY_ICON}<span>Copy</span>` });
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
  return h("div", { class: "dsg-cmd" }, label ? h("span", { class: "dsg-cmd-label" }, label) : null, h("code", { class: "dsg-cmd-code", tabindex: "0" }, cmd), btn);
}

/** items: [{provenance: 'recomputed'|'transcribed', source, label?}] */
function provenanceFoot(env, { items, command }) {
  const kinds = [...new Set(items.map((it) => (it.provenance === "transcribed" ? "transcribed" : "recomputed")))];
  return h(
    "footer",
    { class: "card-foot dsg-foot" },
    h(
      "div",
      { class: "dsg-foot-row" },
      h("span", { class: "dsg-foot-label" }, "Provenance"),
      ...kinds.map((k) => (k === "transcribed" ? pill("neutral", "file", "Transcribed from NEGATIVE_RESULTS.md") : pill("info", "cpu", "Recomputed by our code at export")))
    ),
    command ? cmdLine(env, command) : null,
    h(
      "details",
      { class: "dsg-src" },
      h("summary", {}, items.length > 1 ? `Where these numbers come from (${items.length} sources)` : "Where these numbers come from"),
      ...items.map((it) => h("p", {}, it.label ? h("strong", {}, `${it.label}: `) : null, it.source || "No source recorded."))
    )
  );
}

function designCard({ id, kicker, title, tour }) {
  return h(
    "section",
    { class: "card dsg-card reveal-on-scroll", id, tabindex: "-1", "aria-labelledby": `${id}-title`, "data-tour": tour || null },
    h("div", { class: "dsg-card-head" }, h("div", { class: "dsg-kicker" }, h("span", { class: "dsg-num" }, kicker), h("span", { class: "tag-real" }, "Recomputed")), h("h2", { class: "dsg-card-title", id: `${id}-title` }, title))
  );
}

/** Where the evidence for a decision is shown: another tab, or a card further down this page. */
function decisionTarget(c) {
  const decision = String(c.decision || "").toLowerCase();
  const all = `${decision} ${String(c.why || "").toLowerCase()} ${String(c.evidence || "").toLowerCase()}`;
  if (/accuracy/.test(all)) return { tab: "results", label: "See it measured in Real Results" };
  if (/ledger|privacy/.test(all)) return { tab: "detector", label: "See the ledger in the Live Detector" };
  if (/segment length|elephant|eho|optimiser/.test(all)) return { card: "dsg-eho", label: "See the optimiser check" };
  if (/nsc|objective/.test(all)) return { card: "dsg-objective", label: "See the objective check" };
  return null;
}

/* ------------------------------------------------------------------ head + hero */

function pageHead() {
  return h(
    "header",
    { class: "page-head dsg-head" },
    h(
      "div",
      {},
      h("div", { class: "eyebrow" }, "Design decisions"),
      h("h1", {}, "Why TESSERA is built this way"),
      h(
        "p",
        { class: "lede" },
        "Every non-obvious design choice in TESSERA is backed by a real measurement, not intuition. This page shows the decision, the evidence, and where to find it in the codebase. The optimiser and objective checks below are re-run by the code in this repository at every export."
      )
    )
  );
}

function heroStat(env, root, { label, iconName, value, format, suffix, sub, target, primary = false, aria }) {
  const val = h("span", {});
  countIn(env, val, value, format, 1100);
  return h(
    "button",
    { type: "button", class: `stat dsg-stat${primary ? " primary" : ""}`, "aria-label": aria, onClick: () => scrollToCard(root, target) },
    h("span", { class: "stat-label" }, ico(iconName, "dsg-stat-ico"), h("span", {}, label)),
    h("span", { class: "stat-value tabular" }, val, suffix ? h("span", { class: "dsg-suffix" }, suffix) : null),
    h("span", { class: "stat-sub" }, sub),
    h("span", { class: "dsg-stat-go" }, h("span", {}, "See the evidence"), ico("arrowRight"))
  );
}

function hero(env, root, data) {
  const decisions = data.design_decisions || [];
  const oc = data.optimiser_comparison || {};
  const eho = (oc.results || []).find((r) => /eho/i.test(r.optimiser));
  const rnd = (oc.results || []).find((r) => /random/i.test(r.optimiser));
  const fh = pick(data, ["fh_invariance", "objective_invariance", "fh_invariance_check"]) || {};
  const nsc = fh.nsc_values || [];
  return h(
    "div",
    { class: "grid-3 dsg-hero" },
    heroStat(env, root, {
      primary: true,
      label: "Design decisions recorded",
      iconName: "design",
      value: decisions.length,
      format: (v) => int(v),
      sub: "Each one backed by a real measurement below",
      target: "dsg-decisions",
      aria: `${decisions.length} design decisions recorded, each with real evidence behind it. Jump to the decisions.`,
    }),
    heroStat(env, root, {
      label: "EHO's own gap to the optimum",
      iconName: "gauge",
      value: eho?.optimality_gap_pct,
      format: (v) => gapPct(v),
      sub: rnd ? `vs ${gapPct(rnd.optimality_gap_pct)} for plain random search, same ${oc.budget_per_metaheuristic ?? rnd.n_evaluations}-evaluation budget` : "Measured against exhaustive search",
      target: "dsg-eho",
      aria: `EHO, a candidate optimiser we evaluated, ends ${gapPct(eho?.optimality_gap_pct)} from the optimum${rnd ? `, random search ${gapPct(rnd.optimality_gap_pct)}` : ""}. Jump to the optimiser check.`,
    }),
    heroStat(env, root, {
      label: "Change in the objective we drafted",
      iconName: "chart",
      value: fh.absolute_spread,
      format: (v) => (v === 0 ? "0.0" : v.toExponential(2)),
      sub: nsc.length ? `Across every NSC from ${int(Math.min(...nsc))} to ${int(Math.max(...nsc))}: it never moves` : "Across every NSC tested",
      target: "dsg-objective",
      aria: `The objective we drafted changes by ${fh.absolute_spread} across every NSC tested. Jump to the objective check.`,
    })
  );
}

/* ------------------------------------------------------------------ decisions grid */

function buildDecisions(env, root, data) {
  const decisions = data.design_decisions;
  if (!Array.isArray(decisions) || !decisions.length) throw new Error("design_decisions is missing");
  const wrap = h("section", { class: "dsg-decisions-wrap reveal-on-scroll", id: "dsg-decisions", tabindex: "-1", "aria-labelledby": "dsg-decisions-title", "data-tour": "design-decisions" });
  wrap.append(
    h(
      "div",
      { class: "dsg-section-head" },
      h("div", {}, h("div", { class: "dsg-num" }, "Design decisions"), h("h2", { id: "dsg-decisions-title", class: "dsg-card-title" }, "What we chose, and why")),
      h("p", { class: "dsg-section-sub" }, "One card per decision: the choice, the rationale, the evidence behind it, and the file where it is recorded.")
    )
  );
  const grid = h("div", { class: "dsg-decisions" });
  decisions.forEach((c, i) => {
    const target = decisionTarget(c);
    let action = null;
    if (target?.tab) {
      action = h("a", { class: "btn sm ghost dsg-go", href: `#${target.tab}` }, h("span", {}, target.label), ico("arrowRight"));
    } else if (target?.card) {
      action = h("button", { class: "btn sm ghost dsg-go", type: "button", onClick: () => scrollToCard(root, target.card) }, h("span", {}, target.label), ico("arrowRight"));
    }
    const files = String(c.source_file || "")
      .split(/;\s*/)
      .filter(Boolean);
    grid.append(
      h(
        "article",
        { class: "card lift dsg-decision", style: { "--i": String(i) } },
        h("div", { class: "dsg-decision-top" }, h("span", { class: "dsg-decision-n" }, `Decision ${i + 1}`)),
        h("h3", { class: "dsg-decision-text" }, c.decision),
        h("div", { class: "dsg-decision-value" }, h("span", { class: "dsg-label" }, "Why"), h("p", {}, c.why)),
        h("div", { class: "dsg-evidence" }, h("span", { class: "dsg-label" }, "Evidence"), h("p", {}, c.evidence)),
        h(
          "div",
          { class: "dsg-decision-foot" },
          h("div", { class: "dsg-files" }, h("span", { class: "dsg-label" }, "Recorded in"), ...files.map((f) => h("code", {}, f))),
          action
        )
      )
    );
  });
  wrap.append(grid);
  return wrap;
}

/* ------------------------------------------------------------------ EHO card */

function buildEho(env, data) {
  const un = data.eho_unreachability;
  const oc = data.optimiser_comparison;
  if (!un || !Array.isArray(un.rows)) throw new Error("eho_unreachability.rows is missing");
  const card = designCard({ id: "dsg-eho", kicker: "Optimiser", title: "EHO's own initialisation and update rule can't reach the optimum", tour: "design-optimiser" });
  const dom = un.search_domain || oc?.search_domain || [8, 65536];
  const optimum = un.optimum;
  card.append(
    h(
      "p",
      { class: "dsg-takeaway" },
      `EHO's own initialisation rule only places herds between LH·N/2 and N/2, and its update rule only ever averages existing herds toward the matriarch, so the search can never leave that range. For every learning rate LH we tested, the true optimum, S = ${int(optimum)}, lies outside it.`
    ),
    h(
      "p",
      { class: "dsg-context" },
      `Chain of N = ${int(un.chain_length_n)} entries · search domain S from ${int(dom[0])} to ${int(dom[1])} · reaching S = ${int(optimum)} would need LH ≤ ${num(un.max_learning_rate_admitting_optimum, 4)}`
    )
  );

  const fmtS = (v) => sInt(v);
  const intervals = un.rows.map((r) => ({
    label: `LH = ${r.learning_rate}`,
    from: r.reachable_interval[0],
    to: r.reachable_interval[1],
    color: "--series-2",
    note: `${r.optimum_reachable ? "Contains" : "Excludes"} the optimum S = ${int(optimum)}. Best objective reachable ${num(r.best_possible_objective, 4)} s vs ${num(un.optimum_objective, 4)} s at the optimum.`,
  }));
  const rangeHost = h("div", {});
  withTableToggle(rangeHost, {
    render: (host) =>
      lazyChart(
        env,
        host,
        rangeChart,
        {
          domain: dom,
          log: true,
          intervals,
          markers: Number.isFinite(optimum) ? [{ label: `Optimum S = ${int(optimum)}`, value: optimum, note: "Found by exhaustive search over every integer S" }] : [],
          xLabel: "Segment length S (log scale)",
          format: fmtS,
          ariaLabel: `Reachable segment lengths per learning rate on a log axis from ${int(dom[0])} to ${int(dom[1])}: ${un.rows.map((r) => `LH ${r.learning_rate}: ${fmtS(r.reachable_interval[0])} to ${fmtS(r.reachable_interval[1])}`).join("; ")}. The optimum S = ${int(optimum)} lies outside every interval.`,
        },
        un.rows.length * 30 + 70
      ),
    table: {
      columns: [
        { key: "learning_rate", label: "Learning rate LH", num: true, format: (v) => String(v) },
        { key: "from", label: "Reachable from", num: true, format: fmtS },
        { key: "to", label: "Reachable to", num: true, format: fmtS },
        { key: "optimum_reachable", label: `Contains S = ${int(optimum)}?`, format: (v) => (v ? "Yes" : "No") },
        { key: "best_possible_objective", label: "Best reachable objective (s)", num: true, format: (v) => num(v, 4) },
      ],
      rows: un.rows.map((r) => ({ ...r, from: r.reachable_interval[0], to: r.reachable_interval[1] })),
    },
  });
  card.append(
    panel(
      "Where each learning rate lets the herd search",
      "Each bar is the whole interval the init rule allows; the line is the true optimum",
      legend([
        { color: "--series-2", label: "The range this optimiser's own equations let it reach" },
        { color: "--text", label: "True optimum (exhaustive search)", shape: "line" },
      ]),
      rangeHost
    )
  );

  const lower = h("div", { class: "grid-2 dsg-lower" });
  if (oc && Array.isArray(oc.results)) {
    const gt = oc.ground_truth || {};
    const bars = oc.results.map((r) => {
      const isEho = /eho/i.test(r.optimiser);
      const short = isEho ? "EHO (the candidate we evaluated)" : r.optimiser.charAt(0).toUpperCase() + r.optimiser.slice(1);
      return { id: r.optimiser, label: short, value: r.optimality_gap_pct, color: isEho ? "--series-2" : "--series-1", note: `${r.optimiser}: best S = ${int(r.best_segment_length)}, objective ${num(r.best_objective, 6)} s, ${int(r.n_evaluations)} evaluations` };
    });
    const gapHost = h("div", {});
    withTableToggle(gapHost, {
      render: (host) =>
        lazyChart(
          env,
          host,
          barChart,
          {
            horizontal: true,
            height: bars.length * 34 + 36,
            yFormat: gapPct,
            valueName: "Gap to the optimum",
            bars,
            ariaLabel: `Optimality gap on the same budget: ${bars.map((b) => `${b.label} ${gapPct(b.value)}`).join(", ")}.`,
          },
          bars.length * 34 + 60
        ),
      table: {
        columns: [
          { key: "optimiser", label: "Optimiser" },
          { key: "best_segment_length", label: "Best S", num: true, format: (v) => int(v) },
          { key: "best_objective", label: "Objective (s)", num: true, format: (v) => num(v, 6) },
          { key: "n_evaluations", label: "Evaluations", num: true, format: (v) => int(v) },
          { key: "optimality_gap_pct", label: "Gap", num: true, format: (v) => (v == null ? "—" : gapPct(v)) },
        ],
        rows: [{ optimiser: gt.optimiser || "Exhaustive (ground truth)", best_segment_length: gt.best_segment_length, best_objective: gt.best_objective, n_evaluations: gt.n_evaluations, optimality_gap_pct: null }, ...oc.results],
      },
    });
    lower.append(
      panel(
        "Same budget, same cost model",
        `Gap to the exhaustive optimum after ${oc.budget_per_metaheuristic ?? "the same number of"} evaluations each (lower is better)`,
        legend([
          { color: "--series-2", label: "EHO, the candidate we evaluated" },
          { color: "--series-1", label: "Baselines" },
        ]),
        gapHost
      )
    );
  }
  const hist = oc?.eho_history || [];
  if (hist.length > 1) {
    const first = hist[0];
    const last = hist[hist.length - 1];
    const herdHost = h("div", {});
    lazyChart(
      env,
      herdHost,
      lineChart,
      {
        height: 200,
        xLabel: "Iteration",
        yLabel: "Herd spread (S)",
        series: [{ id: "spread", label: "Herd spread", color: "--series-2", points: hist.map((r) => ({ x: r.iteration, y: r.herd_spread })) }],
        ariaLabel: `EHO herd spread by iteration: ${hist.map((r) => `${r.iteration}: ${sInt(r.herd_spread)}`).join(", ")}.`,
      },
      200
    );
    lower.append(
      panel(
        "Why more iterations never help",
        `The spread halves each iteration (${int(first.herd_spread)} → ${int(last.herd_spread)}) while the matriarch (the herd's leader) stays near S ≈ ${int(last.matriarch)}: the herd shrinks around where it started instead of moving towards S = ${int(optimum)}.`,
        herdHost
      )
    );
  }
  card.append(lower);

  card.append(
    h("div", { class: "dsg-why" }, h("div", { class: "dsg-why-title" }, ico("info"), h("span", {}, "The structural argument")), h("p", {}, un.argument), un.finding ? h("p", {}, h("strong", {}, un.finding)) : null)
  );

  const rr = oc?.recorded_run;
  if (rr) {
    const rrEho = (rr.results || []).find((r) => /eho/i.test(r.optimiser));
    card.append(
      h(
        "div",
        { class: "notice dsg-notice" },
        h("span", { class: "notice-icon", html: icon("clock") }),
        h(
          "div",
          {},
          h("strong", {}, `${rr.label ? rr.label.charAt(0).toUpperCase() + rr.label.slice(1) : "Recorded run"}, for comparison only. `),
          `Every other number on this page, including the decision cards, comes from this export (optimum S = ${int(oc.ground_truth?.best_segment_length)}). NEGATIVE_RESULTS.md (N2) recorded the optimum at S = ${int(rr.ground_truth?.best_segment_length)}${rrEho ? ` and an EHO gap of ${num(rrEho.optimality_gap_pct, 3)}%` : ""}. Hash, sign and verify times are measured on each machine${oc.calibration?.measured_on ? ` (this export: ${oc.calibration.measured_on})` : ""}, so the optimum moves by a few units between runs; the conclusion does not.${oc.grid_ground_truth?.note ? ` ${oc.grid_ground_truth.note}` : ""}`
        )
      )
    );
  }

  card.append(
    provenanceFoot(env, {
      items: [
        { provenance: un.provenance, source: un.source, label: "Reachable intervals" },
        ...(oc ? [{ provenance: oc.provenance, source: oc.source, label: "Optimiser comparison" }] : []),
        ...(rr ? [{ provenance: rr.provenance, source: rr.source, label: rr.label ? rr.label.charAt(0).toUpperCase() + rr.label.slice(1) : "Recorded run" }] : []),
      ],
      command: CMD.report,
    })
  );
  return card;
}

/* ------------------------------------------------------------------ objective-invariance card */

function buildFh(env, data) {
  const fh = pick(data, ["fh_invariance", "objective_invariance", "fh_invariance_check"]);
  if (!fh || !Array.isArray(fh.nsc_values) || !Array.isArray(fh.fh_values)) throw new Error("the objective-invariance data is missing");
  const card = designCard({ id: "dsg-objective", kicker: "Objective", title: "The objective we drafted ignores the variable it optimises" });
  const nsc = fh.nsc_values;
  const vals = fh.fh_values;
  const lo = Math.min(...nsc);
  const hi = Math.max(...nsc);
  const fold = hi / lo;
  card.append(
    h(
      "p",
      { class: "dsg-takeaway" },
      `Evaluated literally, the objective we drafted gives exactly the same value for every NSC from ${int(lo)} to ${int(hi)}, a ${int(fold)}-fold range: absolute spread ${fh.absolute_spread === 0 ? "0.0" : fh.absolute_spread}. The objective the optimiser searches over does not depend on the thing it searches.`
    ),
    h("div", { class: "dsg-eq", role: "note", "aria-label": "The delay/energy objective we drafted" }, h("span", { class: "dsg-label" }, "Equation"), h("code", {}, fh.equation))
  );

  const maxAbs = Math.max(...vals.map((v) => Math.abs(v)).filter(Number.isFinite));
  const e = maxAbs > 0 ? Math.floor(Math.log10(maxAbs)) : 0;
  const scale = Math.pow(10, -e);
  const scaled = vals.map((v) => v * scale);
  const yTop = Math.max(1, Math.ceil(Math.max(...scaled) * 1.5));
  const log2 = (v) => Math.log2(v);
  const unitLabel = e === 0 ? "fh (J·s)" : `fh (×10${superscript(e)} J·s)`;
  const chartHost = h("div", {});
  withTableToggle(chartHost, {
    render: (host) =>
      lazyChart(
        env,
        host,
        lineChart,
        {
          height: 230,
          xLabel: "NSC (log scale)",
          yLabel: unitLabel,
          xFormat: (v) => int(Math.pow(2, v)),
          yFormat: (v) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(5)),
          yDomain: [0, yTop],
          series: [{ id: "fh", label: "fh (drafted objective)", color: "--series-2", points: nsc.map((n, i) => ({ x: log2(n), y: scaled[i] })) }],
          ariaLabel: `fh against NSC on a log axis: ${nsc.map((n, i) => `${int(n)}: ${vals[i].toExponential(5)}`).join(", ")}. The line is flat.`,
        },
        230
      ),
    table: {
      columns: [
        { key: "nsc", label: "NSC", num: true, format: (v) => int(v) },
        { key: "fh", label: "fh (J·s)", num: true, format: (v) => (Number.isFinite(v) ? v.toExponential(5) : "—") },
      ],
      rows: nsc.map((n, i) => ({ nsc: n, fh: vals[i] })),
    },
  });

  const chip = (value, label) => h("div", { class: "dsg-chip" }, h("span", { class: "dsg-chip-value" }, value), h("span", { class: "dsg-chip-label" }, label));
  const chips = h(
    "div",
    { class: "dsg-chips" },
    chip(fh.absolute_spread === 0 ? "0.0" : String(fh.absolute_spread), "absolute spread of fh"),
    chip(fh.relative_spread === 0 ? "0.0" : String(fh.relative_spread), "relative spread"),
    chip(fh.d_fh_d_nsc_is_zero ? "0" : "≠ 0", "d fh / d NSC")
  );
  card.append(h("div", { class: "dsg-split" }, panel("fh for each split point NSC", "A dead-flat line: every point sits at the same value", chartHost), panel("What that means", "Measured, not estimated", chips)));

  const defects = h("ul", { class: "dsg-defects" });
  if (fh.units) defects.append(h("li", {}, ico("alert", "dsg-defect-ico"), h("div", {}, h("strong", {}, "Units. "), `A sum of delays multiplied by an energy has units of ${fh.units}.`)));
  if (fh.undefined_symbol) defects.append(h("li", {}, ico("alert", "dsg-defect-ico"), h("div", {}, h("strong", {}, "Undefined symbol. "), `${fh.undefined_symbol}.`)));
  card.append(
    h("div", { class: "dsg-why" }, h("div", { class: "dsg-why-title" }, ico("info"), h("span", {}, "Two more defects in the same draft")), defects),
    h(
      "p",
      { class: "dsg-context" },
      "So any block-delay, energy or throughput gain attributed to optimising NSC with this objective can't have actually come from it.",
      fh.parameters_note ? ` ${fh.parameters_note}` : ""
    ),
    provenanceFoot(env, { items: [{ provenance: fh.provenance, source: fh.source }], command: CMD.report })
  );
  return card;
}

/* ------------------------------------------------------------------ closing */

function buildInstead(data) {
  const oc = data.optimiser_comparison;
  const base = (oc?.results || []).filter((r) => !/eho/i.test(r.optimiser));
  const worst = base.length ? Math.max(...base.map((r) => r.optimality_gap_pct).filter(Number.isFinite)) : null;
  const gt = oc?.ground_truth;
  const links = [
    { tab: "results", iconName: "results", title: "Honest evaluation", text: "AP and MCC on real held-out replicas, and a leakage certificate that shows what a random split would have hidden." },
    { tab: "lab", iconName: "lab", title: "Train it yourself", text: "Train the real architecture in your browser on synthetic data, and watch a leaky split inflate the score." },
    { tab: "detector", iconName: "shield", title: "A ledger that stores hashes, not telemetry", text: "Verdicts go into an RFC 6962 Merkle log whose leaves hash six fields and include no IP, URL, username or body. Try rewriting history and watch the proof fail." },
    { tab: "about", iconName: "about", title: "Scope and limits", text: "What TESSERA claims, what it deliberately does not, and why." },
  ];
  return h(
    "section",
    { class: "card dsg-card dsg-instead reveal-on-scroll", id: "dsg-instead", "aria-labelledby": "dsg-instead-title" },
    h("div", { class: "dsg-card-head" }, h("div", { class: "dsg-kicker" }, h("span", { class: "dsg-num" }, "What next")), h("h2", { class: "dsg-card-title", id: "dsg-instead-title" }, "What we built instead")),
    gt && Number.isFinite(worst)
      ? h(
          "p",
          { class: "dsg-takeaway" },
          `A well-posed cost model with a real interior optimum (S = ${int(gt.best_segment_length)}), which plain random search and Optuna TPE both reach to within ${gapPct(worst)}, plus the pieces below, each checked against real data or a reference implementation.`
        )
      : null,
    h(
      "div",
      { class: "dsg-links" },
      ...links.map((l) =>
        h(
          "a",
          { class: "card lift dsg-link", href: `#${l.tab}` },
          h("span", { class: "dsg-link-ico", html: icon(l.iconName) }),
          h("span", { class: "dsg-link-body" }, h("span", { class: "dsg-link-title" }, l.title), h("span", { class: "dsg-link-text" }, l.text)),
          h("span", { class: "dsg-link-go", html: icon("arrowRight") })
        )
      )
    )
  );
}

/* ------------------------------------------------------------------ mount */

function skeleton() {
  return h(
    "div",
    { class: "dsg dsg-loading", "aria-busy": "true", "aria-label": "Loading design decisions" },
    h("div", { class: "skeleton", style: { height: "22px", width: "120px" } }),
    h("div", { class: "skeleton", style: { height: "40px", width: "min(560px, 100%)" } }),
    h("div", { class: "grid-3" }, ...[0, 1, 2].map(() => h("div", { class: "skeleton", style: { height: "118px" } }))),
    h("div", { class: "skeleton", style: { height: "300px" } })
  );
}

export async function mount(el, ctx = {}) {
  const store = ctx.store || (await import("../store.js")).store;
  const env = makeEnv(ctx);
  el.replaceChildren(skeleton());
  let data;
  try {
    data = await store.loadJSON("data/design_notes.json");
    if (!data || typeof data !== "object") throw new Error("the file is empty");
  } catch (err) {
    el.replaceChildren(h("div", { class: "mount-error", role: "alert" }, `The design notes could not be loaded (${err.message}). It lives in data/design_notes.json, written by ${CMD.export}.`));
    return null;
  }

  const root = h("div", { class: "dsg" });
  const top = h("div", { class: "dsg-top" }, pageHead(), section("headline numbers", () => hero(env, root, data)));
  const when = data.generated_utc ? new Date(data.generated_utc) : null;
  const whenText = when && !Number.isNaN(when.getTime()) ? `${when.toISOString().slice(0, 16).replace("T", " ")} UTC` : null;
  root.append(
    top,
    section("decisions", () => buildDecisions(env, root, data)),
    section("optimiser check", () => buildEho(env, data)),
    section("objective check", () => buildFh(env, data)),
    section("what we built instead", () => buildInstead(data)),
    h(
      "div",
      { class: "dsg-colophon" },
      h("p", { class: "dsg-context" }, `${whenText ? `Page data exported ${whenText}` : "Page data exported"} by ${data.generated_by || CMD.export}.`),
      data.provenance_note ? h("details", { class: "dsg-src" }, h("summary", {}, "How this page's numbers were exported"), h("p", {}, data.provenance_note)) : null
    )
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
