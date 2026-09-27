// About tab: model card, how we evaluate, reproduce-it-yourself commands, a viva
// FAQ, tech stack and credits.
//
// Every answer here is taken from the repository's own records (README.md,
// RESULTS.md, NEGATIVE_RESULTS.md, THREAT_MODEL.md, LICENSE-DATA and the source
// docstrings they cite). Numbers are read from data/real_results.json and
// data/replica_stats.json at load time, never typed in. The one interactive
// calculator is plain arithmetic and is labelled as an illustration.

import { h, s, stagger, revealOnScroll, reducedMotion, fmt, copyText, syncRangeFill, escapeHtml, debounce } from "../ui/dom.js";
import { icon } from "../ui/icons.js";
import { makeRng } from "../lab/rng.js";

const COPY_ICON =
  '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 00-2-2H6a2 2 0 00-2 2v8a2 2 0 002 2h2"/></svg>';

const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const ap3 = (v) => (isNum(v) ? v.toFixed(3) : "—");
const ap4 = (v) => (isNum(v) ? v.toFixed(4) : "—");
const pct0 = (v) => (isNum(v) ? `${Math.round(Math.min(1, Math.max(0, v)) * 100)}%` : "—");

function realTag(title) {
  return h("span", { class: "tag-real", title: title || "Measured on real AIT data" }, "Real");
}
function srcNote(text) {
  return h("div", { class: "ab-src", html: `${icon("file")}<span>${escapeHtml(text)}</span>` });
}
function fromMarkup(markup) {
  const t = document.createElement("template");
  t.innerHTML = markup.trim();
  return t.content.firstElementChild;
}
function inView(el, fn, { once = true } = {}) {
  if (typeof IntersectionObserver !== "function") {
    fn(true);
    return () => {};
  }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      fn(e.isIntersecting);
      if (e.isIntersecting && once) io.disconnect();
    }
  });
  io.observe(el);
  return () => io.disconnect();
}

/* ------------------------------------------------------------------ data */

async function loadData(store) {
  const settle = (p) => p.then((v) => v, () => null);
  const [real, stats, weights] = await Promise.all([
    settle(store.loadJSON("data/real_results.json")),
    settle(store.loadJSON("data/replica_stats.json")),
    settle(store.loadPretrainedWeights()),
  ]);
  return { real, stats, weights };
}

function countParams(w) {
  try {
    const lin = (l) => l.weight.flat().length + l.bias.length;
    let n = 0;
    for (const e of w.encoders) n += lin(e.linear0) + lin(e.linear1) + e.groupnorm.weight.length + e.groupnorm.bias.length;
    return n + lin(w.fusion.gate_linear0) + lin(w.fusion.gate_linear1) + lin(w.head.linear0) + lin(w.head.linear1);
  } catch {
    return null;
  }
}

function derive({ real, stats, weights }) {
  const D = {};
  const loro = real?.loro;
  if (loro?.summary?.average_precision) {
    D.ap = loro.summary.average_precision;
    D.mcc = loro.summary.mcc;
    D.naive = loro.summary.naive_all_folds_ap;
    D.excluded = loro.summary.excluded_low_support || [];
    D.minSupport = loro.summary.min_support ?? 20;
    D.folds = loro.folds || [];
    D.shaw = D.folds.find((f) => f.held_out === "shaw");
    D.harrison = D.folds.find((f) => f.held_out === "harrison");
  }
  D.minSupport = D.minSupport ?? stats?.min_support ?? 20;
  const ds = real?.dataset;
  if (ds?.replicas?.length) {
    D.nReplicas = ds.replicas.length;
    D.nWindows = ds.totals?.n_windows;
    D.nPositive = ds.totals?.n_positive;
    const sorted = [...ds.replicas].sort((a, b) => a.prevalence - b.prevalence);
    D.minPrev = sorted[0];
    D.maxPrev = sorted[sorted.length - 1];
    D.replicas = ds.replicas;
  }
  D.leak = real?.leakage_duplicates || null;
  D.calendar = real?.calendar_leak?.rows?.length >= 2 ? real.calendar_leak : null;
  D.allMod = real?.all_modalities_r0_r1 || null;
  D.tb = real?.tessera_base || null;
  const fl = real?.mask_only_floor;
  D.floor = fl?.summary?.average_precision && Number.isFinite(fl.summary.average_precision.mean) ? fl : null;
  const spans = (ds?.replicas || []).map((r) => r.span_hours).filter(Number.isFinite);
  D.spanDays = spans.length ? [Math.min(...spans) / 24, Math.max(...spans) / 24] : null;
  const abl = real?.attribution_vs_ablation;
  if (abl?.ablation) {
    const find = (re) => abl.ablation.find((r) => re.test(r.check));
    D.abl = { m1Alone: find(/M1 alone/i), m3Alone: find(/M3 alone/i), hostAlone: find(/host_bucket alone/i), withM1: find(/with M1/i), withoutM1: find(/without M1/i), attr: abl.mean_attribution };
  }
  D.params = countParams(weights) ?? D.tb?.n_parameters ?? null;
  const gp = stats?.global_pooled;
  const availOf = (cls) => {
    const c = gp?.[cls];
    if (!c || !(c.n >= (stats?.min_support ?? 20)) || !Array.isArray(c.patterns)) return null;
    let m1 = 0;
    for (const p of c.patterns) m1 += (p.a?.[0] ? 1 : 0) * p.p;
    return { m1: Math.min(1, m1) };
  };
  D.avail = gp ? { benign: availOf("benign"), attack: availOf("attack") } : null;
  return D;
}

/** A prevalence below min_support positives is shown as a count, never a rate. */
function prevPhrase(r, minSupport) {
  if (!r) return "—";
  if (r.n_positive < minSupport) return `${fmt.int(r.n_positive)} of ${fmt.int(r.n_windows)} windows`;
  return fmt.pct(r.prevalence, 1);
}

/* ------------------------------------------------------------------ page head + jump nav */

function buildHead(sections) {
  const jump = h("nav", { class: "ab-jump", "aria-label": "On this page" });
  for (const sec of sections) {
    jump.appendChild(
      h("button", {
        class: "ab-jump-btn",
        type: "button",
        html: `${icon(sec.icon)}<span>${escapeHtml(sec.label)}</span>`,
        onClick: () => {
          const target = document.getElementById(sec.id);
          if (!target) return;
          target.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
          const heading = target.querySelector("h2");
          if (heading) {
            heading.setAttribute("tabindex", "-1");
            heading.focus({ preventScroll: true });
          }
        },
      })
    );
  }
  return h(
    "header",
    { class: "page-head ab-head" },
    h(
      "div",
      {},
      h("div", { class: "eyebrow" }, "About"),
      h("h1", {}, "Model card, method and viva FAQ"),
      h(
        "p",
        { class: "lede" },
        "What the model is for, how it was measured, how to reproduce the numbers yourself, and straight answers to the questions a panel is likely to ask. Every answer comes from the project's own records."
      )
    ),
    jump
  );
}

function section(id, eyebrow, title, sub, ...body) {
  return h(
    "section",
    { class: "ab-section reveal-on-scroll", id, "aria-labelledby": `${id}-h` },
    h(
      "div",
      { class: "ab-section-head" },
      h("div", { class: "eyebrow" }, eyebrow),
      h("h2", { id: `${id}-h` }, title),
      sub ? h("p", { class: "ab-section-sub" }, sub) : null
    ),
    ...body
  );
}

/* ------------------------------------------------------------------ model card */

function bulletCard(iconName, title, items, { tone = "" } = {}) {
  const list = h("ul", { class: "ab-list" });
  for (const it of items) {
    if (!it) continue;
    const li = h("li", {});
    if (typeof it === "string") li.textContent = it;
    else {
      li.appendChild(h("span", {}, it.text));
      if (it.tag) li.appendChild(h("span", { class: "ab-li-tag" }, it.tag));
    }
    list.appendChild(li);
  }
  return h(
    "article",
    { class: `card ab-mc-card${tone ? ` is-${tone}` : ""}` },
    h("div", { class: "ab-mc-head" }, h("span", { class: "ab-mc-icon", html: icon(iconName) }), h("h3", {}, title)),
    list
  );
}

function buildModelCard(D) {
  const tb = D.tb;
  const b = D.avail?.benign;
  const a = D.avail?.attack;
  const minS = D.minSupport;

  const summary = h(
    "div",
    { class: "card accent ab-mc-summary" },
    h(
      "div",
      { class: "ab-mc-id" },
      h("span", { class: "ab-mc-mark", "aria-hidden": "true" }, h("span"), h("span"), h("span"), h("span")),
      h("div", {}, h("div", { class: "eyebrow" }, "Model"), h("h3", { class: "ab-mc-name" }, "TESSERA-base"), h("p", { class: "small muted" }, "A small multimodal anomaly detector for 60-second host windows, with availability-masked gated fusion."))
    ),
    h(
      "div",
      { class: "ab-mc-facts" },
      fact("Parameters", D.params ? fmt.int(D.params) : "—", "Counted from data/weights.json"),
      fact("Training windows", tb ? fmt.int(tb.n_train_windows) : "—", "from 7 real replicas; santos held out"),
      fact("Held-out AP (one fold)", tb ? ap4(tb.tessera_ap) : "—", tb ? `MCC ${ap4(tb.tessera_mcc)} on santos; one run` : ""),
      fact("Training time", tb ? `${tb.train_seconds} s` : "—", tb ? tb.device : "")
    ),
    h("div", { class: "ab-mc-prov" }, realTag("Transcribed from RESULTS.md finding 4; recorded by tests/test_tessera_base.py::test_tessera_base_matches_lightgbm_on_real_data"), h("span", { class: "tiny faint" }, "Training figures from RESULTS.md finding 4 (tests/test_tessera_base.py); parameter count from the shipped weights."))
  );

  function fact(label, value, sub) {
    return h("div", { class: "ab-mc-fact" }, h("div", { class: "stat-label" }, label), h("div", { class: "ab-mc-val tabular" }, value), h("div", { class: "stat-sub" }, sub));
  }

  const grid = h(
    "div",
    { class: "grid ab-mc-grid" },
    bulletCard("check", "Intended use", [
      "Research and teaching: how multimodal telemetry fusion and leakage-aware evaluation behave on a public, labelled benchmark.",
      "Scoring 60-second host windows built by this project's own 42-feature pipeline (M1 to M4) from AIT-style logs.",
      "Demonstrating a tamper-evident verdict log whose entries hash six fields and include no IP, URL, username or body.",
    ]),
    bulletCard("x", "Out of scope", [
      "Protecting a production network or taking automated blocking decisions: nothing here has been tested outside the AIT scenario.",
      "Claims about other environments or attack types: cross-replica results are not cross-organisation transfer.",
      "Adversarial settings: evasion, poisoning and concept drift are explicitly out of scope (THREAT_MODEL.md).",
      "Explaining a single verdict from the gate weights alone.",
      "Commercial use of anything derived from AIT data (the licence is NonCommercial).",
    ]),
    bulletCard("database", "Data", [
      D.nReplicas
        ? { text: `AIT Log Data Set V2.1 (Landauer et al., Zenodo record 19483937): ${D.nReplicas} replicas, ${fmt.int(D.nWindows)} windows, ${fmt.int(D.nPositive)} of them attacks.`, tag: realTag("Counted from the real cache by the exporter") }
        : "AIT Log Data Set V2.1 (Landauer et al., Zenodo record 19483937).",
      "The same fixed subset in every replica: 3 hosts, four log files plus Suricata network events, 60-second windows, 42 features.",
      D.minPrev ? { text: `The attack rate per replica ranges from ${prevPhrase(D.minPrev, minS)} (${D.minPrev.id}) to ${prevPhrase(D.maxPrev, minS)} (${D.maxPrev.id}).`, tag: realTag("real_results.json, dataset.replicas") } : null,
      tb ? { text: `TESSERA-base trained on the seven replicas other than santos (${fmt.int(tb.n_train_windows)} windows after a 15% validation carve) and was tested on santos.`, tag: realTag("RESULTS.md finding 4") } : null,
      "This site ships aggregate statistics only; the Live detector and Training Lab windows are synthetic.",
    ]),
    bulletCard("gauge", "Metrics", [
      "Average precision first: it needs no threshold, and chance level equals the attack rate.",
      "Matthews correlation (MCC) second, at a fixed 0.5 threshold.",
      "Accuracy is never a headline.",
      `A fold or class with fewer than ${minS} positives is reported as a count and kept out of summary means.`,
      "Calibration is reported per class: at high benign prevalence an overall calibration error mostly describes the benign windows.",
    ]),
    bulletCard(
      "alert",
      "Measured limitations",
      [
        D.harrison ? { text: `harrison: AP ${ap3(D.harrison.average_precision)} but MCC ${ap3(D.harrison.mcc)}. The ranking is near-perfect but the scores are shifted for that replica, so a fixed 0.5 cut-off misfires.`, tag: realTag("real_results.json, loro.folds") } : null,
        D.shaw ? { text: `shaw: only ${fmt.int(D.shaw.n_pos_test)} attack windows in its test set (AP ${ap3(D.shaw.average_precision)}); reported, but kept out of the mean.`, tag: realTag("real_results.json, loro.folds") } : null,
        D.abl?.attr ? { text: `The gate's own attribution (M1 ${pct0(D.abl.attr.m1_log)}, M3 ${pct0(D.abl.attr.m3_identity)}) understates how much the model needs log templates.`, tag: realTag("RESULTS.md finding 4") } : null,
        "With all four sources, the E1 halt gate's 0.02 margin was not cleared (gap about 0.015). It was investigated and found benign rather than waved through.",
        D.floor
          ? {
              text: `Which sources are present is itself a strong signal: labels come only from the log files M1 reads, so every attack window has log activity. A model given only the presence bits scores mean AP ${ap3(D.floor.summary.average_precision.mean)} across held-out replicas; see the FAQ "Is log-template presence just the label?".`,
              tag: realTag("real_results.json, mask_only_floor (recomputed)"),
            }
          : b && a
            ? { text: `Whether a source is present differs by class (log templates in ${pct0(b.m1)} of benign vs ${pct0(a.m1)} of attack windows), so presence alone carries signal.`, tag: realTag("data/replica_stats.json, pooled over 8 replicas") }
            : null,
        D.tb ? { text: `The neural model was compared with LightGBM on one held-out fold (santos) from one run; an 8-fold paired comparison is future work (NEGATIVE_RESULTS C2).`, tag: realTag("RESULTS.md finding 4") } : null,
        "Cross-replica is not cross-organisation: the replicas share one environment and attack repertoire.",
      ],
      { tone: "warn" }
    ),
    bulletCard("shield", "Ethics and privacy", [
      "Hash on chain, data off chain: no IP, geolocation, username, URL, header or body reaches the log, and a test enforces it (test_e2e, test_verdict_canonical_cross_language).",
      "host_hash is meant to be a salted pseudonym, but the salt helper (ledger/sth.py) is not yet wired in and the leaf hash itself is unsalted. The M3 host bucket is a hash of one of three known hosts, so it is a label, not a privacy measure. Destination IPs are cut to /24 subnets.",
      "Pseudonymisation here is a design argument, not an empirical privacy result: injection and known-flow attacks can unravel prefix-preserving schemes.",
      "AIT is fully synthetic, so no claim about protecting real people is made.",
      "The signing key lives outside the repository, guarded by a build-failing test (test_no_secrets_tracked).",
      "This site runs locally: nothing you do on it is sent anywhere.",
    ])
  );
  stagger(grid);

  const details = h(
    "details",
    { class: "acc ab-details" },
    h("summary", {}, h("span", { class: "row" }, h("span", { html: icon("cpu") }), "Architecture and training recipe")),
    h(
      "div",
      { class: "acc-body" },
      h(
        "div",
        { class: "table-wrap" },
        h(
          "table",
          { class: "table ab-table" },
          h("caption", { class: "sr-only" }, "TESSERA-base architecture and training recipe"),
          h(
            "tbody",
            {},
            row("Encoders", "One per source: Linear → GroupNorm (4 groups) → GELU → Linear, from each source's features to 32 then 16 numbers. Features go in raw; GroupNorm copes with scale."),
            row("Fusion", "A gated multimodal unit: the four embeddings plus 4 availability bits (68 inputs) → 8 → 4 gate logits; missing sources are masked out before the softmax."),
            row("Head", "16 → 32 → 1 with dropout 0.2, then a sigmoid: the attack score."),
            row("Training", "AdamW (learning rate 1e-3, weight decay 1e-2), cosine schedule, class-weighted binary cross-entropy, gradient norm clipped at 1.0, batch 256, up to 30 epochs with early stopping on validation AP (patience 5)."),
            row("GELU", "The tanh approximation on purpose: JavaScript has no built-in erf, so both sides use the same closed form. The browser forward pass is shown to match PyTorch to within 4.4e-7 on 40 reference vectors (cd web && npm test)."),
            row("Source", "src/tessera/models/tessera_base.py, models/fusion/gmu.py, models/encoders/small_mlp.py")
          )
        )
      )
    )
  );
  function row(k, v) {
    return h("tr", {}, h("th", { scope: "row" }, k), h("td", { class: "ab-td-wrap" }, v));
  }

  return section("about-model-card", "Model card", "What the model is, and what it is not for", "The format follows the usual model-card headings: intended use, out-of-scope uses, data, metrics, measured limitations, and ethical considerations.", summary, grid, details);
}

/* ------------------------------------------------------------------ how we evaluate */

const ROLE_COLORS = { train: "var(--c-train)", val: "var(--c-val)", test: "var(--c-test)", unused: "var(--c-unused)" };
const ROLE_LABELS = { train: "Train", val: "Validation", test: "Test", unused: "Not used (gap or dropped)" };

/** Deterministic split pictogram: rows x cols cells coloured by role. */
function splitPicto(kind, { rows, cols }) {
  const rng = makeRng(7, `about-picto-${kind}`);
  const grid = [];
  for (let r = 0; r < rows; r++) {
    const line = [];
    for (let c = 0; c < cols; c++) {
      let role = "train";
      if (kind === "r0") {
        const u = rng.next();
        role = u < 0.6 ? "train" : u < 0.8 ? "val" : "test";
      } else if (kind === "r1" || kind === "r2") {
        const t1 = Math.round(cols * 0.6);
        const t2 = Math.round(cols * 0.8);
        role = c < t1 ? "train" : c === t1 ? "unused" : c < t2 ? "val" : c === t2 ? "unused" : "test";
        // R2: a short-lived entity (session) seen in training is dropped from the later partition.
        if (kind === "r2" && role !== "train" && role !== "unused" && rng.next() < 0.14) role = "unused";
      } else if (kind === "r3") {
        role = rng.next() < 0.15 ? "val" : "train";
      }
      line.push(role);
    }
    grid.push(line);
  }
  const cell = 9;
  const gap = 2;
  const W = cols * (cell + gap) - gap;
  const H = rows * (cell + gap) - gap;
  const svg = s("svg", { class: "ab-picto", viewBox: `0 0 ${W} ${H + 14}`, role: "img", "aria-label": `${kind.toUpperCase()} split pictogram` });
  const cells = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const rect = s("rect", { x: c * (cell + gap), y: r * (cell + gap), width: cell, height: cell, rx: 2, class: "ab-cell" });
      rect.style.fill = ROLE_COLORS[grid[r][c]];
      rect.style.setProperty("--d", `${(c * 12 + r * 30) % 600}ms`);
      rect.dataset.role = grid[r][c];
      rect.dataset.row = String(r);
      svg.appendChild(rect);
      cells.push(rect);
    }
  }
  // time axis
  svg.appendChild(s("line", { x1: 0, y1: H + 7, x2: W - 6, y2: H + 7, class: "ab-axis" }));
  svg.appendChild(s("path", { d: `M${W - 9} ${H + 4} L${W - 4} ${H + 7} L${W - 9} ${H + 10}`, class: "ab-axis" }));
  return { svg, cells, grid, rows, cols };
}

function buildEvaluate(D) {
  const legend = h(
    "div",
    { class: "viz-legend ab-legend", "aria-label": "Legend" },
    ...["train", "val", "test", "unused"].map((k) => {
      const sw = h("span", { class: "swatch" });
      sw.style.background = ROLE_COLORS[k];
      return h("span", { class: "item" }, sw, ROLE_LABELS[k]);
    }),
    h("span", { class: "item faint" }, "Rows are replicas or hosts; time runs left to right")
  );

  const specs = [
    {
      kind: "r0",
      code: "R0",
      title: "Random split",
      pill: ["warn", "alert", "Leakage upper bound"],
      text: "Windows are shuffled and dealt into train, validation and test at random. Neighbouring minutes and exact duplicates land on both sides, so the model is tested on near-copies of what it trained on. Reported only as an upper bound, never as a capability claim.",
      num: D.leak ? [`AP ${ap3(D.leak.r0_random_ap)}`, `${D.leak.replica}, network metrics only`] : null,
      rows: 3,
      cols: 22,
    },
    {
      kind: "r1",
      code: "R1",
      title: "Chronological split",
      pill: ["info", "clock", "Honest in time"],
      text: "Each replica is cut by time: train on the first 60%, validate on the next 20%, test on the last 20%, with a 10-minute gap at each cut so no window straddles it.",
      num: D.leak ? [`AP ${ap3(D.leak.r1_chronological_ap)}`, "same data and features as R0"] : null,
      rows: 3,
      cols: 22,
    },
    {
      kind: "r2",
      code: "R2",
      title: "Chronological + entity-disjoint",
      pill: ["info", "split", "Honest in time and entity"],
      text: "R1, plus short-lived entities such as sessions stay on one side of each cut. Hosts can't be split this way: they live for the whole capture, so a time cut plus host-disjointness empties the test set (measured, NEGATIVE_RESULTS.md C3). Host-level separation comes from R3.",
      num: null,
      rows: 3,
      cols: 22,
    },
    {
      kind: "r3",
      code: "R3",
      title: "Leave one replica out",
      pill: ["good", "check", "Headline protocol"],
      text: "Train on seven replicas and test on the eighth, once per replica. The replicas are randomised runs of one scenario, so this measures robustness to that randomisation, not transfer to a new organisation.",
      num: D.ap ? [`mean AP ${ap3(D.ap.mean)} ± ${ap3(D.ap.std)}`, `${D.ap.n} folds; ${D.excluded.join(", ")} reported but kept out`] : null,
      rows: 8,
      cols: 22,
    },
  ];

  const grid = h("div", { class: "grid-2 ab-split-grid" });
  const cyclers = [];
  for (const sp of specs) {
    const picto = splitPicto(sp.kind, { rows: sp.rows, cols: sp.cols });
    const pictoWrap = h("div", { class: "ab-picto-wrap" }, picto.svg);
    picto.svg.setAttribute(
      "aria-label",
      sp.kind === "r3"
        ? "Eight rows of windows; one whole row is the test set and the rest train, and the held-out row changes each fold."
        : sp.kind === "r0"
          ? "Three rows of windows with train, validation and test cells scattered at random."
          : sp.kind === "r1"
            ? "Three rows of windows: the first 60% train, a gap, 20% validation, a gap, the last 20% test."
            : "Like R1, with a few later cells dropped because their session also appears in training."
    );
    const numEl = sp.num
      ? h("div", { class: "ab-split-num" }, h("span", { class: "ab-split-val tabular" }, sp.num[0]), h("span", { class: "tiny faint" }, sp.num[1]), realTag(sp.kind === "r3" ? "real_results.json, loro (recomputed; tests/test_loro_real.py)" : "real_results.json, leakage_duplicates (recomputed; tests/test_pipeline_real_data.py protocol)"))
      : null;
    const foldLabel = sp.kind === "r3" ? h("div", { class: "ab-fold tiny muted", "aria-live": "off" }) : null;
    const card = h(
      "article",
      { class: "card ab-split" },
      h(
        "div",
        { class: "ab-split-head" },
        h("span", { class: "ab-split-code" }, sp.code),
        h("h3", {}, sp.title),
        h("span", { class: `pill ${sp.pill[0]}`, html: `${icon(sp.pill[1])}${escapeHtml(sp.pill[2])}` })
      ),
      pictoWrap,
      foldLabel,
      h("p", { class: "ab-split-text" }, sp.text),
      numEl
    );
    grid.appendChild(card);

    // Draw the cells in when the card is first seen.
    inView(card, (vis) => vis && pictoWrap.classList.add("is-in"));

    if (sp.kind === "r3") {
      const replicaIds = D.replicas?.map((r) => r.id) || null;
      let fold = replicaIds ? Math.max(0, replicaIds.indexOf("santos")) : 1;
      const paintFold = () => {
        for (const rect of picto.cells) {
          const r = Number(rect.dataset.row);
          const role = r === fold ? "test" : rect.dataset.role;
          rect.style.fill = ROLE_COLORS[role];
        }
        foldLabel.textContent = replicaIds ? `Fold ${fold + 1} of ${picto.rows}: ${replicaIds[fold] ?? "?"} held out` : `Fold ${fold + 1} of ${picto.rows}`;
      };
      paintFold();
      let timer = 0;
      const startCycle = () => {
        if (reducedMotion() || timer) return;
        timer = setInterval(() => {
          fold = (fold + 1) % picto.rows;
          paintFold();
        }, 1700);
      };
      const stopCycle = () => {
        clearInterval(timer);
        timer = 0;
      };
      inView(card, (vis) => (vis ? startCycle() : stopCycle()), { once: false });
      cyclers.push(stopCycle);
    }
  }
  stagger(grid);

  const lowSupport = h(
    "div",
    { class: "notice ab-note" },
    fromMarkup(icon("info", { cls: "notice-icon" })),
    h(
      "div",
      {},
      h("strong", {}, "The low-support rule. "),
      `A test set with fewer than ${D.minSupport} attack windows is reported in full but kept out of summary means and standard deviations (MIN_SUPPORT_FOR_RATES = ${D.minSupport} in src/tessera/eval/metrics.py). A recall of 1.0 measured on three positives is noise presented as a measurement.`
    )
  );

  const calc = buildCalculator(D);

  return {
    el: section(
      "about-evaluate",
      "How we evaluate",
      "Four ways to split the data, and why the choice decides the score",
      "The same model can look excellent or mediocre depending only on how train and test are separated. Reporting several protocols side by side shows how much of a number is the split.",
      legend,
      grid,
      lowSupport,
      calc
    ),
    stop: () => cyclers.forEach((f) => f()),
  };
}

/* ------------------------------------------------------------------ accuracy calculator */

let calcCount = 0;
const PI_MIN = 1e-4;
const PI_MAX = 0.5;
const toPi = (v) => Math.pow(10, Math.log10(PI_MIN) + (v / 1000) * (Math.log10(PI_MAX) - Math.log10(PI_MIN)));
const fromPi = (p) => Math.round(((Math.log10(p) - Math.log10(PI_MIN)) / (Math.log10(PI_MAX) - Math.log10(PI_MIN))) * 1000);

function basePhrase(p) {
  if (p >= 0.01) return `${fmt.pct(p, 1)} of windows (1 in ${fmt.int(Math.round(1 / p))})`;
  return `1 in ${fmt.int(Math.round(1 / p))} windows (${fmt.pct(p, p < 0.001 ? 3 : 2)})`;
}
function pctSmart(v) {
  if (!isNum(v)) return "—";
  if (v >= 0.999 && v < 1) return `${(v * 100).toFixed(2)}%`;
  if (v < 0.01) return `${(v * 100).toFixed(2)}%`;
  return `${(v * 100).toFixed(1)}%`;
}

function buildCalculator(D) {
  const idBase = `ab-calc-${++calcCount}`;
  const mkRange = (id, label, { min, max, step, value, hint }) => {
    const input = h("input", { type: "range", id, min, max, step, value });
    const out = h("output", { class: "ab-range-val tabular", for: id });
    const field = h("div", { class: "field ab-range" }, h("div", { class: "row between" }, h("label", { for: id }, label), out), input, hint ? h("div", { class: "hint" }, hint) : null);
    return { input, out, field };
  };
  const base = mkRange(`${idBase}-pi`, "How common attacks are", { min: 0, max: 1000, step: 1, value: fromPi(1e-4), hint: "Log scale, from 1 in 10,000 to one in two" });
  const tpr = mkRange(`${idBase}-tpr`, "Detection rate (recall)", { min: 50, max: 100, step: 0.5, value: 95 });
  const fpr = mkRange(`${idBase}-fpr`, "False-alarm rate on benign windows", { min: 0, max: 10, step: 0.1, value: 1 });

  const presets = h("div", { class: "ab-presets" }, h("span", { class: "tiny faint" }, "Try:"));
  const addPreset = (label, p, title, real) => {
    const b = h("button", { class: "btn sm ghost ab-preset", type: "button", title, onClick: () => setPi(p) }, label);
    if (real) b.appendChild(h("span", { class: "tag-real ab-preset-tag", title: "Attack rate of a real held-out test set (real_results.json)" }, "Real"));
    presets.appendChild(b);
  };
  addPreset("1 in 10,000", 1e-4, "The realistic base rate README.md uses", false);
  const santos = D.folds?.find((f) => f.held_out === "santos");
  if (santos && santos.n_pos_test >= D.minSupport) addPreset(`santos test set (${fmt.pct(santos.test_prevalence, 1)})`, santos.test_prevalence, "Attack rate of the real santos held-out test set", true);
  const wardbeck = D.folds?.find((f) => f.held_out === "wardbeck");
  if (wardbeck && wardbeck.n_pos_test >= D.minSupport) addPreset(`wardbeck test set (${fmt.pct(wardbeck.test_prevalence, 1)})`, wardbeck.test_prevalence, "Attack rate of the real wardbeck held-out test set", true);

  const precVal = h("div", { class: "hero-number ab-prec tabular" }, "—");
  const precSub = h("div", { class: "small muted" }, "of alerts are real attacks");
  const accVal = h("span", { class: "ab-acc tabular" }, "—");
  const faVal = h("span", { class: "ab-fa tabular" }, "—");
  const lazyVal = h("span", { class: "tabular" }, "—");

  const segReal = h("div", { class: "ab-bar-seg is-real" });
  const segFalse = h("div", { class: "ab-bar-seg is-false" });
  segReal.style.background = "var(--c-attack)";
  segFalse.style.background = "var(--c-benign)";
  const bar = h("div", { class: "ab-bar", role: "img" }, segReal, segFalse);
  const swatch = (color) => {
    const sw = h("span", { class: "swatch" });
    sw.style.background = color;
    return sw;
  };
  const barLegend = h(
    "div",
    { class: "viz-legend ab-bar-legend" },
    h("span", { class: "item" }, swatch("var(--c-attack)"), "Real attacks"),
    h("span", { class: "item" }, swatch("var(--c-benign)"), "False alarms (benign windows)")
  );

  const live = h("div", { class: "sr-only", "aria-live": "polite" });
  const announce = debounce((t) => (live.textContent = t), 400);

  function setPi(p) {
    base.input.value = String(fromPi(p));
    base.fill?.();
    update();
  }

  function update() {
    const pi = toPi(Number(base.input.value));
    const tprPct = Number(tpr.input.value);
    const fprPct = Number(fpr.input.value);
    const t = tprPct / 100;
    const f = fprPct / 100;
    base.out.textContent = basePhrase(pi);
    tpr.out.textContent = `${Number.isInteger(tprPct) ? tprPct : tprPct.toFixed(1)}%`;
    fpr.out.textContent = `${fprPct.toFixed(1)}%`;
    const acc = t * pi + (1 - f) * (1 - pi);
    const tp = t * pi;
    const fp = f * (1 - pi);
    const prec = tp + fp > 0 ? tp / (tp + fp) : null;
    const faPer = tp > 0 ? fp / tp : null;
    precVal.textContent = prec == null ? "no alerts" : pctSmart(prec);
    accVal.textContent = pctSmart(acc);
    faVal.textContent = faPer == null ? "—" : faPer < 10 ? faPer.toFixed(1) : fmt.int(faPer);
    lazyVal.textContent = pctSmart(1 - pi);
    const realShare = prec == null ? 0 : prec;
    segReal.style.width = `${(realShare * 100).toFixed(2)}%`;
    segFalse.style.width = `${((prec == null ? 0 : 1 - realShare) * 100).toFixed(2)}%`;
    const realN = prec == null ? 0 : Math.round(realShare * 100);
    bar.setAttribute("aria-label", prec == null ? "No alerts are raised" : `Of every 100 alerts, about ${realN} are real attacks and ${100 - realN} are false alarms`);
    barCap.textContent = prec == null ? "No alerts at these settings" : `Of every 100 alerts: about ${realN} real, ${100 - realN} false`;
    announce(`Precision ${precVal.textContent}; accuracy ${accVal.textContent}.`);
  }

  const barCap = h("div", { class: "small muted ab-bar-cap" });
  for (const r of [base, tpr, fpr]) {
    r.fill = syncRangeFill(r.input);
    r.input.addEventListener("input", update);
  }

  const card = h(
    "div",
    { class: "card ab-calc" },
    h(
      "div",
      { class: "card-head" },
      h("div", {}, h("h3", {}, "Why average precision, not accuracy"), h("div", { class: "card-sub" }, "Accuracy counts every correct call, so when attacks are rare the benign majority decides it.")),
      h("span", { class: "pill neutral", html: `${icon("info")}Arithmetic illustration, not a measurement` })
    ),
    h(
      "div",
      { class: "ab-calc-body" },
      h(
        "div",
        { class: "ab-calc-explain stack-sm" },
        h("p", {}, "A detector that never raises an alert is ", h("strong", {}, "99.99% accurate"), " when 1 window in 10,000 is an attack, and it catches nothing."),
        h(
          "p",
          {},
          h("strong", {}, "Average precision"),
          " ranks every window by its score and averages the precision each time another real attack is reached. It needs no threshold, and a random ranking scores only the attack rate, so the benign majority cannot inflate it."
        ),
        h("p", {}, h("strong", {}, "MCC"), " (Matthews correlation) is our second metric: it uses all four cells of the confusion matrix and is 0 for any constant guess."),
        h(
          "blockquote",
          { class: "ab-quote" },
          "“At 5 % prevalence a detector with ROC-AUC 0.98 and 97.4 % accuracy yields P(attack | alert) = 0.008 at a realistic 1-in-10,000 base rate.”",
          h("cite", {}, "README.md, Honest framing")
        )
      ),
      h(
        "div",
        { class: "ab-calc-panel" },
        base.field,
        presets,
        tpr.field,
        fpr.field,
        h(
          "div",
          { class: "ab-calc-out" },
          h("div", { class: "stat primary ab-calc-prec" }, h("div", { class: "stat-label" }, "Precision: P(attack | alert)"), precVal, precSub),
          h(
            "div",
            { class: "ab-calc-side" },
            h("div", { class: "ab-kv" }, h("span", { class: "k" }, "Accuracy"), accVal, h("span", { class: "tiny faint" }, "looks excellent, means little here")),
            h("div", { class: "ab-kv" }, h("span", { class: "k" }, "False alarms per real attack"), faVal),
            h("div", { class: "ab-kv" }, h("span", { class: "k" }, "A detector that never alerts"), lazyVal, h("span", { class: "tiny faint" }, "accuracy, with zero attacks caught"))
          )
        ),
        barLegend,
        bar,
        barCap,
        live
      )
    )
  );
  update();
  return card;
}

/* ------------------------------------------------------------------ reproduce */

const COMMANDS = [
  {
    group: "Set up",
    icon: "cpu",
    items: [
      { cmd: "uv sync", what: "Install the pinned Python environment (uv.lock)." },
      { cmd: "uv run pytest tests -q", what: "The whole test suite, including the build-failing leakage guards." },
    ],
  },
  {
    group: "Reproduce the real findings",
    icon: "database",
    note: "These need the AIT Log Data Set downloaded from Zenodo (record 19483937) and processed locally: derived features are never redistributed.",
    items: [
      { cmd: "uv run pytest tests/test_pipeline_real_data.py tests/test_cross_replica.py -q -v", what: "Findings 1 and 2: random-split inflation from duplicate rows, and the calendar-feature leak." },
      { cmd: "uv run pytest tests/test_loro_real.py -q -v", what: "Finding 3: leave-one-replica-out across all 8 replicas." },
      { cmd: "uv run pytest tests/test_tessera_base.py -q -v -m slow", what: "Finding 4: trains the neural model on real data and checks its attribution (slow, about 5 minutes)." },
      { cmd: "uv run pytest tests/test_e2e.py -q -v", what: "The whole pipeline in one test: ingest, features, detect, explain, ledger, inclusion proof, tamper detection, optimiser." },
      { cmd: "uv run python -m tessera.chainsim.report", what: "The two findings from evaluating candidate objectives and optimisers for sidechain segment-length tuning." },
      { cmd: "just web-data", what: "Re-export this site's aggregate statistics, real results (including the mask-only floor) and design-decision data into web/data/." },
      { cmd: "uv run pytest tests/test_web_exports.py -q", what: "Check the exported files: schema, licence scan, provenance, and (with the cache) that the shipped numbers are what the exporter computes today." },
    ],
  },
  {
    group: "Check the browser code",
    icon: "check",
    items: [
      { cmd: "cd web && npm test", what: "The JS model against real PyTorch outputs, and the JS ledger against the real Python ledger." },
      { cmd: 'node --test "web/tests/*.test.mjs"', what: "Unit tests for the Training Lab (data generator, splits, metrics, trainer, pipeline), the charts and the score-your-own-data panel. Run from the repository root." },
    ],
  },
  {
    group: "Run this site",
    icon: "play",
    items: [{ cmd: "python3 -m http.server 8743 --directory web", what: "Serve it locally from the repository root, then open http://localhost:8743. Any static host works; there is no server-side code." }],
  },
];

async function doCopy(text, btn, toast, label) {
  const ok = await copyText(text);
  if (ok) {
    btn.classList.add("is-copied");
    btn.innerHTML = icon("check");
    btn.setAttribute("aria-label", `Copied ${label}`);
    toast(`Copied: ${text.length > 60 ? `${text.slice(0, 57)}…` : text}`, { kind: "good", timeout: 1800 });
    setTimeout(() => {
      btn.classList.remove("is-copied");
      btn.innerHTML = COPY_ICON;
      btn.setAttribute("aria-label", `Copy ${label}`);
    }, 1600);
  } else {
    toast("Your browser blocked the clipboard. Select the command and press Ctrl or Cmd + C.", { kind: "warn" });
  }
}

function buildReproduce(ctx) {
  const wrap = h("div", { class: "ab-cmd-groups" });
  COMMANDS.forEach((g) => {
    const list = h("ol", { class: "ab-cmds" });
    for (const it of g.items) {
      const code = h("code", { class: "ab-cmd-code" }, it.cmd);
      const btn = h("button", { class: "icon-btn ab-copy", type: "button", "aria-label": `Copy command: ${it.cmd}`, title: "Copy", html: COPY_ICON });
      btn.addEventListener("click", () => doCopy(it.cmd, btn, ctx.toast, `command: ${it.cmd}`));
      list.appendChild(h("li", { class: "ab-cmd" }, h("div", { class: "ab-cmd-line" }, h("span", { class: "ab-prompt", "aria-hidden": "true" }, "$"), code, btn), h("p", { class: "ab-cmd-what" }, it.what)));
    }
    const all = h("button", { class: "btn sm ghost", type: "button", html: `${COPY_ICON}<span>Copy all</span>` });
    all.addEventListener("click", async () => {
      const ok = await copyText(g.items.map((i) => i.cmd).join("\n"));
      ctx.toast(ok ? `Copied ${g.items.length} command${g.items.length > 1 ? "s" : ""}` : "Your browser blocked the clipboard.", { kind: ok ? "good" : "warn", timeout: 1800 });
    });
    wrap.appendChild(
      h(
        "article",
        { class: `card ab-cmd-card${g.items.length > 3 ? " is-tall" : ""}` },
        h("div", { class: "card-head" }, h("div", { class: "row" }, h("span", { class: "ab-mc-icon", html: icon(g.icon) }), h("h3", {}, g.group)), g.items.length > 1 ? all : null),
        g.note ? h("div", { class: "notice warn ab-cmd-note" }, fromMarkup(icon("alert", { cls: "notice-icon" })), h("div", {}, g.note)) : null,
        list
      )
    );
  });
  stagger(wrap);
  return section(
    "about-reproduce",
    "Reproduce it yourself",
    "Where every number comes from",
    "Run these from the repository root. Every number on the real-data tabs traces to a recorded run or a shipped file, and says which: recomputed by the exporter, or transcribed from RESULTS.md (one-off ablations that no test re-runs are marked on the page). The generated tables are also checked by check_no_hardcoded_numbers against the run index.",
    wrap
  );
}

/* ------------------------------------------------------------------ FAQ */

/** How the full model compares with the strongest structural floor, fold by fold. */
function headlineVsFloor(D, fl) {
  const floorBlock = fl.mask_and_host?.summary?.average_precision ? fl.mask_and_host : fl;
  const name = floorBlock === fl ? "presence-only" : "presence-plus-host";
  const floorBy = Object.fromEntries((floorBlock.folds || []).map((f) => [f.held_out, f.average_precision]));
  const scored = (D.folds || []).filter((f) => !f.low_support && Number.isFinite(f.average_precision) && Number.isFinite(floorBy[f.held_out]));
  const beat = scored.filter((f) => f.average_precision > floorBy[f.held_out]).length;
  const fm = floorBlock.summary.average_precision.mean;
  if (!D.ap || !scored.length) return `the ${name} floor is ${ap3(fm)} mean AP; the full model must beat it to show it reads feature values. `;
  return `the full model's mean AP is ${Math.round((D.ap.mean - fm) * 100)} points above the ${name} floor (${ap3(D.ap.mean)} vs ${ap3(fm)}), and it beats that floor on ${beat} of ${scored.length} scoreable replicas, so it uses the feature values, not only which sources are on and which host it is. `;
}

/** "Is log-template presence just the label?": the labelling rule and the real floor it sets. */
function floorFaq(D, p, b) {
  const fl = D.floor;
  const rule = p(
    b("Partly, by construction. "),
    "A window is an attack window only if a labelled log line falls inside it. The AIT labels exist only for the four log files M1 is computed from (auth.log, audit.log, openvpn.log, dnsmasq.log); the Suricata events behind M2 and M4 carry no labels. So every attack window has M1 activity, and the availability bits, plus M3's count of present groups, carry part of the label."
  );
  if (!fl) return [rule, p("The measured floor could not be loaded (real_results.json, mask_only_floor).")];
  const sh = fl.attack_share || {};
  const m = fl.summary.average_precision;
  const host = fl.host_only?.summary?.average_precision;
  const both = fl.mask_and_host?.summary?.average_precision;
  const excl = fl.summary.excluded_low_support || [];
  const others = Number.isFinite(sh.n_attack_windows) && Number.isFinite(sh.n_attack_on_host) ? sh.n_attack_windows - sh.n_attack_on_host : null;
  return [
    rule,
    p(
      `Measured on the real cache: M1 is present in ${fmt.pct(sh.m1_present_rate_attack, 1)} of ${fmt.int(sh.n_attack_windows)} attack windows and ${fmt.pct(sh.m1_present_rate_benign, 1)} of ${fmt.int(sh.n_benign_windows)} benign ones, and ${fmt.pct(sh.share_of_attacks_on_host, 1)} of attack windows are on ${sh.host} (attack rate ${fmt.pct(sh.prevalence_on_host, 0)} there, ${fmt.pct(sh.prevalence_elsewhere, 2)} on the other two hosts).`
    ),
    p(
      `On the same leave-one-replica-out folds as the headline, a LightGBM that sees only the three presence bits, and no feature values, scores mean AP ${ap3(m.mean)} ± ${ap3(m.std)} (${m.n} folds, min ${ap3(m.min)}, max ${ap3(m.max)}${excl.length ? `; ${excl.join(", ")} excluded, as in the headline` : ""}).`,
      Number.isFinite(host?.mean) ? ` Host identity alone scores ${ap3(host.mean)} ± ${ap3(host.std)}` : "",
      Number.isFinite(both?.mean) ? `, and presence plus host ${ap3(both.mean)} ± ${ap3(both.std)}.` : ".",
      D.ap ? ` The full-feature LightGBM scores ${ap3(D.ap.mean)} ± ${ap3(D.ap.std)}.` : ""
    ),
    p(
      b("What the headline shows: "),
      headlineVsFloor(D, fl),
      b("What it does not show: "),
      `detection on the VPN or intranet hosts. Cross-replica AP is dominated by detecting the attack on the firewall host${others != null ? `; the other two hosts contribute ${fmt.int(others)} attack windows in total` : ""}. And because labels come from the logs, it cannot measure attacks that leave no trace in them.`
    ),
  ];
}

function faqItems(D) {
  const p = (...parts) => h("p", {}, ...parts);
  const b = (t) => h("strong", {}, t);
  const ul = (...items) => h("ul", { class: "ab-list" }, ...items.map((i) => h("li", {}, i)));
  const leak = D.leak;
  const cal = D.calendar?.rows;

  return [
    {
      q: "Why do you report average precision instead of accuracy?",
      a: [
        p(
          "Because attacks are rare, ",
          b("accuracy"),
          " is misleading at low attack prevalence: a detector that never fires can still score well on accuracy alone. We report average precision and MCC instead, since both stay meaningful when the positive class is small."
        ),
        p(
          leak
            ? `Split choice matters just as much as the metric: on the same AIT data, a random split scores AP ${ap3(leak.r0_random_ap)} while a chronological split of the same data scores AP ${ap3(leak.r1_chronological_ap)}. `
            : "Split choice matters just as much as the metric: a random split inflates the score relative to a chronological split of the same data. "
        ),
        p(
          "So we report average precision on splits designed to limit leakage, each with a leakage certificate, and show the random split only as a labelled upper bound rather than a headline number."
        ),
        p(
          D.ap
            ? `Where our numbers are high (mean cross-replica AP ${ap3(D.ap.mean)} for the LightGBM baseline), they are scoped to replicas of one scenario and dominated by the firewall host (see "Is log-template presence just the label?"), and the fold with too few attacks (${D.shaw?.held_out ?? "shaw"}, AP ${ap3(D.shaw?.average_precision)}) is shown rather than hidden.`
            : "Where our numbers are high, they are scoped to replicas of one scenario, and low-support folds are shown rather than hidden."
        ),
      ],
      src: "README.md (Honest framing); RESULTS.md findings 1 and 3",
    },
    {
      q: "Why is the browser demo on synthetic data?",
      a: [
        p(
          "Project policy, not a licence ban. The AIT Log Data Set is CC BY-NC-SA 4.0, which permits non-commercial redistribution under the same licence. ShareAlike would treat derived feature tables as adaptations, so the project's policy (LICENSE-DATA) is never to redistribute derived AIT features, which keeps the code permissively licensed."
        ),
        p(
          `This site ships only aggregate statistics, each over at least ${D.minSupport} real windows. The Live detector and the Training Lab generate synthetic windows from them, and everything synthetic carries a `,
          h("span", { class: "tag-synthetic" }, "Synthetic"),
          " tag."
        ),
        p("The pretrained model itself was trained on real data, and every performance claim comes from real held-out data on the Real results tab. Synthetic scores show the mechanism; they are never evidence of performance."),
      ],
      src: "LICENSE-DATA; web/README.md",
    },
    {
      q: "Is this a real blockchain?",
      a: [
        p(
          "No, and deliberately so. It is an ",
          b("RFC 6962 Merkle transparency log"),
          ", the construction Certificate Transparency uses, with Ed25519-signed tree heads and signed-checkpoint segmentation in the Python ledger. There is no mining, no consensus and no token."
        ),
        p("It guarantees exactly one thing: an auditor holding a past signed tree head can detect any retroactive modification or deletion of a retained verdict."),
        p(
          "Storing raw IPs, geolocation or request/response bodies on a shared immutable ledger turns permanence into a privacy liability — TESSERA's ledger never does. It commits an RFC 6962 leaf hash (SHA-256 over a 0x00 prefix and the canonical JSON) of six fields: window id, host_hash, time bucket, verdict, score and model version. host_hash is meant to be a salted pseudonym; the salt helper in ledger/sth.py is not yet wired in, and the leaf hash itself is unsalted. No IP, URL, username or body is included."
        ),
      ],
      src: "THREAT_MODEL.md; src/tessera/ledger/merkle.py (leaf_hash); src/tessera/data/contract.py",
    },
    {
      q: "Why do you say 'replicas', not 'organisations'?",
      a: [
        p(
          "Because the eight AIT testbeds are not independent. The Zenodo record says the attack parameters and execution order vary per dataset, while the environment and attack repertoire are shared. An early draft called them independent organisations; that was wrong, and it is recorded as correction C5."
        ),
        p("Leave-one-replica-out therefore measures robustness to that randomisation: real and useful, but not transfer to a different network or attack type, which would need scenarios AIT does not contain."),
      ],
      src: "NEGATIVE_RESULTS.md C5; RESULTS.md finding 3",
    },
    {
      q: "Why does LightGBM do as well as the neural network?",
      a: [
        p(
          D.tb
            ? `Because it should. On one held-out fold (santos), from one run, TESSERA-base scores AP ${ap4(D.tb.tessera_ap)} and a default-configured 200-tree LightGBM ${ap4(D.tb.lightgbm_ap)}: a tie, not a win. An 8-fold paired comparison is future work (NEGATIVE_RESULTS C2). `
            : "Because it should. ",
          "Gradient-boosted trees are expected to be highly competitive on tabular features, and the project said so before training anything."
        ),
        p(
          `The neural model isn't there to win a leaderboard. Its per-source encoders and availability-masked gate are what let it handle a missing feature group explicitly, and at ${D.params ? fmt.int(D.params) : "about five thousand"} parameters its forward pass is small enough to hand-write in JavaScript, where it is shown to match PyTorch to within 4.4e-7 on 40 reference vectors.`
        ),
      ],
      src: "RESULTS.md finding 4; NEGATIVE_RESULTS.md C2; src/tessera/models/baselines/gbdt.py",
    },
    {
      q: "What is average precision, and why not accuracy?",
      a: [
        p(
          "Rank every window by its score and walk down the list: ",
          b("average precision"),
          " is the mean precision you have each time you reach another real attack. It needs no threshold, and a random ranking scores only the attack rate, so the benign majority cannot inflate it. Accuracy can."
        ),
        p("README.md puts it concretely: at 5% prevalence, a detector with ROC-AUC 0.98 and 97.4% accuracy yields P(attack | alert) = 0.008 at a realistic 1-in-10,000 base rate. The calculator above lets you try other rates."),
        p(
          "MCC is our second metric: threshold-based (0.5 here) and using all four confusion-matrix cells. ",
          D.harrison
            ? `The two can disagree usefully: harrison has AP ${ap3(D.harrison.average_precision)} but MCC ${ap3(D.harrison.mcc)}, meaning the ranking is near-perfect but the scores are shifted for that replica, so a fixed cut-off misfires.`
            : ""
        ),
      ],
      src: "README.md; RESULTS.md finding 3; src/tessera/eval/metrics.py",
    },
    {
      q: "What is leakage, and how do you show you avoided it?",
      a: [
        p("Leakage is any route by which information about the test set reaches training, so a score measures memory rather than detection. We found it twice in our own pipeline:"),
        ul(
          leak
            ? `Duplicate rows: ${fmt.pct(leak.exact_duplicate_rate, 0)} of windows are exact duplicates, and a random split left ${fmt.int(leak.test_rows_identical_to_train)} test rows byte-identical to a training row (AP ${ap3(leak.r0_random_ap)} random vs ${ap3(leak.r1_chronological_ap)} chronological).`
            : "Duplicate rows: a random split left test rows byte-identical to training rows.",
          cal
            ? `Calendar features: hour-of-day let a model memorise when attacks happened in one capture. Removing them shrank the random-vs-chronological gap from ${cal[0].gap.toFixed(3)} to ${cal[1].gap.toFixed(3)}.`
            : "Calendar features: hour-of-day let a model memorise when attacks happened in one capture."
        ),
        p(
          "Guarding against it is built in, not promised: build-failing tests (test_no_group_overlap, test_scaler_train_only, test_permutation_chance); a leakage certificate under every table (exact and near duplicates, cross-split twins, group disjointness, the availability-mask shortcut floor, source-provenance AUC and a label-permutation check); and an E1 halt gate that fails when a chronological split doesn't show the expected inflation. The Training Lab checks each split before it trains."
        ),
        p("One honest wrinkle: with all four feature groups, the E1 gate's 0.02 margin was not cleared (a gap of about 0.015). It was investigated and found benign, not waved through. The splits limit leakage and measure what is left; they do not make it impossible."),
      ],
      src: "README.md (Guarantees the build enforces); RESULTS.md findings 1 and 2",
    },
    {
      q: "Is log-template presence just the label?",
      a: floorFaq(D, p, b),
      src: "real_results.json (mask_only_floor, recomputed); src/tessera/data/ait/window_builder.py; src/tessera/eval/leakage.py (mask_only_score)",
    },
    {
      q: "Why shouldn't we trust the fusion gate's attribution?",
      a: [
        p(
          D.abl?.attr
            ? `Because we checked it, and it was misleading. On held-out santos the gate's mean weights were M1 ${pct0(D.abl.attr.m1_log)}, M2 ${pct0(D.abl.attr.m2_metrics)}, M3 ${pct0(D.abl.attr.m3_identity)} and M4 ${pct0(D.abl.attr.m4_graph)}, which reads as "the model barely uses log templates".`
            : "Because we checked it, and it was misleading: it said the model barely uses log templates."
        ),
        p(
          D.abl?.m1Alone && D.abl?.withM1 && D.abl?.withoutM1
            ? `But a gradient-boosted model on log templates alone scores AP ${ap4(D.abl.m1Alone.ap)}, and removing them honestly from TESSERA-base (zeroed and marked missing) drops AP from ${ap4(D.abl.withM1.ap)} to ${ap4(D.abl.withoutM1.ap)}, while host_bucket alone is barely above the prevalence floor with MCC 0.`
            : "But ablations show log templates carry most of the signal, and removing them costs real AP."
        ),
        p("So gate values are shown as a hint, never as proof of what the model used. A per-modality ablation is the reliable measure, and a test locks the gap in so it cannot quietly disappear."),
      ],
      src: "RESULTS.md finding 4; NEGATIVE_RESULTS.md F9",
    },
    {
      q: "What does the Merkle ledger protect against, and what doesn't it?",
      a: [
        p(b("It protects against"), " retroactive edits or deletions of retained verdicts: anyone holding a past signed tree head can detect them."),
        p(b("It does not protect against:")),
        ul(
          "Split-view attacks: an operator controlling both the log and tree-head distribution can show different auditors different histories (needs gossip or an external witness; neither is implemented).",
          "Signing-key compromise: a key holder can re-sign a rewritten history for anyone who kept no tree head.",
          "Archive destruction: that yields unverifiability, not tamper evidence.",
          "Confidentiality: the log provides integrity only.",
          "Untrue records: it can't check that a record reflects reality, or stop events being omitted before they reach it.",
          "Writes themselves: edits become detectable, not impossible.",
          "Inference-time evasion, poisoning and concept drift."
        ),
      ],
      src: "THREAT_MODEL.md",
    },
    {
      q: "Why is the shaw fold kept out of the average?",
      a: [
        p(
          D.shaw
            ? `shaw has only ${fmt.int(D.shaw.n_pos_test)} attack windows out of ${fmt.int(D.shaw.n_test)} for our fixed three-host subset, all within one 44-minute episode near the end of a ${Number.isFinite(D.replicas?.find((r) => r.id === "shaw")?.span_hours) ? `${Math.round(D.replicas.find((r) => r.id === "shaw").span_hours)}-hour` : "162-hour"} capture. It was investigated: a consequence of the per-replica randomisation, not a bug.`
            : "shaw has only a handful of attack windows for our fixed three-host subset, all within one short episode. It was investigated: a consequence of the per-replica randomisation, not a bug."
        ),
        p(
          `A precision measured on so few positives is noise, so under the project's MIN_SUPPORT_FOR_RATES = ${D.minSupport} rule the fold is reported in full${D.shaw ? ` (AP ${ap3(D.shaw.average_precision)})` : ""} but kept out of the summary.`,
          D.naive && D.ap ? ` Averaging it in would read ${ap3(D.naive.mean)} ± ${ap3(D.naive.std)} instead of ${ap3(D.ap.mean)} ± ${ap3(D.ap.std)}.` : ""
        ),
      ],
      src: "RESULTS.md finding 3; NEGATIVE_RESULTS.md F8",
    },
    {
      q: "What would you do next?",
      a: [
        ul(
          D.harrison
            ? `Calibrate per replica: harrison's MCC of ${ap3(D.harrison.mcc)} points at shifted scores. A per-replica threshold, or the TPR-at-fixed-FPR metrics already in eval/metrics.py, is the next step.`
            : "Calibrate per replica: a per-replica threshold, or the TPR-at-fixed-FPR metrics already in eval/metrics.py.",
          "Cross-corpus transfer (R4, AIT to the corrected CIC-IDS2017 release): the only genuine cross-corpus evidence available, pending a licence check on that release.",
          "Run the headline paired comparison over the 8 leave-one-replica-out folds: five seeds can never reach p < 0.05 with a Wilcoxon test (minimum p = 0.0625).",
          "Close the ledger's split-view gap with gossip or an external witness.",
          "Test on attack scenarios AIT does not contain before claiming anything beyond cross-replica robustness."
        ),
      ],
      src: "RESULTS.md finding 3; NEGATIVE_RESULTS.md C2 and C5; LICENSE-DATA; THREAT_MODEL.md",
    },
  ];
}

function buildFaq(D) {
  const items = faqItems(D);
  const list = h("div", { class: "ab-faq-list" });
  const details = [];
  items.forEach((it, i) => {
    const d = h(
      "details",
      { class: "acc ab-faq" },
      h("summary", {}, h("span", { class: "ab-q" }, h("span", { class: "ab-q-num tabular", "aria-hidden": "true" }, String(i + 1).padStart(2, "0")), h("span", {}, it.q))),
      h("div", { class: "acc-body" }, ...it.a, srcNote(it.src))
    );
    d.dataset.text = `${it.q} ${d.textContent}`.toLowerCase();
    details.push(d);
    list.appendChild(d);
  });

  const input = h("input", { type: "text", id: "ab-faq-filter", placeholder: "Try: leakage, ledger, shaw", autocomplete: "off", spellcheck: "false" });
  const count = h("span", { class: "tiny faint ab-faq-count", "aria-live": "polite" }, `${items.length} questions`);
  const toggle = h("button", { class: "btn sm", type: "button" }, "Expand all");
  const empty = h("p", { class: "small faint ab-faq-empty", hidden: true }, "No question matches. Try another word.");

  input.addEventListener(
    "input",
    debounce(() => {
      const q = input.value.trim().toLowerCase();
      let shown = 0;
      for (const d of details) {
        const hit = !q || d.dataset.text.includes(q);
        d.hidden = !hit;
        if (hit) shown++;
        if (q && hit) d.open = true;
      }
      empty.hidden = shown > 0;
      count.textContent = q ? `${shown} of ${items.length} questions` : `${items.length} questions`;
      syncToggle();
    }, 120)
  );
  const syncToggle = () => {
    const visible = details.filter((d) => !d.hidden);
    toggle.textContent = visible.length && visible.every((d) => d.open) ? "Collapse all" : "Expand all";
  };
  toggle.addEventListener("click", () => {
    const visible = details.filter((d) => !d.hidden);
    const open = !visible.every((d) => d.open);
    visible.forEach((d) => (d.open = open));
    syncToggle();
  });
  details.forEach((d) => d.addEventListener("toggle", syncToggle));

  const tools = h("div", { class: "ab-faq-tools" }, h("div", { class: "field ab-faq-field" }, h("label", { for: "ab-faq-filter" }, "Filter questions"), input), h("div", { class: "row" }, count, toggle));
  const el = section("about-faq", "Viva FAQ", "Questions a panel is likely to ask", "Straight answers, each with the file that backs it.", tools, list, empty);
  el.setAttribute("data-tour", "about-faq");
  return el;
}

/* ------------------------------------------------------------------ stack + credits */

function buildStack() {
  const col = (title, iconName, items) =>
    h(
      "article",
      { class: "card ab-stack-card" },
      h("div", { class: "ab-mc-head" }, h("span", { class: "ab-mc-icon", html: icon(iconName) }), h("h3", {}, title)),
      h("ul", { class: "ab-chips" }, ...items.map(([name, what]) => h("li", { class: "ab-chip" }, h("strong", {}, name), h("span", {}, what))))
    );
  const stack = h(
    "div",
    { class: "grid-2 ab-stack-grid" },
    col("Research code (Python)", "flask", [
      ["Python 3.11 + uv", "pinned environment"],
      ["PyTorch", "the neural model, trained on MPS"],
      ["LightGBM, scikit-learn", "baselines and metrics"],
      ["Drain3", "log-template mining (M1)"],
      ["networkx", "graph features (M4)"],
      ["Hydra", "experiment configuration"],
      ["Optuna", "optimiser baseline for the segment-length comparison"],
      ["cryptography", "Ed25519 signed tree heads"],
      ["pytest + Hypothesis", "tests and property-based ledger checks"],
    ]),
    col("This website", "overview", [
      ["Vanilla ES modules", "no framework, no CDN, no npm dependencies"],
      ["js/forward.js", "hand-written forward pass, parity-tested against PyTorch"],
      ["js/merkle.js", "RFC 6962 ledger port, parity-tested against Python"],
      ["Web Crypto", "SHA-256 for the ledger"],
      ["Web Worker", "the Training Lab trains off the main thread"],
      ["node:test", "unit and parity tests"],
      ["Static hosting", "zero servers, zero cost"],
    ])
  );
  stagger(stack);

  const credits = h(
    "div",
    { class: "card ab-credits" },
    h("div", { class: "ab-mc-head" }, h("span", { class: "ab-mc-icon", html: icon("sparkles") }), h("h3", {}, "Credits and licences")),
    h(
      "dl",
      { class: "ab-dl" },
      h("dt", {}, "Dataset"),
      h(
        "dd",
        {},
        "AIT Log Data Set V2.1, Landauer et al., Zenodo record 19483937, licensed CC BY-NC-SA 4.0. The dataset authors ask users to cite: M. Landauer et al., “Maintainable Log Datasets for Evaluation of Intrusion Detection Systems”, IEEE Transactions on Dependable and Secure Computing 20(4), 3466–3482. This site contains aggregate summary statistics only, never rows, log lines or addresses (project policy, LICENSE-DATA)."
      ),
      h("dt", {}, "Code"),
      h("dd", {}, "MIT licence, © 2026 TESSERA project contributors."),
      h("dt", {}, "Constructions"),
      h("dd", {}, "RFC 6962 Merkle trees (as used by Certificate Transparency); the Drain3 log-template miner.")
    )
  );
  return section("about-stack", "Tech stack and credits", "What it's built with, and whose work it rests on", null, stack, credits);
}

/* ------------------------------------------------------------------ mount */

export async function mount(el, ctx) {
  const sections = [
    { id: "about-model-card", label: "Model card", icon: "file" },
    { id: "about-evaluate", label: "How we evaluate", icon: "split" },
    { id: "about-reproduce", label: "Reproduce", icon: "reset" },
    { id: "about-faq", label: "Viva FAQ", icon: "info" },
    { id: "about-stack", label: "Stack and credits", icon: "layers" },
  ];
  el.replaceChildren();
  const root = h("div", { class: "ab" });
  const head = buildHead(sections);
  const loading = h("div", { class: "ab-loading" }, h("div", { class: "skeleton", style: { height: "180px" } }), h("div", { class: "skeleton", style: { height: "120px" } }));
  root.append(head, loading);
  el.appendChild(root);
  stagger(root);

  const D = derive(await loadData(ctx.store));
  loading.remove();

  const evalSec = buildEvaluate(D);
  root.append(buildModelCard(D), evalSec.el, buildReproduce(ctx), buildFaq(D), buildStack());
  revealOnScroll(root);

  return {
    onHide() {
      evalSec.stop();
    },
  };
}
