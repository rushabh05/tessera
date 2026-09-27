// Pure-helper tests for web/js/ui/charts.js (no DOM needed).
// Run: node --test "web/tests/*.test.mjs"   (Node 24 does not accept a bare directory)
import test from "node:test";
import assert from "node:assert/strict";

import * as charts from "../js/ui/charts.js";

const {
  niceStep,
  niceTicks,
  ticksInDomain,
  logTicks,
  niceLogDomain,
  stepDecimals,
  formatTick,
  formatLogTick,
  formatAuto,
  formatCompact,
  linearScale,
  logScale,
  log1pScale,
  log1pTicks,
  nearestIndex,
  roundedBarPath,
  pillPath,
  arcPath,
  linePath,
  stepPath,
  resolveColor,
  textWidth,
  truncateText,
  rampMix,
  rampFill,
  supportsColorMix,
  clamp,
  lerp,
  easeOutCubic,
} = charts;

const MINUS = "−";

test("module imports under Node and exposes every chart factory", () => {
  for (const name of ["lineChart", "curveChart", "barChart", "histogram", "stackedBar", "confusionMatrix", "ring", "sparkline", "rangeChart", "dataTable", "withTableToggle", "showTooltip", "hideTooltip"]) {
    assert.equal(typeof charts[name], "function", name);
  }
  // factories need a DOM; they must fail loudly rather than half-render
  assert.throws(() => charts.lineChart(null, {}), /DOM element/);
});

test("niceStep picks 1/2/5 x 10^k", () => {
  assert.equal(niceStep(1, 5), 0.2);
  assert.equal(niceStep(10, 5), 2);
  assert.equal(niceStep(1234, 5), 200);
  assert.equal(niceStep(29, 5), 5);
  assert.equal(niceStep(0.87, 5), 0.2);
  assert.equal(niceStep(7, 7), 1);
  assert.equal(niceStep(0, 5), 1);
  assert.equal(niceStep(Number.NaN, 5), 1);
});

test("niceTicks extends the domain to round numbers", () => {
  assert.deepEqual(niceTicks(0, 0.87, 5), { min: 0, max: 1, step: 0.2, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1] });
  const t = niceTicks(0, 1234, 5);
  assert.equal(t.min, 0);
  assert.equal(t.max, 1400);
  assert.deepEqual(t.ticks, [0, 200, 400, 600, 800, 1000, 1200, 1400]);
  const neg = niceTicks(-0.13, 0.42, 5);
  assert.equal(neg.min, -0.2);
  assert.equal(neg.max, 0.5);
  assert.ok(neg.ticks.includes(0));
  assert.ok(!neg.ticks.some((v) => Object.is(v, -0)));
  // degenerate domains
  assert.deepEqual(niceTicks(0, 0).ticks.at(0), 0);
  assert.equal(niceTicks(0, 0).max, 1);
  const flat = niceTicks(5, 5);
  assert.ok(flat.min < 5 && flat.max > 5);
  assert.deepEqual(niceTicks(Number.NaN, 3).ticks, niceTicks(0, 1).ticks);
  // no floating-point junk such as 0.30000000000000004
  for (const v of niceTicks(0, 0.3, 3).ticks) assert.equal(v, Number(v.toFixed(6)));
});

test("ticksInDomain stays inside a fixed domain and respects integer axes", () => {
  assert.deepEqual(ticksInDomain(0, 1, 5), [0, 0.2, 0.4, 0.6, 0.8, 1]);
  assert.deepEqual(ticksInDomain(1, 30, 5), [5, 10, 15, 20, 25, 30]);
  assert.deepEqual(ticksInDomain(1, 3, 6, { integer: true }), [1, 2, 3]);
  assert.deepEqual(ticksInDomain(0.05, 0.95, 4), [0.2, 0.4, 0.6, 0.8]);
  assert.deepEqual(ticksInDomain(2, 2), [2]);
  // narrow axes still get at least two ticks (a lone tick cannot show the scale)
  assert.deepEqual(ticksInDomain(1, 30, 2), [10, 20, 30]);
  assert.ok(ticksInDomain(1, 30, 1).length >= 2);
  for (const v of ticksInDomain(0.05, 0.95, 4)) assert.ok(v >= 0.05 && v <= 0.95);
});

test("stepDecimals", () => {
  assert.equal(stepDecimals(1), 0);
  assert.equal(stepDecimals(500), 0);
  assert.equal(stepDecimals(0.2), 1);
  assert.equal(stepDecimals(0.05), 2);
  assert.equal(stepDecimals(0.1 + 0.2), 1); // 0.30000000000000004
  assert.equal(stepDecimals(0), 0);
});

test("logTicks and niceLogDomain", () => {
  assert.deepEqual(logTicks(1, 100), [1, 3, 10, 30, 100]);
  assert.deepEqual(logTicks(1, 10), [1, 2, 5, 10]);
  assert.deepEqual(logTicks(1e-6, 1e2, { maxTicks: 9 }), [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100]);
  const thin = logTicks(1e-6, 1e2, { maxTicks: 4 });
  assert.ok(thin.length <= 5 && thin.length >= 3, `thinned: ${thin}`);
  assert.equal(thin[0], 1e-6);
  assert.deepEqual(logTicks(0, 10), []);
  assert.deepEqual(niceLogDomain(0.03, 7), [0.01, 10]);
  assert.deepEqual(niceLogDomain(1, 1000), [1, 1000]);
});

test("formatTick: separators, step decimals, compact, unicode minus", () => {
  assert.equal(formatTick(1000, 500), "1,000");
  assert.equal(formatTick(0.2, 0.2), "0.2");
  assert.equal(formatTick(1, 0.2), "1.0");
  assert.equal(formatTick(0, 0.2), "0");
  assert.equal(formatTick(-0.4, 0.2), `${MINUS}0.4`);
  assert.equal(formatTick(15000, 5000), "15K");
  assert.equal(formatTick(2500000, 500000), "2.5M");
  assert.equal(formatTick(Number.NaN, 1), "—");
});

test("formatLogTick", () => {
  assert.equal(formatLogTick(1), "1");
  assert.equal(formatLogTick(1000), "1,000");
  assert.equal(formatLogTick(20000), "20K");
  assert.equal(formatLogTick(0.01), "0.01");
  assert.equal(formatLogTick(0.001), "0.001");
  assert.equal(formatLogTick(1e-5), "10⁻⁵");
  assert.equal(formatLogTick(2e-6), "2×10⁻⁶");
  assert.equal(formatLogTick(0), "—");
});

test("formatAuto and formatCompact", () => {
  assert.equal(formatAuto(0.81234), "0.812");
  assert.equal(formatAuto(12.345), "12.35");
  assert.equal(formatAuto(1234), "1,234");
  assert.equal(formatAuto(-0.25), `${MINUS}0.250`);
  assert.equal(formatAuto(0.00001234), "1.23e-5");
  assert.equal(formatAuto(null), "—");
  assert.equal(formatAuto(Number.POSITIVE_INFINITY), "—");
  assert.equal(formatCompact(12345), "12.3K");
  assert.equal(formatCompact(-2000000), `${MINUS}2M`);
});

test("linearScale maps and inverts", () => {
  const s = linearScale([0, 1], [10, 110]);
  assert.equal(s(0), 10);
  assert.equal(s(0.5), 60);
  assert.equal(s(1), 110);
  assert.equal(s.invert(60), 0.5);
  const flipped = linearScale([0, 10], [200, 0]); // SVG y axis
  assert.equal(flipped(10), 0);
  assert.equal(flipped.invert(100), 5);
  const flat = linearScale([3, 3], [0, 100]);
  assert.equal(flat(3), 50);
});

test("logScale and log1pScale", () => {
  const s = logScale([1, 100], [0, 200]);
  assert.equal(s(1), 0);
  assert.equal(s(10), 100);
  assert.equal(s(100), 200);
  assert.ok(Math.abs(s.invert(100) - 10) < 1e-9);
  assert.equal(s(0), 0); // non-positive pinned to the domain minimum
  const c = log1pScale([0, 99], [100, 0]);
  assert.equal(c(0), 100);
  assert.equal(c(99), 0);
  assert.ok(Math.abs(c(9) - 50) < 1e-9);
  assert.ok(Math.abs(c.invert(50) - 9) < 1e-9);
  assert.deepEqual(log1pTicks(1234), [0, 1, 10, 100, 1000]);
  assert.deepEqual(log1pTicks(0), [0, 1]);
});

test("nearestIndex", () => {
  const xs = [1, 2, 4, 8];
  assert.equal(nearestIndex(xs, 0), 0);
  assert.equal(nearestIndex(xs, 2.9), 1);
  assert.equal(nearestIndex(xs, 3.1), 2);
  assert.equal(nearestIndex(xs, 100), 3);
  assert.equal(nearestIndex([], 1), -1);
  assert.equal(nearestIndex([5], 1), 0);
});

test("roundedBarPath: rounded data end, square baseline, clamped radius", () => {
  assert.equal(roundedBarPath(0, 0, 10, 0), "");
  assert.equal(roundedBarPath(0, 0, 0, 10), "");
  // top end: starts at the square bottom-left, arcs only at the top
  const up = roundedBarPath(10, 20, 20, 50, 4, "top");
  assert.equal(up, "M10,70V24A4,4 0 0 1 14,20H26A4,4 0 0 1 30,24V70Z");
  const down = roundedBarPath(10, 20, 20, 50, 4, "bottom");
  assert.equal(down, "M10,20H30V66A4,4 0 0 1 26,70H14A4,4 0 0 1 10,66Z");
  const right = roundedBarPath(0, 0, 50, 12, 4, "right");
  assert.equal(right, "M0,0H46A4,4 0 0 1 50,4V8A4,4 0 0 1 46,12H0Z");
  const left = roundedBarPath(0, 0, 50, 12, 4, "left");
  assert.equal(left, "M50,0V12H4A4,4 0 0 1 0,8V4A4,4 0 0 1 4,0Z");
  // radius never exceeds half the bar width or the bar length
  assert.match(roundedBarPath(0, 0, 4, 50, 4, "top"), /A2,2/);
  assert.match(roundedBarPath(0, 49, 20, 1, 4, "top"), /A1,1/);
  assert.equal(roundedBarPath(0, 0, 10, 10, 0, "top"), "M0,0H10V10H0Z");
});

test("pillPath and arcPath", () => {
  assert.equal(pillPath(0, 0, 40, 10), "M5,0H35A5,5 0 0 1 35,10H5A5,5 0 0 1 5,0Z");
  assert.equal(pillPath(0, 0, 0, 10), "");
  assert.equal(arcPath(50, 50, 40, 0, 0), "");
  // quarter circle from 12 o'clock to 3 o'clock
  assert.equal(arcPath(50, 50, 40, 0, Math.PI / 2), "M50,10A40,40 0 0 1 90,50");
  // > 180 degrees sets the large-arc flag
  assert.match(arcPath(50, 50, 40, 0, Math.PI * 1.5), /A40,40 0 1 1/);
  // full circle is drawn as two arcs
  assert.equal((arcPath(50, 50, 40, 0, Math.PI * 2).match(/A/g) || []).length, 2);
});

test("linePath / stepPath build and decimate paths", () => {
  assert.equal(linePath([[0, 0], [10, 5], [20, 0]]), "M0,0L10,5L20,0");
  // sub-half-pixel neighbours are dropped, but the last point is always kept
  assert.equal(linePath([[0, 0], [0.1, 0.1], [0.2, 0.2], [10, 10]]), "M0,0L10,10");
  assert.equal(linePath([[0, 0], [Number.NaN, 3], [5, 5]]), "M0,0L5,5");
  assert.equal(linePath([]), "");
  assert.equal(stepPath([[0, 10], [5, 20], [10, 30]]), "M0,10H5V20H10V30");
});

test("resolveColor maps roles and tokens to CSS variables, never cycles", () => {
  assert.equal(resolveColor("train"), "var(--c-train)");
  assert.equal(resolveColor("validation"), "var(--c-val)");
  assert.equal(resolveColor("attack"), "var(--c-attack)");
  assert.equal(resolveColor("--series-3"), "var(--series-3)");
  assert.equal(resolveColor("series-2"), "var(--series-2)");
  assert.equal(resolveColor("var(--accent)"), "var(--accent)");
  assert.equal(resolveColor(null, 0), "var(--series-1)");
  assert.equal(resolveColor(undefined, 7), "var(--series-8)");
  assert.equal(resolveColor(undefined, 8), "var(--neutral-fill)");
  assert.equal(resolveColor(undefined, 20), "var(--neutral-fill)");
});

test("text measurement fallback and truncation", () => {
  assert.ok(textWidth("hello", 11) > 0);
  assert.ok(textWidth("a much longer label", 11) > textWidth("short", 11));
  assert.equal(truncateText("short", 1000), "short");
  const t = truncateText("russellmitchell", 40, 11);
  assert.ok(t.endsWith("…"));
  assert.ok(textWidth(t, 11) <= 40);
});

test("rampFill uses color-mix only where supported, else an opacity ramp on a token", () => {
  // Node has no CSS.supports: treated as unsupported
  assert.equal(supportsColorMix(), false);
  const mixed = rampFill(0.25, true);
  assert.equal(mixed.fill, `color-mix(in srgb, var(--series-1) ${rampMix(0.25)}%, var(--surface-2))`);
  assert.equal(mixed.fillOpacity, null);
  assert.equal(mixed.swatch, mixed.fill);
  for (const share of [0, 0.01, 0.25, 0.5, 1]) {
    const f = rampFill(share, false);
    assert.equal(f.fill, "var(--series-1)");
    assert.doesNotMatch(f.fill + f.swatch, /color-mix/);
    assert.equal(f.fillOpacity, rampMix(share) / 100);
    assert.equal(f.mix, rampMix(share));
  }
  assert.ok(rampFill(0.1, false).fillOpacity < rampFill(0.6, false).fillOpacity, "monotone");
  assert.equal(rampFill(0.3).fill, "var(--series-1)", "default follows supportsColorMix()");
});

test("confusion-matrix cells never depend on an unsupported color-mix()", async () => {
  const cellFills = (el) => el.all((n) => hasClass(n, "viz-cm-rect")).map((n) => ({ fill: n.style.fill, op: n.style.fillOpacity }));
  // (a) no CSS.supports (older Safari behaves the same: color-mix unsupported)
  let C = await loadWithFakeDom();
  try {
    const el = new FakeNode("div");
    C.confusionMatrix(el, { tp: 40, fp: 5, tn: 900, fn: 3 });
    const fills = cellFills(el);
    assert.equal(fills.length, 4);
    for (const f of fills) {
      assert.equal(f.fill, "var(--series-1)");
      assert.ok(Number(f.op) > 0 && Number(f.op) <= 1, `fill-opacity ${f.op}`);
    }
  } finally {
    delete globalThis.window;
    delete globalThis.document;
  }
  // (b) a browser that supports it gets the opaque mix and no opacity
  globalThis.CSS = { supports: (prop, val) => prop === "color" && /^color-mix\(/.test(val) };
  try {
    C = await loadWithFakeDom();
    const el = new FakeNode("div");
    C.confusionMatrix(el, { tp: 40, fp: 5, tn: 900, fn: 3 });
    for (const f of cellFills(el)) {
      assert.match(f.fill, /^color-mix\(in srgb, var\(--series-1\) \d+%, var\(--surface-2\)\)$/);
      assert.equal(f.op, "");
    }
  } finally {
    delete globalThis.CSS;
    delete globalThis.window;
    delete globalThis.document;
  }
});

test("charts.css gives confusion-matrix cells a token fill and uses no color-mix()", async () => {
  const { readFileSync } = await import("node:fs");
  const css = readFileSync(new URL("../css/charts.css", import.meta.url), "utf8");
  assert.match(css, /\.viz-cm-rect\s*\{[^}]*fill:\s*var\(--surface-2\)/);
  assert.doesNotMatch(css, /color-mix\(/);
  const js = readFileSync(new URL("../js/ui/charts.js", import.meta.url), "utf8");
  const uses = js.split("\n").filter((l) => /color-mix\(in srgb, var\(/.test(l));
  assert.equal(uses.length, 1, `only rampFill builds a color-mix: ${uses.join(" | ")}`);
});

test("rampMix is monotone and bounded", () => {
  assert.equal(rampMix(0), 10);
  assert.equal(rampMix(1), 85);
  assert.ok(rampMix(0.25) < rampMix(0.5));
  assert.equal(rampMix(-1), 10);
  assert.equal(rampMix(Number.NaN), 10);
});

test("small numeric helpers", () => {
  assert.equal(clamp(5, 0, 1), 1);
  assert.equal(clamp(-5, 0, 1), 0);
  assert.equal(lerp(0, 10, 0.25), 2.5);
  assert.equal(easeOutCubic(0), 0);
  assert.equal(easeOutCubic(1), 1);
  assert.ok(easeOutCubic(0.5) > 0.5);
});

/* ------------------------------------------------------------------ degenerate-domain helpers */

const {
  autoDomain,
  nextNiceStep,
  tickBudget,
  thinLabels,
  pickTickFormat,
  tickStep,
  finiteValues,
  extent,
  validDomain,
} = charts;

const distinct = (arr) => new Set(arr).size === arr.length;

test("nextNiceStep walks the 1-2-5 sequence", () => {
  assert.deepEqual([0.01, 0.02, 0.05, 0.1, 1, 2, 5, 10, 500].map(nextNiceStep), [0.02, 0.05, 0.1, 0.2, 2, 5, 10, 20, 1000]);
  assert.equal(nextNiceStep(0), 1);
  assert.equal(nextNiceStep(Number.NaN), 1);
});

test("tickBudget: about one tick per 36 px, never fewer than two", () => {
  assert.equal(tickBudget(156), 5);
  assert.equal(tickBudget(36), 2);
  assert.equal(tickBudget(10), 2);
  assert.equal(tickBudget(0), 2);
  assert.equal(tickBudget(Number.NaN), 2);
  assert.equal(tickBudget(360, 60), 7);
});

test("niceTicks: integer and maxTicks options", () => {
  // count axes: a max count of 1 must not produce 0, 0.2, ... or 0, 0.5, 1
  assert.deepEqual(niceTicks(0, 1, 5, { integer: true }).ticks, [0, 1]);
  assert.deepEqual(niceTicks(0, 3, 6, { integer: true }).ticks, [0, 1, 2, 3]);
  for (const t of niceTicks(0, 7, 8, { integer: true }).ticks) assert.ok(Number.isInteger(t));
  // the cap wins over the requested count
  const capped = niceTicks(0, 1, 20, { maxTicks: 3 });
  assert.ok(capped.ticks.length <= 3, String(capped.ticks));
  assert.equal(capped.min, 0);
  assert.equal(capped.max, 1);
  // default behaviour is unchanged
  assert.deepEqual(niceTicks(0, 0.87, 5).ticks, [0, 0.2, 0.4, 0.6, 0.8, 1]);
});

test("ticksInDomain: maxTicks cap and integer rounding", () => {
  assert.deepEqual(ticksInDomain(0, 1, 6, { maxTicks: 3 }), [0, 0.5, 1]);
  assert.ok(ticksInDomain(0, 1, 50, { maxTicks: 4 }).length <= 4);
  assert.deepEqual(ticksInDomain(1, 2, 5, { integer: true }), [1, 2]);
  assert.ok(ticksInDomain(1, 30, 30, { integer: true, maxTicks: 6 }).every(Number.isInteger));
});

test("autoDomain: one point / near-flat AP never yields a hairline axis (the '0.9999...' tick stack)", () => {
  for (const vals of [[0.99993], [0.9999, 1], [0.99990, 0.99995], [1, 1, 1]]) {
    const d = autoDomain(vals, { count: 3, maxTicks: tickBudget(156) });
    assert.ok(d.max - d.min >= 0.04, `span too small for ${vals}: ${d.min}..${d.max}`);
    assert.ok(d.max <= 1, `probability data padded above 1: ${d.max}`);
    assert.ok(d.min <= Math.min(...vals) && d.max >= Math.max(...vals));
    assert.ok(d.ticks.length >= 2 && d.ticks.length <= 5, String(d.ticks));
    assert.ok(stepDecimals(d.step) <= 2, `too many decimals: step ${d.step}`);
    // every label is distinct with the Lab's own toFixed(3) formatter
    assert.ok(distinct(d.ticks.map((v) => v.toFixed(3))), String(d.ticks));
  }
  const mid = autoDomain([0.5], { count: 3, maxTicks: 5 });
  assert.ok(mid.min < 0.5 && mid.max > 0.5);
});

test("autoDomain: empty, all-zero, NaN/null and negative inputs", () => {
  assert.deepEqual([autoDomain([]).min, autoDomain([]).max], [0, 1]);
  assert.deepEqual([autoDomain(null).min, autoDomain(null).max], [0, 1]);
  const zeros = autoDomain([0, 0, 0]);
  assert.equal(zeros.min, 0);
  assert.equal(zeros.max, 1);
  // null / NaN are skipped, not read as zero
  const withGaps = autoDomain([null, Number.NaN, undefined, 0.4, Number.POSITIVE_INFINITY], { count: 3, maxTicks: 5 });
  assert.ok(withGaps.min > 0.3 && withGaps.max < 0.5, `${withGaps.min}..${withGaps.max}`);
  // non-negative data is never padded below zero
  assert.ok(autoDomain([0.01], { count: 3 }).min >= 0);
  assert.ok(autoDomain([0, 0.001], { count: 3 }).min >= 0);
  // negative data is allowed and padded symmetrically
  const neg = autoDomain([-3], { count: 3, maxTicks: 5 });
  assert.ok(neg.min < -3 && neg.max > -3 && neg.max < 0);
  // loss above 1 is not clamped to [0, 1]
  const loss = autoDomain([2.3, 2.3], { count: 3, maxTicks: 5 });
  assert.ok(loss.min < 2.3 && loss.max > 2.3);
  // an explicit bound is respected even after nice extension
  const b = autoDomain([0.3, 0.45], { bounds: [0, 0.5], count: 3 });
  assert.ok(b.max <= 0.5 && b.min >= 0);
  // 'none' lets padding cross zero
  assert.ok(autoDomain([0], { bounds: "none" }).min < 0);
});

test("autoDomain: integer count axes", () => {
  const opts = { count: 4, maxTicks: 5, includeZero: true, integer: true };
  assert.deepEqual(autoDomain([0, 1], opts).ticks, [0, 1]);
  assert.deepEqual(autoDomain([0, 0], opts).ticks, [0, 1]);
  assert.deepEqual(autoDomain([2], opts).ticks, [0, 1, 2]);
  assert.deepEqual(autoDomain([0, 3], opts).ticks, [0, 1, 2, 3]);
  for (const n of [1, 2, 3, 5, 7, 11, 19, 37, 101]) {
    const d = autoDomain([0, n], opts);
    assert.ok(d.ticks.every(Number.isInteger), `${n}: ${d.ticks}`);
    assert.ok(d.ticks.length <= 5, `${n}: ${d.ticks}`);
    assert.ok(d.max >= n);
  }
});

test("autoDomain: tick count respects the pixel budget", () => {
  for (const px of [40, 80, 120, 156, 240, 400]) {
    for (const vals of [[0, 1], [0.93, 0.97], [0, 12345], [-5, 5]]) {
      const d = autoDomain(vals, { count: 10, maxTicks: tickBudget(px) });
      assert.ok(d.ticks.length <= tickBudget(px), `${px}px ${vals}: ${d.ticks}`);
    }
  }
});

test("thinLabels drops overlapping labels, keeps order", () => {
  assert.deepEqual(thinLabels([0, 5, 10, 40, 41, 80], 10, 3), [0, 3, 5]);
  assert.deepEqual(thinLabels([0, 50, 100], 11, 3), [0, 1, 2]);
  // widths per label (x axis)
  assert.deepEqual(thinLabels([0, 30, 60], [50, 50, 50], 4), [0, 2]);
  // unsorted input (a y axis runs top-down) and NaN positions
  assert.deepEqual(thinLabels([100, 0, Number.NaN, 50], 11, 3), [0, 1, 3]);
  assert.deepEqual(thinLabels([], 11), []);
  // every tick of a collapsed axis (mid-morph) stacks at one point: only one survives
  assert.equal(thinLabels([12, 12.2, 12.4, 12.6, 12.8], 11, 3).length, 1);
});

test("pickTickFormat falls back to step decimals when the caller's format repeats a label", () => {
  const fixed3 = (v) => v.toFixed(3);
  assert.equal(pickTickFormat(fixed3, [0.9, 0.95, 1], 0.05), fixed3);
  const f = pickTickFormat(fixed3, [0.9999, 0.99995, 1], 0.00005);
  assert.notEqual(f, fixed3);
  assert.ok(distinct([0.9999, 0.99995, 1].map(f)));
  assert.equal(pickTickFormat(null, [0, 0.5, 1], 0.5)(0.5), "0.5");
});

test("small domain utilities", () => {
  assert.deepEqual(finiteValues([1, null, Number.NaN, "2", undefined, 3, Number.NEGATIVE_INFINITY]), [1, 3]);
  assert.deepEqual(finiteValues(null), []);
  assert.deepEqual(extent([3, null, -1, Number.NaN, 7]), [-1, 7]);
  assert.equal(extent([]), null);
  assert.equal(extent([Number.NaN, null]), null);
  // long arrays do not blow the stack (Math.min(...arr) would)
  assert.deepEqual(extent(Float64Array.from({ length: 300000 }, (_, i) => i)), [0, 299999]);
  assert.deepEqual(validDomain([1, 0]), [0, 1]);
  assert.equal(validDomain([1, 1]), null);
  assert.equal(validDomain([0, Number.NaN]), null);
  assert.equal(validDomain(null), null);
  assert.equal(tickStep([0.1, 0.2, 0.3]), 0.1);
  assert.equal(tickStep([1]), undefined);
});

test("niceLogDomain survives zeros, negatives and a single value", () => {
  assert.deepEqual(niceLogDomain(0, 10), [1, 10]);
  assert.deepEqual(niceLogDomain(-5, 100), [10, 100]);
  assert.deepEqual(niceLogDomain(Number.NaN, Number.NaN), [1, 10]);
  assert.deepEqual(niceLogDomain(0, 0), [1, 10]);
  assert.deepEqual(niceLogDomain(5, 5), [1, 10]);
  assert.deepEqual(niceLogDomain(10, 10), [1, 10]);
  assert.deepEqual(niceLogDomain(7, 0.03), [0.01, 10]);
  assert.deepEqual(logTicks(0, 0), []);
  assert.deepEqual(logScale([0, 100], [0, 100])(0), 0); // zero pinned to the low end, not NaN
});

/* ------------------------------------------------------------------ factories under a minimal fake DOM */

class FakeNode {
  constructor(tag) {
    this.tagName = tag;
    this.attrs = new Map();
    this.children = [];
    this.parent = null;
    this._text = "";
    this.style = { setProperty() {} };
    this.hidden = false;
    const cls = new Set();
    this.classList = {
      add: (...c) => c.forEach((x) => cls.add(x)),
      remove: (...c) => c.forEach((x) => cls.delete(x)),
      toggle: (c, on) => ((on ?? !cls.has(c)) ? cls.add(c) : cls.delete(c)),
      contains: (c) => cls.has(c),
    };
  }
  set className(v) {
    this.setAttribute("class", v);
  }
  get className() {
    return this.getAttribute("class") || "";
  }
  set textContent(v) {
    this.children = [];
    this._text = String(v);
  }
  get textContent() {
    return this._text + this.children.map((c) => c.textContent).join("");
  }
  set innerHTML(v) {
    this.children = [];
    this._text = String(v);
  }
  setAttribute(k, v) {
    this.attrs.set(k, String(v));
  }
  getAttribute(k) {
    return this.attrs.has(k) ? this.attrs.get(k) : null;
  }
  appendChild(c) {
    if (c.parent) c.parent.children = c.parent.children.filter((x) => x !== c);
    c.parent = this;
    this.children.push(c);
    return c;
  }
  append(...cs) {
    for (const c of cs) this.appendChild(typeof c === "string" ? Object.assign(new FakeNode("#text"), { _text: c }) : c);
  }
  replaceChildren(...cs) {
    this.children = [];
    this._text = "";
    this.append(...cs);
  }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((x) => x !== this);
    this.parent = null;
  }
  addEventListener() {}
  removeEventListener() {}
  focus() {}
  getBoundingClientRect() {
    return { width: 360, height: 200, left: 0, top: 0, right: 360, bottom: 200 };
  }
  get offsetWidth() {
    return 360;
  }
  querySelector(sel) {
    for (const c of this.children) {
      if ((sel.startsWith("#") && c.getAttribute("id") === sel.slice(1)) || c.tagName === sel) return c;
      const hit = c.querySelector(sel);
      if (hit) return hit;
    }
    return null;
  }
  all(pred, out = []) {
    for (const c of this.children) {
      if (pred(c)) out.push(c);
      c.all(pred, out);
    }
    return out;
  }
}

async function loadWithFakeDom() {
  const body = new FakeNode("body");
  globalThis.window = { innerWidth: 1024, innerHeight: 768, addEventListener() {}, removeEventListener() {} };
  globalThis.document = {
    body,
    createElement: (t) => new FakeNode(t),
    createElementNS: (_ns, t) => new FakeNode(t),
    createTextNode: (s) => Object.assign(new FakeNode("#text"), { _text: String(s) }),
  };
  // a fresh module instance that sees the DOM (the static import above did not)
  return import(`../js/ui/charts.js?fake-dom=${Date.now()}`);
}

const hasClass = (n, c) => (n.getAttribute("class") || "").split(/\s+/).includes(c);
const yTickLabels = (el) => el.all((n) => n.tagName === "text" && hasClass(n, "viz-tick") && n.getAttribute("text-anchor") === "end").map((n) => n.textContent);
const xTickLabels = (el) => el.all((n) => n.tagName === "text" && hasClass(n, "viz-tick") && n.getAttribute("text-anchor") === "middle").map((n) => n.textContent);
const noNaN = (el) => el.all((n) => [...n.attrs.values()].some((v) => /NaN|Infinity/.test(v)));

test("chart factories render degenerate inputs without NaN, stacked or fractional count ticks", async () => {
  const C = await loadWithFakeDom();
  try {
    // (1) the Lab's live AP chart after its first epoch
    const el = new FakeNode("div");
    const ap = C.lineChart(el, { series: [{ id: "ap", label: "Validation AP", color: "val", points: [{ x: 1, y: 0.99993 }] }], xDomain: [1, 20], height: 210, xLabel: "Epoch", yLabel: "AP", yFormat: (v) => v.toFixed(3) });
    let ys = yTickLabels(el);
    assert.ok(ys.length >= 2 && ys.length <= C.tickBudget(210 - 12 - 42), `y ticks: ${ys}`);
    assert.ok(distinct(ys), `duplicate y ticks: ${ys}`);
    assert.ok(ys.every((s) => !/9999/.test(s)), `hairline domain: ${ys}`);
    assert.deepEqual(noNaN(el), []);
    // x ticks on an integer (epoch) axis are integers
    assert.ok(xTickLabels(el).every((s) => /^\d+$/.test(s)), `x ticks: ${xTickLabels(el)}`);
    // second, near-identical epoch
    ap.update({ series: [{ id: "ap", label: "Validation AP", color: "val", points: [{ x: 1, y: 0.9999 }, { x: 2, y: 1 }] }] });
    ys = yTickLabels(el);
    assert.ok(distinct(ys) && ys.length <= 5, `after update: ${ys}`);
    // empty, NaN/null-only, and a flat zero series
    for (const points of [[], [{ x: 1, y: Number.NaN }, { x: null, y: 2 }, null], [{ x: 1, y: 0 }, { x: 2, y: 0 }]]) {
      const e2 = new FakeNode("div");
      C.lineChart(e2, { series: [{ id: "s", points }] });
      assert.deepEqual(noNaN(e2), [], JSON.stringify(points));
      assert.ok(distinct(yTickLabels(e2)));
    }

    // (2) histogram with tiny counts: integer ticks only
    const h = new FakeNode("div");
    C.histogram(h, { edges: [0, 0.25, 0.5, 0.75, 1], series: [{ label: "Benign", counts: [1, 0, 0, 0] }, { label: "Attack", counts: [0, null, Number.NaN, -3] }] });
    const hy = yTickLabels(h);
    assert.ok(hy.length >= 2 && hy.every((s) => /^\d+$/.test(s)), `histogram ticks: ${hy}`);
    assert.deepEqual(noNaN(h), []);
    // histogram with broken edges shows its empty state instead of NaN geometry
    const h2 = new FakeNode("div");
    C.histogram(h2, { edges: [0, Number.NaN, 1], series: [{ label: "x", counts: [1, 2] }] });
    assert.deepEqual(noNaN(h2), []);
    assert.ok(h2.all((n) => hasClass(n, "viz-empty")).length === 1);

    // (3) bar chart of counts: integer ticks; all zeros; nulls; very long labels
    const b = new FakeNode("div");
    C.barChart(b, { bars: [{ label: "a", value: 1 }, { label: "b", value: 0 }, { label: "c", value: 2 }] });
    assert.ok(yTickLabels(b).every((s) => /^\d+$/.test(s)), `bar ticks: ${yTickLabels(b)}`);
    const b0 = new FakeNode("div");
    C.barChart(b0, { bars: [{ label: "a", value: 0 }, { label: "b", value: null }, { label: "c", value: Number.NaN }] });
    assert.deepEqual(noNaN(b0), []);
    assert.ok(distinct(yTickLabels(b0)));
    const long = "an extremely long replica name that would never fit beside its bar ".repeat(3).trim();
    const bh = new FakeNode("div");
    C.barChart(bh, { horizontal: true, bars: [{ label: long, value: 0.8 }, { label: "short", value: 0.4 }], yDomain: [0, 1] });
    const cat = bh.all((n) => n.tagName === "text" && hasClass(n, "viz-cat"))[0];
    assert.ok(cat._text.endsWith("…") && cat._text.length < long.length, cat._text);
    assert.equal(cat.children.find((c) => c.tagName === "title")?.textContent, long);
    assert.ok(bh.all((n) => (n.getAttribute("aria-label") || "").startsWith(long)).length === 1, "full label in aria-label");
    // a value outside a fixed domain is drawn to the frame, never beyond it
    const bo = new FakeNode("div");
    C.barChart(bo, { bars: [{ label: "x", value: 1.7 }, { label: "y", value: -0.2 }], yDomain: [0, 1] });
    assert.deepEqual(noNaN(bo), []);

    // (4) legend: long series names carry the full text in a title
    const lg = new FakeNode("div");
    C.lineChart(lg, { series: [{ id: "a", label: long, points: [{ x: 0, y: 1 }] }, { id: "b", label: "b", points: [{ x: 0, y: 2 }] }] });
    assert.ok(lg.all((n) => n.title === long).length === 1);

    // (5) remaining factories with empty / degenerate data
    const fns = [
      () => C.curveChart(new FakeNode("div"), { kind: "pr", series: [{ label: "one point", x: [0.5], y: [0.5] }, { label: "empty", x: [], y: [] }], baseline: 0.2 }),
      () => C.curveChart(new FakeNode("div"), { kind: "roc", series: [] }),
      () => C.stackedBar(new FakeNode("div"), { segments: [{ label: "a", value: 0 }, { label: "b", value: -2 }, { label: "c", value: Number.NaN }] }),
      () => C.confusionMatrix(new FakeNode("div"), { tp: -1, fp: Number.NaN, tn: null, fn: 0, labels: { pos: long, neg: long } }),
      () => C.ring(new FakeNode("div"), { value: Number.NaN }),
      () => C.sparkline(new FakeNode("div"), { values: [] }),
      () => C.sparkline(new FakeNode("div"), { values: [null, 3, Number.NaN, 3] }),
      () => C.rangeChart(new FakeNode("div"), { log: true, intervals: [], markers: [] }),
      () => C.rangeChart(new FakeNode("div"), { log: true, intervals: [{ label: "z", from: 0, to: 100 }], markers: [{ label: "zero", value: 0 }] }),
      () => C.rangeChart(new FakeNode("div"), { log: true, domain: [0, 1000], intervals: [{ label: "z", from: 2, to: 100 }] }),
      () => C.rangeChart(new FakeNode("div"), { intervals: [{ label: "flat", from: 5, to: 5 }] }),
    ];
    for (const fn of fns) {
      const chart = fn();
      assert.equal(typeof chart.update, "function");
      chart.destroy();
    }
    for (const mk of [
      (e) => C.curveChart(e, { kind: "pr", series: [{ label: "p", x: [0.5], y: [0.5] }] }),
      (e) => C.confusionMatrix(e, { tp: -1, fp: Number.NaN, tn: null, fn: 0, labels: { pos: long, neg: long } }),
      (e) => C.sparkline(e, { values: [null, 3, Number.NaN, 3] }),
      (e) => C.rangeChart(e, { log: true, intervals: [{ label: "z", from: 0, to: 100 }], markers: [{ label: "zero", value: 0 }] }),
      (e) => C.rangeChart(e, { log: true, intervals: [], markers: [] }),
    ]) {
      const e = new FakeNode("div");
      mk(e);
      assert.deepEqual(noNaN(e).map((n) => n.tagName), [], mk.toString());
    }
  } finally {
    delete globalThis.window;
    delete globalThis.document;
  }
});
