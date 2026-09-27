// TESSERA chart library: small, dependency-free, responsive SVG charts.
//
// Every factory is (el, opts) -> {update(opts), destroy()}. update() MERGES the
// new options into the current ones and animates from what is on screen now
// (never from zero). Colours are CSS variables only (role names such as
// "train" / "attack", token names such as "--series-2", or "var(...)"), so a
// theme switch restyles every chart without a re-render. All motion is skipped
// under prefers-reduced-motion.
//
// The pure helpers (ticks, scales, formatters, path builders) are named exports
// and are safe to import under Node: nothing at module top level touches the DOM.

import { reducedMotion as domReducedMotion, debounce } from "./dom.js";

const HAS_DOM = typeof document !== "undefined" && typeof window !== "undefined";
const SVG_NS = "http://www.w3.org/2000/svg";
const MINUS = "−";
const noop = () => {};

/* ================================================================== pure helpers */

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
export const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

const r2 = (v) => Math.round(v * 100) / 100;
const crisp = (v) => Math.round(v) + 0.5;

/** Number of decimals needed to print `step` exactly (0.2 -> 1, 0.05 -> 2, 500 -> 0). */
export function stepDecimals(step) {
  if (!(step > 0) || !Number.isFinite(step)) return 0;
  for (let d = 0; d <= 12; d++) {
    const p = Math.pow(10, d);
    if (Math.abs(step * p - Math.round(step * p)) < 1e-6) return d;
  }
  return 12;
}

function roundTo(v, d) {
  const r = Number(v.toFixed(Math.min(20, Math.max(0, d))));
  return r === 0 ? 0 : r; // no -0
}

/** A "nice" step (1, 2 or 5 x 10^k) that splits `span` into about `count` parts. */
export function niceStep(span, count = 5) {
  if (!(span > 0) || !Number.isFinite(span)) return 1;
  const raw = span / Math.max(1, count);
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / p;
  // geometric midpoints between 1, 2, 5, 10 -> the tick count lands nearest the request
  const nf = f < Math.SQRT2 ? 1 : f < Math.sqrt(10) ? 2 : f < Math.sqrt(50) ? 5 : 10;
  return nf * p;
}

/** The next larger step in the 1-2-5 sequence: 0.02 -> 0.05, 0.05 -> 0.1, 1 -> 2. */
export function nextNiceStep(step) {
  if (!(step > 0) || !Number.isFinite(step)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(step) + 1e-9));
  const f = Math.round((step / p) * 1e6) / 1e6;
  const nf = f < 2 ? 2 : f < 5 ? 5 : 10;
  return Number((nf * p).toPrecision(12));
}

/**
 * Most ticks an axis of `px` pixels should carry: about one per `per` px
 * (36 px keeps 11 px labels well apart), never fewer than two.
 */
export function tickBudget(px, per = 36) {
  if (!(px > 0) || !Number.isFinite(px)) return 2;
  return Math.max(2, Math.floor(px / Math.max(1, per)) + 1);
}

/**
 * Nice ticks that EXTEND the domain outwards to round numbers.
 * niceTicks(0, 0.87) -> {min: 0, max: 1, step: 0.2, ticks: [0, 0.2, ..., 1]}
 * integer: steps are whole numbers (count axes). maxTicks: the step grows along
 * 1-2-5 until the tick count fits (a hard cap from the axis length).
 */
export function niceTicks(min, max, count = 5, { integer = false, maxTicks = Infinity } = {}) {
  let a = Number(min);
  let b = Number(max);
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    a = 0;
    b = 1;
  }
  if (a > b) [a, b] = [b, a];
  if (a === b) {
    if (a === 0) b = 1;
    else {
      const pad = Math.max(Math.abs(a) * 0.5, integer ? 1 : 0);
      a -= pad;
      b += pad;
    }
  }
  let step = niceStep(b - a, count);
  if (integer) step = Math.max(1, Math.round(step));
  const cap = Math.max(2, Number.isFinite(maxTicks) ? Math.floor(maxTicks) : Infinity);
  let lo;
  let hi;
  let n;
  const span = (s) => {
    lo = Math.floor(a / s + 1e-9) * s;
    hi = Math.ceil(b / s - 1e-9) * s;
    if (hi <= lo) hi = lo + s;
    n = Math.max(1, Math.round((hi - lo) / s));
  };
  span(step);
  for (let guard = 0; n + 1 > cap && guard < 40; guard++) {
    step = nextNiceStep(step);
    span(step);
  }
  const d = stepDecimals(step);
  const ticks = [];
  for (let i = 0; i <= n; i++) ticks.push(roundTo(lo + i * step, d));
  return { min: roundTo(lo, d), max: roundTo(hi, d), step, ticks };
}

/** Nice ticks INSIDE a fixed domain (the domain is not extended). */
export function ticksInDomain(min, max, count = 5, { integer = false, maxTicks = Infinity } = {}) {
  let a = Number(min);
  let b = Number(max);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return [];
  if (a > b) [a, b] = [b, a];
  if (a === b) return [a];
  const stepFor = (c) => {
    const s = niceStep(b - a, c);
    return integer ? Math.max(1, Math.round(s)) : s;
  };
  const fromStep = (step) => {
    const d = stepDecimals(step);
    const first = Math.ceil(a / step - 1e-9);
    const last = Math.floor(b / step + 1e-9);
    const out = [];
    for (let i = first; i <= last && out.length < 200; i++) out.push(roundTo(i * step, d));
    return out;
  };
  let step = stepFor(count);
  let out = fromStep(step);
  for (let c = count + 1; out.length < 2 && c <= count + 6; c++) {
    step = stepFor(c);
    out = fromStep(step);
  }
  const cap = Math.max(2, Number.isFinite(maxTicks) ? Math.floor(maxTicks) : Infinity);
  for (let guard = 0; out.length > cap && guard < 40; guard++) {
    step = nextNiceStep(step);
    const next = fromStep(step);
    if (next.length < 2) break; // a lone tick cannot show the scale
    out = next;
  }
  return out;
}

/**
 * Finite numbers only: null, undefined, NaN, +-Infinity and non-numbers are skipped
 * (Number(null) === 0 would otherwise plot a missing value as zero).
 */
export function finiteValues(values) {
  const out = [];
  if (!values) return out;
  for (const v of values) if (typeof v === "number" && Number.isFinite(v)) out.push(v);
  return out;
}

/** [min, max] of the finite values (loop, not spread: safe for long arrays), or null when there are none. */
export function extent(values) {
  let lo = Infinity;
  let hi = -Infinity;
  if (values) {
    for (const v of values) {
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return lo <= hi ? [lo, hi] : null;
}

/** A caller-supplied [min, max] domain, sorted, or null when it is unusable (missing, NaN, zero span). */
export function validDomain(d) {
  if (!d || d.length < 2) return null;
  const a = Number(d[0]);
  const b = Number(d[1]);
  if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) return null;
  return a < b ? [a, b] : [b, a];
}

/**
 * Automatic axis domain + ticks for a set of values, robust to degenerate data.
 *  - no finite values -> [0, 1]
 *  - a zero or near-zero span (one point, a flat line, 0.99990 vs 0.99995) is
 *    padded to at least minSpanFrac x |value| (or minSpan); all zeros -> [0, 1]
 *  - bounds 'auto': padding never pushes non-negative data below 0, and never
 *    pushes probability-like data (all in [0, 1]) above 1; [lo, hi] = explicit
 *    bounds; 'none' = unbounded
 *  - integer: whole-number ticks (count axes); maxTicks caps the tick count
 * Returns {min, max, step, ticks} like niceTicks.
 */
export function autoDomain(values, { count = 5, maxTicks = Infinity, includeZero = false, integer = false, minSpanFrac = 0.05, minSpan = 0, bounds = "auto" } = {}) {
  const ext = extent(values);
  let lo = ext ? ext[0] : 0;
  let hi = ext ? ext[1] : 1;
  let bLo = -Infinity;
  let bHi = Infinity;
  if (Array.isArray(bounds)) {
    if (Number.isFinite(bounds[0])) bLo = bounds[0];
    if (Number.isFinite(bounds[1])) bHi = bounds[1];
  } else if (bounds === "auto") {
    if (lo >= 0) bLo = 0;
    if (lo >= 0 && hi <= 1) bHi = 1;
  }
  if (includeZero) {
    lo = Math.min(lo, 0);
    hi = Math.max(hi, 0);
  }
  const mag = Math.max(Math.abs(lo), Math.abs(hi));
  let need = Math.max(Number(minSpan) || 0, mag > 0 ? mag * minSpanFrac : 1);
  if (integer) need = Math.max(need, 1);
  if (bHi - bLo < need) need = bHi - bLo;
  if (hi - lo < need) {
    const c = (lo + hi) / 2;
    lo = c - need / 2;
    hi = c + need / 2;
    if (lo < bLo) {
      hi += bLo - lo;
      lo = bLo;
    }
    if (hi > bHi) {
      lo = Math.max(bLo, lo - (hi - bHi));
      hi = bHi;
    }
  }
  const nt = niceTicks(lo, hi, count, { integer, maxTicks });
  if (nt.min >= bLo - 1e-12 && nt.max <= bHi + 1e-12) return nt;
  // the nice extension crossed an explicit bound: keep the bound, ticks inside it
  const min = Math.max(nt.min, bLo);
  const max = Math.min(nt.max, bHi);
  const ticks = ticksInDomain(min, max, count, { integer, maxTicks });
  return { min, max, step: ticks.length > 1 ? ticks[1] - ticks[0] : nt.step, ticks };
}

/**
 * Collision thinning for axis labels. centers: pixel positions; sizes: label
 * extent along the axis (a number or one per label). Returns the indices to
 * keep (ascending) so no two kept labels come closer than `gap` px.
 */
export function thinLabels(centers, sizes, gap = 4) {
  const order = [];
  for (let i = 0; i < (centers || []).length; i++) if (Number.isFinite(centers[i])) order.push(i);
  order.sort((a, b) => centers[a] - centers[b]);
  const keep = [];
  let lastEnd = -Infinity;
  for (const i of order) {
    const s = typeof sizes === "number" ? sizes : Number(sizes?.[i]) || 0;
    if (!keep.length || centers[i] - s / 2 >= lastEnd + gap) {
      keep.push(i);
      lastEnd = centers[i] + s / 2;
    }
  }
  return keep.sort((a, b) => a - b);
}

/**
 * The tick formatter to use: the caller's, unless it prints two ticks the same
 * (e.g. toFixed(3) on a 0.0001 step), in which case decimals come from the step.
 */
export function pickTickFormat(userFmt, ticks, step) {
  const fallback = (v) => formatTick(v, step);
  if (typeof userFmt !== "function") return fallback;
  const seen = new Set();
  for (const t of ticks || []) {
    const s = String(userFmt(t));
    if (seen.has(s)) return fallback;
    seen.add(s);
  }
  return userFmt;
}

/** Distance between the first two ticks (the step), or undefined for fewer than two. */
export function tickStep(ticks) {
  return ticks && ticks.length > 1 ? Number((ticks[1] - ticks[0]).toPrecision(12)) : undefined;
}

/** Log-scale ticks: 1-2-5 per decade for narrow ranges, thinned powers of ten for wide ones. */
export function logTicks(min, max, { maxTicks = 8 } = {}) {
  let a = Number(min);
  let b = Number(max);
  if (!(a > 0) || !(b > 0)) return [];
  if (a > b) [a, b] = [b, a];
  const e0 = Math.floor(Math.log10(a) + 1e-9);
  const e1 = Math.ceil(Math.log10(b) - 1e-9);
  const decades = Math.max(0, e1 - e0);
  const mult = decades <= 1 ? [1, 2, 5] : decades <= 2 ? [1, 3] : [1];
  const stride = mult.length === 1 ? Math.max(1, Math.ceil((decades + 1) / maxTicks)) : 1;
  const out = [];
  for (let e = e0; e <= e1; e++) {
    if (mult.length === 1 && (e - e0) % stride !== 0) continue;
    for (const m of mult) {
      const v = Number((m * Math.pow(10, e)).toPrecision(12));
      if (v >= a * (1 - 1e-9) && v <= b * (1 + 1e-9)) out.push(v);
    }
  }
  return out;
}

/**
 * Extend a positive range to whole decades: [0.03, 7] -> [0.01, 10].
 * Zeros, negatives and NaN cannot sit on a log axis: a non-positive min falls
 * back to one decade below max, and no usable values at all give [1, 10]. A
 * single value spans its own decade ([5, 5] -> [1, 10]; [10, 10] -> [1, 10]).
 */
export function niceLogDomain(min, max) {
  let a = Number(min);
  let b = Number(max);
  const okA = a > 0 && Number.isFinite(a);
  const okB = b > 0 && Number.isFinite(b);
  if (!okA && !okB) return [1, 10];
  if (!okB) b = a;
  if (!okA) a = b / 10;
  if (a > b) [a, b] = [b, a];
  let lo = Math.pow(10, Math.floor(Math.log10(a) + 1e-9));
  let hi = Math.pow(10, Math.ceil(Math.log10(b) - 1e-9));
  if (!(hi > lo)) lo = hi / 10;
  lo = Number(lo.toPrecision(12));
  hi = Number(hi.toPrecision(12));
  return [lo, hi];
}

export function formatCompact(v) {
  if (!Number.isFinite(v)) return "—";
  const s = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(Math.abs(v));
  return (v < 0 ? MINUS : "") + s;
}

/** Axis tick label: thousands separators, decimals from the tick step, compact from 10,000. */
export function formatTick(v, step) {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) < 1e-12) return "0";
  const a = Math.abs(v);
  let s;
  if (a >= 1e4) s = formatCompact(a);
  else {
    const d = step != null ? stepDecimals(step) : a >= 100 ? 0 : a >= 1 ? 2 : 3;
    s = a.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  return (v < 0 ? MINUS : "") + s;
}

const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const superscript = (n) => String(n).split("").map((c) => SUP[c] ?? c).join("");

/** Log-axis tick label: 0.01, 1, 1,000, 12K, 10⁻⁵, 2×10⁻⁶. */
export function formatLogTick(v) {
  if (!(v > 0) || !Number.isFinite(v)) return "—";
  if (v >= 1e4) return formatCompact(v);
  if (v >= 1) return Number(v.toPrecision(6)).toLocaleString("en-US");
  if (v >= 0.001) return String(Number(v.toPrecision(3)));
  const e = Math.floor(Math.log10(v) + 1e-9);
  const m = Math.round(v / Math.pow(10, e));
  return m === 1 ? `10${superscript(e)}` : `${m}×10${superscript(e)}`;
}

/** Readable value for tooltips and direct labels when no formatter is supplied. */
export function formatAuto(v) {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  let s;
  if (Number.isInteger(v)) s = a >= 1e6 ? formatCompact(a) : a.toLocaleString("en-US");
  else if (a >= 1e4) s = formatCompact(a);
  else if (a >= 100) s = a.toLocaleString("en-US", { maximumFractionDigits: 1 });
  else if (a >= 1) s = a.toFixed(2);
  else if (a >= 0.001) s = a.toFixed(3);
  else s = a.toExponential(2);
  return (v < 0 ? MINUS : "") + s;
}

export function linearScale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;
  const f = (v) => (span === 0 ? (r0 + r1) / 2 : r0 + ((v - d0) / span) * (r1 - r0));
  f.invert = (p) => (r1 === r0 ? d0 : d0 + ((p - r0) / (r1 - r0)) * span);
  f.domain = [d0, d1];
  f.range = [r0, r1];
  return f;
}

/** Base-10 log scale; values <= 0 are pinned to the lower end of the domain. */
export function logScale(domain, range) {
  const lo = domain[0] > 0 ? domain[0] : 1e-12;
  const hi = domain[1] > lo ? domain[1] : lo * 10;
  const lin = linearScale([Math.log10(lo), Math.log10(hi)], range);
  const f = (v) => lin(Math.log10(v > 0 ? v : lo));
  f.invert = (p) => Math.pow(10, lin.invert(p));
  f.domain = [lo, hi];
  f.range = [...range];
  return f;
}

/** Count scale for zero-inflated histograms: y = log10(1 + count). */
export function log1pScale(domain, range) {
  const lin = linearScale([Math.log10(1 + Math.max(0, domain[0])), Math.log10(1 + Math.max(0, domain[1]))], range);
  const f = (v) => lin(Math.log10(1 + Math.max(0, v)));
  f.invert = (p) => Math.pow(10, lin.invert(p)) - 1;
  f.domain = [...domain];
  f.range = [...range];
  return f;
}

/** Ticks for a log1p count axis: 0, 1, 10, 100, ... up to max. */
export function log1pTicks(max) {
  const out = [0];
  for (let v = 1; v <= Math.max(1, max) * 1.0000001 && out.length < 12; v *= 10) out.push(v);
  return out;
}

/** Index of the value in a sorted array nearest to x. */
export function nearestIndex(sorted, x) {
  const n = sorted.length;
  if (n === 0) return -1;
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] <= x) lo = mid;
    else hi = mid;
  }
  return Math.abs(sorted[lo] - x) <= Math.abs(sorted[hi] - x) ? lo : hi;
}

/**
 * Bar path with a rounded DATA end and a square baseline. (x, y) is the
 * top-left of the bar's box; `end` is the data end: top | bottom | right | left.
 */
export function roundedBarPath(x, y, w, h, r = 4, end = "top") {
  if (!(w > 0) || !(h > 0)) return "";
  const vertical = end === "top" || end === "bottom";
  const rr = r2(Math.max(0, Math.min(r, vertical ? w / 2 : h / 2, vertical ? h : w)));
  const X0 = r2(x);
  const X1 = r2(x + w);
  const Y0 = r2(y);
  const Y1 = r2(y + h);
  if (rr === 0) return `M${X0},${Y0}H${X1}V${Y1}H${X0}Z`;
  const A = `A${rr},${rr} 0 0 1`;
  switch (end) {
    case "bottom":
      return `M${X0},${Y0}H${X1}V${r2(Y1 - rr)}${A} ${r2(X1 - rr)},${Y1}H${r2(X0 + rr)}${A} ${X0},${r2(Y1 - rr)}Z`;
    case "right":
      return `M${X0},${Y0}H${r2(X1 - rr)}${A} ${X1},${r2(Y0 + rr)}V${r2(Y1 - rr)}${A} ${r2(X1 - rr)},${Y1}H${X0}Z`;
    case "left":
      return `M${X1},${Y0}V${Y1}H${r2(X0 + rr)}${A} ${X0},${r2(Y1 - rr)}V${r2(Y0 + rr)}${A} ${r2(X0 + rr)},${Y0}Z`;
    default:
      return `M${X0},${Y1}V${r2(Y0 + rr)}${A} ${r2(X0 + rr)},${Y0}H${r2(X1 - rr)}${A} ${X1},${r2(Y0 + rr)}V${Y1}Z`;
  }
}

/** Pill (both ends rounded) - for range bars, which have no baseline. */
export function pillPath(x, y, w, h) {
  if (!(w > 0) || !(h > 0)) return "";
  const rr = r2(Math.min(w, h) / 2);
  const X0 = r2(x);
  const X1 = r2(x + w);
  const Y0 = r2(y);
  const Y1 = r2(y + h);
  return `M${r2(X0 + rr)},${Y0}H${r2(X1 - rr)}A${rr},${rr} 0 0 1 ${r2(X1 - rr)},${Y1}H${r2(X0 + rr)}A${rr},${rr} 0 0 1 ${r2(X0 + rr)},${Y0}Z`;
}

/** Circular arc; angles in radians, 0 = 12 o'clock, clockwise. */
export function arcPath(cx, cy, r, a0, a1) {
  const span = a1 - a0;
  if (!(span > 1e-6)) return "";
  const pt = (a) => `${r2(cx + r * Math.sin(a))},${r2(cy - r * Math.cos(a))}`;
  if (span >= Math.PI * 2 - 1e-6) {
    return `M${pt(a0)}A${r},${r} 0 1 1 ${pt(a0 + Math.PI)}A${r},${r} 0 1 1 ${pt(a0 + Math.PI * 2 - 1e-4)}`;
  }
  return `M${pt(a0)}A${r},${r} 0 ${span > Math.PI ? 1 : 0} 1 ${pt(a1)}`;
}

/** Polyline through pixel points [[x, y], ...], dropping sub-half-pixel steps. */
export function linePath(pts) {
  let d = "";
  let lx = NaN;
  let ly = NaN;
  for (let i = 0; i < pts.length; i++) {
    const [x, y] = pts[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (d && i < pts.length - 1 && Math.abs(x - lx) < 0.5 && Math.abs(y - ly) < 0.5) continue;
    d += `${d ? "L" : "M"}${r2(x)},${r2(y)}`;
    lx = x;
    ly = y;
  }
  return d;
}

/** "steps-post" path (matplotlib drawstyle used by sklearn's PR display). */
export function stepPath(pts) {
  let d = "";
  let lx = NaN;
  let ly = NaN;
  for (let i = 0; i < pts.length; i++) {
    const [x, y] = pts[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (!d) {
      d = `M${r2(x)},${r2(y)}`;
    } else {
      if (i < pts.length - 1 && Math.abs(x - lx) < 0.5 && Math.abs(y - ly) < 0.5) continue;
      d += `H${r2(x)}V${r2(y)}`;
    }
    lx = x;
    ly = y;
  }
  return d;
}

const ROLE_COLORS = {
  train: "--c-train",
  val: "--c-val",
  validation: "--c-val",
  test: "--c-test",
  unused: "--c-unused",
  benign: "--c-benign",
  attack: "--c-attack",
  accent: "--accent",
  neutral: "--neutral-fill",
};

/**
 * Colour expression for a mark. Accepts a role name ("train", "attack"), a
 * token ("--series-3", "series-3"), or any CSS expression ("var(--x)").
 * Missing colours take --series-(i+1) in fixed order; past 8 -> neutral (never cycled).
 */
export function resolveColor(c, i = 0) {
  if (c == null || c === "") return i < 8 ? `var(--series-${i + 1})` : "var(--neutral-fill)";
  const str = String(c).trim();
  if (str.startsWith("var(")) return str;
  if (str.startsWith("--")) return `var(${str})`;
  if (ROLE_COLORS[str]) return `var(${ROLE_COLORS[str]})`;
  const m = /^series-?([1-8])$/.exec(str);
  if (m) return `var(--series-${m[1]})`;
  return str;
}

let measureCtx = null;
/** Rendered text width in px (canvas when available, a character estimate otherwise). */
export function textWidth(str, size = 11, weight = 400) {
  const s = String(str ?? "");
  if (HAS_DOM) {
    try {
      if (measureCtx === null) measureCtx = document.createElement("canvas").getContext("2d") || false;
      if (measureCtx) {
        measureCtx.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
        return measureCtx.measureText(s).width;
      }
    } catch {
      measureCtx = false;
    }
  }
  return s.length * size * 0.56;
}

/** Shorten text with an ellipsis so it fits in maxPx. */
export function truncateText(str, maxPx, size = 11, weight = 400) {
  const s = String(str ?? "");
  if (textWidth(s, size, weight) <= maxPx) return s;
  let lo = 0;
  let hi = s.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (textWidth(s.slice(0, mid) + "…", size, weight) <= maxPx) lo = mid;
    else hi = mid - 1;
  }
  return lo <= 0 ? "…" : s.slice(0, lo).trimEnd() + "…";
}

/** Share of total -> mix percentage for the confusion-matrix single-hue ramp. */
export function rampMix(share) {
  const s = clamp(Number(share) || 0, 0, 1);
  return Math.round(10 + 75 * Math.sqrt(s));
}

let colorMixSupport = null;
/** True when the browser understands CSS color-mix() (Safari >= 16.2, Chrome
 *  >= 111, Firefox >= 113). Cached; false without a CSS.supports (e.g. Node). */
export function supportsColorMix() {
  if (colorMixSupport === null) {
    try {
      colorMixSupport = typeof CSS !== "undefined" && typeof CSS.supports === "function" && CSS.supports("color", "color-mix(in srgb, red, blue)");
    } catch {
      colorMixSupport = false;
    }
  }
  return colorMixSupport;
}

/**
 * rampFill(share, colorMix = supportsColorMix()) -> {fill, fillOpacity, swatch}
 * for the confusion-matrix ramp. With color-mix: an opaque mix of --series-1
 * into --surface-2 (fillOpacity null). Without it (older Safari would otherwise
 * drop the value and paint the cell black): plain --series-1 at an opacity equal
 * to the same mix share, which reads the same over the card surface. `swatch` is
 * the matching colour for a tooltip key.
 */
export function rampFill(share, colorMix = supportsColorMix()) {
  const mix = rampMix(share);
  if (colorMix) {
    const c = `color-mix(in srgb, var(--series-1) ${mix}%, var(--surface-2))`;
    return { mix, fill: c, fillOpacity: null, swatch: c };
  }
  return { mix, fill: "var(--series-1)", fillOpacity: +(mix / 100).toFixed(2), swatch: "var(--series-1)" };
}

/* ================================================================== DOM plumbing */

function reducedMotion() {
  return HAS_DOM && domReducedMotion();
}

function sv(tag, attrs, parent) {
  const e = document.createElementNS(SVG_NS, tag);
  if (attrs) {
    for (const k in attrs) {
      const v = attrs[k];
      if (v === undefined || v === null || v === false) continue;
      if (k === "text") e.textContent = v;
      else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
      else e.setAttribute(k, String(v));
    }
  }
  if (parent) parent.appendChild(e);
  return e;
}

function hd(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

/** rAF tween: frame(t) with LINEAR t in [0, 1]. Returns stop(). Instant under reduced motion. */
function tween(duration, frame, done) {
  if (!HAS_DOM || reducedMotion() || !(duration > 0) || typeof requestAnimationFrame !== "function") {
    frame(1);
    if (done) done();
    return noop;
  }
  let raf = 0;
  let t0 = -1;
  let alive = true;
  const step = (now) => {
    if (!alive) return;
    if (t0 < 0) t0 = now;
    const t = Math.min(1, (now - t0) / duration);
    frame(t);
    if (t < 1) raf = requestAnimationFrame(step);
    else {
      alive = false;
      if (done) done();
    }
  };
  raf = requestAnimationFrame(step);
  return () => {
    if (alive) {
      alive = false;
      cancelAnimationFrame(raf);
    }
  };
}

/* ---------------------------------------------------------- shared tooltip */

let tipEl = null;
let tipOwner = null;

function tipNode() {
  if (!tipEl) {
    tipEl = hd("div", "viz-tooltip");
    tipEl.setAttribute("aria-hidden", "true");
    document.body.appendChild(tipEl);
  }
  return tipEl;
}

/**
 * Show the one shared tooltip. content = {title, rows: [{label, value, color?, shape?: 'line'|'rect'|'dot'}], note}.
 * All text goes in via textContent (labels are data).
 */
export function showTooltip(content, clientX, clientY, owner = null) {
  if (!HAS_DOM) return;
  const tip = tipNode();
  tipOwner = owner;
  tip.replaceChildren();
  if (content.title) tip.appendChild(hd("div", "tt-title", content.title));
  for (const row of content.rows || []) {
    const r = hd("div", "tt-row");
    const k = hd("span", "k");
    if (row.color) {
      const key = hd("span", `tt-key ${row.shape || "line"}`);
      key.style.background = row.color;
      k.appendChild(key);
    }
    k.appendChild(document.createTextNode(row.label ?? ""));
    r.append(k, hd("span", "v", row.value ?? ""));
    tip.appendChild(r);
  }
  if (content.note) tip.appendChild(hd("div", "tt-note", content.note));
  const rect = tip.getBoundingClientRect();
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  let left = clientX + 14;
  let top = clientY - rect.height - 12;
  if (left + rect.width > vw - 8) left = clientX - rect.width - 14;
  if (left < 8) left = Math.max(8, Math.min(vw - rect.width - 8, clientX - rect.width / 2));
  if (top < 8) top = clientY + 18;
  if (top + rect.height > vh - 8) top = Math.max(8, vh - rect.height - 8);
  tip.style.left = `${Math.round(left)}px`;
  tip.style.top = `${Math.round(top)}px`;
  tip.classList.add("is-visible");
}

export function hideTooltip(owner = null) {
  if (!tipEl) return;
  if (owner && tipOwner && owner !== tipOwner) return;
  tipEl.classList.remove("is-visible");
  tipOwner = null;
}

/** Wire hover + focus + touch tooltips on a mark. getContent() -> tooltip content. */
function bindMarkTooltip(node, owner, getContent, onEnter, onLeave) {
  let touchTimer = 0;
  const showAtPointer = (e) => {
    const c = getContent();
    if (!c) return;
    showTooltip(c, e.clientX, e.clientY, owner);
    if (onEnter) onEnter();
  };
  node.addEventListener("pointermove", showAtPointer);
  node.addEventListener("pointerdown", (e) => {
    if (e.pointerType !== "mouse") {
      showAtPointer(e);
      clearTimeout(touchTimer);
      touchTimer = setTimeout(() => {
        hideTooltip(owner);
        if (onLeave) onLeave();
      }, 2600);
    }
  });
  node.addEventListener("pointerleave", (e) => {
    if (e.pointerType === "mouse") {
      hideTooltip(owner);
      if (onLeave) onLeave();
    }
  });
  node.addEventListener("focus", () => {
    const c = getContent();
    if (!c) return;
    const r = node.getBoundingClientRect();
    showTooltip(c, r.left + r.width / 2, r.top, owner);
    if (onEnter) onEnter();
  });
  node.addEventListener("blur", () => {
    hideTooltip(owner);
    if (onLeave) onLeave();
  });
}

/* ---------------------------------------------------------- legend */

function renderLegend(container, items, show = items.length >= 2) {
  container.replaceChildren();
  container.hidden = !show || items.length === 0;
  if (container.hidden) return;
  for (const it of items) {
    const item = hd("span", "item");
    const key = hd("span", `viz-key ${it.shape || "rect"}`);
    if (it.shape !== "hatch") key.style.background = it.color;
    else key.style.setProperty("--key-color", it.color);
    const label = String(it.label ?? "");
    item.append(key, hd("span", "viz-legend-label", label));
    const hasValue = it.value != null && it.value !== "";
    if (hasValue) item.appendChild(hd("span", "viz-legend-value", it.value));
    // long labels are ellipsised by CSS; the full text stays available on hover
    item.title = hasValue ? `${label} ${it.value}` : label;
    container.appendChild(item);
  }
}

/* ---------------------------------------------------------- shell + sizing */

let uid = 0;

function createShell(el, opts, { cls = "", legend = "top" } = {}) {
  if (!HAS_DOM || !el) throw new Error("charts.js: a DOM element is required");
  el.classList.add("viz");
  for (const c of cls.split(/\s+/).filter(Boolean)) el.classList.add(c);
  el.replaceChildren();
  const id = `viz${++uid}`;
  const legendEl = hd("div", "viz-legend");
  legendEl.hidden = true;
  const svg = sv("svg", { role: "img", xmlns: SVG_NS });
  const title = sv("title", null, svg);
  const root = sv("g", null, svg);
  if (legend === "top") el.append(legendEl, svg);
  else el.append(svg, legendEl);
  const shell = {
    el,
    svg,
    root,
    legendEl,
    id,
    width: 0,
    height: 0,
    setAria(label) {
      const t = label || "Chart";
      svg.setAttribute("aria-label", t);
      title.textContent = t;
    },
    size(w, h) {
      shell.width = w;
      shell.height = h;
      svg.setAttribute("width", String(w));
      svg.setAttribute("height", String(h));
      svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    },
    destroyBase() {
      hideTooltip(shell);
      el.replaceChildren();
      el.classList.remove("viz");
      for (const c of cls.split(/\s+/).filter(Boolean)) el.classList.remove(c);
    },
  };
  shell.setAria(opts.ariaLabel);
  return shell;
}

/**
 * Measures the container, renders when it first gets a width (a chart mounted
 * in a hidden tab waits until the tab is shown), re-renders on resize
 * (debounced 100 ms) and after update(). render(width, mode) with mode
 * 'intro' (first visible render), 'morph' (after update) or 'snap' (resize).
 */
function sizer(el, render) {
  let width = 0;
  let rendered = false;
  let dirty = false;
  let dead = false;
  const measure = () => Math.floor(el.getBoundingClientRect().width);
  const run = () => {
    if (dead) return;
    const w = measure();
    if (w <= 0) return;
    if (w === width && !dirty) return;
    const mode = !rendered ? "intro" : dirty ? "morph" : "snap";
    width = w;
    dirty = false;
    rendered = true;
    render(w, mode);
  };
  const later = debounce(run, 100);
  let ro = null;
  const onWin = () => later();
  if (typeof ResizeObserver === "function") {
    // first render deferred one frame: rendering inside the RO callback resizes the
    // observed element and triggers "ResizeObserver loop" warnings
    ro = new ResizeObserver(() => (rendered ? later() : requestAnimationFrame(run)));
    ro.observe(el);
  } else window.addEventListener("resize", onWin);
  run();
  return {
    invalidate() {
      dirty = true;
      run();
    },
    destroy() {
      dead = true;
      if (ro) ro.disconnect();
      else window.removeEventListener("resize", onWin);
    },
  };
}

function axisLeftWidth(labels, extra = 0) {
  let w = 0;
  for (const l of labels) w = Math.max(w, textWidth(l, 11));
  return Math.ceil(w) + 10 + extra;
}

/**
 * SVG text shortened to maxPx with an ellipsis; when shortened, the full text
 * goes in a <title> child (the native tooltip). Labels are data: textContent only.
 */
function fitText(attrs, parent, full, maxPx, size = 11, weight = 400) {
  const s = String(full ?? "");
  const shown = Number.isFinite(maxPx) ? truncateText(s, Math.max(0, maxPx), size, weight) : s;
  const t = sv("text", { ...attrs, text: shown }, parent);
  if (shown !== s) sv("title", { text: s }, t);
  return t;
}

/** Y ticks inside the plot, thinned so labels never stack (they can, mid-morph, when the domain shrinks). */
function drawYAxis(g, { ticks, fmt, y, m, plotW, plotH }) {
  const vis = [];
  for (const t of ticks) {
    const py = y(t);
    if (!Number.isFinite(py) || py < m.t - 1 || py > m.t + plotH + 1) continue;
    vis.push({ t, py });
  }
  for (const i of thinLabels(vis.map((v) => v.py), 11, 3)) {
    const { t, py } = vis[i];
    sv("line", { class: "viz-grid", x1: m.l, x2: m.l + plotW, y1: crisp(py), y2: crisp(py) }, g);
    sv("text", { class: "viz-tick", x: m.l - 8, y: r2(py), dy: "0.32em", "text-anchor": "end", text: fmt(t) }, g);
  }
}

/** X tick labels along the bottom, thinned by measured width so neighbours never overlap. */
function drawXTicks(g, { ticks, fmt, x, m, plotW, y, grid = null }) {
  const vis = [];
  for (const t of ticks) {
    const px = x(t);
    if (!Number.isFinite(px) || px < m.l - 1 || px > m.l + plotW + 1) continue;
    const text = fmt(t);
    vis.push({ t, px, text, w: textWidth(text, 11) });
  }
  for (const i of thinLabels(vis.map((v) => v.px), vis.map((v) => v.w), 8)) {
    const { t, px, text } = vis[i];
    if (grid && grid.when(t)) sv("line", { class: "viz-grid", x1: crisp(px), x2: crisp(px), y1: grid.y1, y2: grid.y2 }, g);
    sv("text", { class: "viz-tick", x: r2(px), y, "text-anchor": "middle", text }, g);
  }
}

function drawAxisLabels(g, { xLabel, yLabel, m, plotW, plotH, H }) {
  if (xLabel) sv("text", { class: "viz-axis-label", x: r2(m.l + plotW / 2), y: H - 4, "text-anchor": "middle", text: xLabel }, g);
  if (yLabel) {
    sv("text", { class: "viz-axis-label", transform: `translate(11 ${r2(m.t + plotH / 2)}) rotate(-90)`, "text-anchor": "middle", text: yLabel }, g);
  }
}

function edgeAnchor(px, lo, hi, pad = 36) {
  if (px > hi - pad) return "end";
  if (px < lo + pad) return "start";
  return "middle";
}

/* ================================================================== lineChart */

/**
 * lineChart(el, {series:[{id,label,color,points:[{x,y}]}], xLabel, yLabel, xDomain?, yDomain?,
 *   xFormat?, yFormat?, height=220, markers?:[{x,label}], refLines?:[{y,label}], area?, endLabels=true,
 *   emptyText, ariaLabel})
 * update() with more points animates the extension (live training curves).
 */
export function lineChart(el, opts = {}) {
  let o = { height: 220, endLabels: true, emptyText: "Waiting for data…", ...opts };
  const shell = createShell(el, o, { cls: "viz-line" });
  let L = null;
  let refs = null;
  let key = "";
  let shown = null;
  let stop = noop;
  let hoverX = null;
  let touchTimer = 0;
  const prevFirst = new Map();

  function layout(W) {
    const H = o.height;
    const series = (o.series || []).map((s, i) => {
      const points = (s.points || [])
        .filter((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y))
        .slice()
        .sort((a, b) => a.x - b.x);
      return { id: String(s.id ?? i), label: s.label ?? `Series ${i + 1}`, color: resolveColor(s.color, i), points, xs: points.map((p) => p.x) };
    });
    const allX = [];
    const allY = [];
    for (const s of series) for (const p of s.points) {
      allX.push(p.x);
      allY.push(p.y);
    }
    for (const r of o.refLines || []) if (Number.isFinite(r.y)) allY.push(r.y);
    const hasData = allX.length > 0;
    const xExt = extent(allX);
    let xDom = validDomain(o.xDomain) || (xExt ? xExt.slice() : [0, 1]);
    if (xDom[0] === xDom[1]) xDom = [xDom[0] - 0.5, xDom[0] + 0.5];
    const m = { t: 12 + ((o.markers || []).some((mk) => mk.label) ? 14 : 0), r: 12, b: 24 + (o.xLabel ? 18 : 0), l: 0 };
    const plotH = Math.max(40, H - m.t - m.b);
    const yCount = Math.max(2, Math.round(plotH / 46));
    const yMaxTicks = tickBudget(plotH);
    let yDom = validDomain(o.yDomain);
    let yTicks;
    let yStep;
    if (yDom) {
      yTicks = ticksInDomain(yDom[0], yDom[1], yCount, { maxTicks: yMaxTicks });
      yStep = tickStep(yTicks);
    } else {
      // one point or a flat line gets a padded span (never a 0.00001-wide axis);
      // padding keeps non-negative data >= 0 and probability-like data <= 1
      const nt = autoDomain(allY, { count: yCount, maxTicks: yMaxTicks });
      yDom = [nt.min, nt.max];
      yTicks = nt.ticks;
      yStep = nt.step;
    }
    const yTickFmt = pickTickFormat(o.yFormat, yTicks, yStep);
    const yValFmt = o.yFormat || formatAuto;
    m.l = axisLeftWidth(yTicks.map(yTickFmt), o.yLabel ? 18 : 0);
    const endLabelsOn = o.endLabels !== false && series.length <= 3 && hasData;
    if (endLabelsOn) {
      let w = 0;
      for (const s of series) if (s.points.length) w = Math.max(w, textWidth(yValFmt(s.points[s.points.length - 1].y), 11.5, 600));
      m.r = Math.max(m.r, Math.ceil(w) + 16);
    }
    const plotW = Math.max(40, W - m.l - m.r);
    // integer x (epochs, iterations): whole-number ticks, also while the chart is still empty
    const integerX = hasData ? allX.every((v) => Number.isInteger(v)) : !!validDomain(o.xDomain) && o.xDomain.every((v) => Number.isInteger(Number(v)));
    const xTicks = ticksInDomain(xDom[0], xDom[1], Math.max(2, Math.floor(plotW / 84)), { integer: integerX, maxTicks: tickBudget(plotW, 56) });
    const xStep = tickStep(xTicks);
    const xTickFmt = o.xFormat || ((v) => (xStep ? formatTick(v, xStep) : formatAuto(v)));
    const xValFmt = o.xFormat || formatAuto;
    const xs = [...new Set(allX)].sort((a, b) => a - b);
    // target scales exist before the first paint, so input events between render() and the next frame are safe
    const sx = linearScale(xDom, [m.l, m.l + plotW]);
    const sy = linearScale(yDom, [m.t + plotH, m.t]);
    return { W, H, m, plotW, plotH, series, xDom, yDom, xTicks, yTicks, yTickFmt, yValFmt, xTickFmt, xValFmt, hasData, xs, endLabelsOn, sx, sy };
  }

  function build() {
    shell.root.replaceChildren();
    const g = shell.root;
    refs = {
      axis: sv("g", { class: "viz-axis-g" }, g),
      areas: [],
      lines: [],
      dots: [],
      labels: [],
    };
    const clipId = `${shell.id}-plot`;
    refs.clip = sv("rect", null, sv("clipPath", { id: clipId }, sv("defs", null, g)));
    const areasG = sv("g", { "clip-path": `url(#${clipId})` }, g);
    const linesG = sv("g", { "clip-path": `url(#${clipId})` }, g);
    refs.endG = sv("g", null, g); // not clipped: the end labels sit in the right margin
    for (const s of L.series) {
      refs.areas.push(o.area ? sv("path", { class: "viz-area", style: { fill: s.color } }, areasG) : null);
      refs.lines.push(sv("path", { class: "viz-line-path", style: { stroke: s.color } }, linesG));
      refs.dots.push(sv("circle", { class: "viz-dot", r: 4, style: { fill: s.color } }, refs.endG));
      refs.labels.push(sv("text", { class: "viz-direct-label", dy: "0.32em" }, refs.endG));
    }
    refs.hover = sv("g", { class: "viz-hover" }, g);
    refs.empty = sv("text", { class: "viz-empty", "text-anchor": "middle" }, g);
    refs.overlay = sv("rect", { class: "viz-overlay" }, g);
    refs.overlay.addEventListener("pointermove", onPointer);
    refs.overlay.addEventListener("pointerdown", onPointer);
    refs.overlay.addEventListener("pointerleave", (e) => {
      if (e.pointerType === "mouse") clearHover();
    });
  }

  function visiblePts(s, c) {
    const n = s.points.length;
    if (!(c < n)) return s.points;
    const k = Math.floor(Math.max(0, c));
    const f = Math.max(0, c) - k;
    const out = s.points.slice(0, k);
    if (f > 0 && k < n) {
      if (k === 0) out.push(s.points[0]);
      else {
        const a = s.points[k - 1];
        const b = s.points[k];
        out.push({ x: lerp(a.x, b.x, f), y: lerp(a.y, b.y, f) });
      }
    }
    return out;
  }

  function paint(st) {
    const { m, plotW, plotH, W, H } = L;
    shell.size(W, H);
    const x = linearScale(st.xDom, [m.l, m.l + plotW]);
    const y = linearScale(st.yDom, [m.t + plotH, m.t]);
    L.sx = x;
    L.sy = y;
    const g = refs.axis;
    g.replaceChildren();
    drawYAxis(g, { ticks: L.yTicks, fmt: L.yTickFmt, y, m, plotW, plotH });
    sv("line", { class: "viz-axis", x1: m.l, x2: m.l + plotW, y1: crisp(m.t + plotH), y2: crisp(m.t + plotH) }, g);
    drawXTicks(g, { ticks: L.xTicks, fmt: L.xTickFmt, x, m, plotW, y: m.t + plotH + 16 });
    drawAxisLabels(g, { xLabel: o.xLabel, yLabel: o.yLabel, m, plotW, plotH, H });
    for (const mk of o.markers || []) {
      if (!Number.isFinite(mk.x)) continue;
      const px = x(mk.x);
      if (px < m.l - 1 || px > m.l + plotW + 1) continue;
      sv("line", { class: "viz-marker-line", x1: crisp(px), x2: crisp(px), y1: m.t, y2: m.t + plotH }, g);
      if (mk.label) fitText({ class: "viz-ref-label", x: r2(px), y: m.t - 6, "text-anchor": edgeAnchor(px, m.l, m.l + plotW) }, g, mk.label, Math.max(60, plotW * 0.45), 10.5, 560);
    }
    for (const rl of o.refLines || []) {
      if (!Number.isFinite(rl.y)) continue;
      const py = y(rl.y);
      if (py < m.t - 1 || py > m.t + plotH + 1) continue;
      sv("line", { class: "viz-ref", x1: m.l, x2: m.l + plotW, y1: crisp(py), y2: crisp(py) }, g);
      if (rl.label) fitText({ class: "viz-ref-label", x: m.l + plotW, y: r2(py - 6), "text-anchor": "end" }, g, rl.label, plotW - 4, 10.5, 560);
    }
    if (refs.clip) {
      // lines never spill outside the plot when a fixed domain is narrower than the data
      const pad = 6;
      Object.entries({ x: m.l - pad, y: m.t - pad, width: plotW + pad * 2, height: plotH + pad * 2 }).forEach(([k, v]) => refs.clip.setAttribute(k, r2(v)));
    }
    const baseY = y(clamp(0, Math.min(...st.yDom), Math.max(...st.yDom)));
    const ends = [];
    L.series.forEach((s, i) => {
      const pts = visiblePts(s, st.counts[s.id] ?? s.points.length);
      const pp = pts.map((p) => [x(p.x), y(p.y)]);
      refs.lines[i].setAttribute("d", linePath(pp));
      if (refs.areas[i]) {
        refs.areas[i].setAttribute(
          "d",
          pp.length > 1 ? `${linePath(pp)}L${r2(pp[pp.length - 1][0])},${r2(baseY)}L${r2(pp[0][0])},${r2(baseY)}Z` : ""
        );
      }
      const dot = refs.dots[i];
      const lab = refs.labels[i];
      if (pp.length) {
        const [px, py] = pp[pp.length - 1];
        dot.setAttribute("cx", r2(px));
        dot.setAttribute("cy", r2(py));
        dot.style.display = "";
        ends.push({ i, px, py, text: L.yValFmt(pts[pts.length - 1].y) });
      } else dot.style.display = "none";
      lab.style.display = "none";
    });
    if (L.endLabelsOn) {
      ends.sort((a, b) => a.py - b.py);
      let lastY = -Infinity;
      for (const e of ends) {
        if (e.py - lastY < 13) continue;
        lastY = e.py;
        const lab = refs.labels[e.i];
        lab.setAttribute("x", r2(e.px + 8));
        lab.setAttribute("y", r2(e.py));
        lab.textContent = e.text;
        lab.style.display = "";
      }
    }
    refs.empty.textContent = L.hasData ? "" : o.emptyText;
    refs.empty.setAttribute("x", r2(m.l + plotW / 2));
    refs.empty.setAttribute("y", r2(m.t + plotH / 2));
    Object.entries({ x: m.l, y: m.t, width: plotW, height: plotH }).forEach(([k, v]) => refs.overlay.setAttribute(k, r2(v)));
    shell.svg.setAttribute("tabindex", L.hasData ? "0" : "-1");
    if (hoverX != null) drawHover();
  }

  function pointAt(s, xv) {
    if (!s.points.length) return null;
    const i = nearestIndex(s.xs, xv);
    const p = s.points[i];
    const tol = (L.xDom[1] - L.xDom[0]) / 60;
    return Math.abs(p.x - xv) <= tol ? p : null;
  }

  function drawHover() {
    const g = refs.hover;
    g.replaceChildren();
    if (hoverX == null || !L.hasData) return [];
    const { m, plotH } = L;
    const px = L.sx(hoverX);
    sv("line", { class: "viz-crosshair", x1: crisp(px), x2: crisp(px), y1: m.t, y2: m.t + plotH }, g);
    const rows = [];
    for (const s of L.series) {
      const p = pointAt(s, hoverX);
      if (!p) continue;
      sv("circle", { class: "viz-dot", r: 4, cx: r2(L.sx(p.x)), cy: r2(L.sy(p.y)), style: { fill: s.color } }, g);
      rows.push({ label: s.label, value: L.yValFmt(p.y), color: s.color, shape: "line" });
    }
    return rows;
  }

  function tipTitle(xv) {
    return o.xLabel ? `${o.xLabel} ${L.xValFmt(xv)}` : L.xValFmt(xv);
  }

  function showAt(xv, clientX, clientY) {
    hoverX = xv;
    const rows = drawHover();
    showTooltip({ title: tipTitle(xv), rows }, clientX, clientY, shell);
  }

  function onPointer(e) {
    if (!L || !L.hasData || !L.sx) return;
    const rect = shell.svg.getBoundingClientRect();
    const xv = L.sx.invert(e.clientX - rect.left);
    const idx = nearestIndex(L.xs, xv);
    if (idx < 0) return;
    showAt(L.xs[idx], e.clientX, e.clientY);
    if (e.pointerType && e.pointerType !== "mouse") {
      clearTimeout(touchTimer);
      touchTimer = setTimeout(clearHover, 2600);
    }
  }

  function clearHover() {
    hoverX = null;
    if (refs) refs.hover.replaceChildren();
    hideTooltip(shell);
  }

  function keyShow(idx) {
    if (!L || !L.xs.length) return;
    const i = clamp(idx, 0, L.xs.length - 1);
    const rect = shell.svg.getBoundingClientRect();
    const xv = L.xs[i];
    showAt(xv, rect.left + L.sx(xv), rect.top + L.m.t + 10);
  }

  shell.svg.addEventListener("keydown", (e) => {
    if (!L || !L.xs.length) return;
    const cur = hoverX == null ? L.xs.length - 1 : nearestIndex(L.xs, hoverX);
    const map = { ArrowLeft: cur - 1, ArrowRight: cur + 1, Home: 0, End: L.xs.length - 1 };
    if (e.key in map) {
      e.preventDefault();
      keyShow(map[e.key]);
    } else if (e.key === "Escape") clearHover();
  });
  shell.svg.addEventListener("focus", () => keyShow(hoverX == null ? L?.xs.length - 1 : nearestIndex(L.xs, hoverX)));
  shell.svg.addEventListener("blur", clearHover);

  function targetState() {
    const counts = {};
    for (const s of L.series) counts[s.id] = s.points.length;
    return { xDom: L.xDom.slice(), yDom: L.yDom.slice(), counts };
  }

  function resetIntroStyles() {
    if (!refs) return;
    refs.lines.forEach((p) => {
      p.style.strokeDasharray = "";
      p.style.strokeDashoffset = "";
    });
    refs.areas.forEach((a) => a && (a.style.opacity = ""));
    refs.endG.style.opacity = "";
  }

  function render(W, mode) {
    stop();
    resetIntroStyles();
    const before = shown;
    L = layout(W);
    const k = `${W}|${L.H}|${o.area ? 1 : 0}|${L.series.map((s) => s.id).join(",")}`;
    if (k !== key || !refs) {
      build();
      key = k;
    }
    shell.setAria(o.ariaLabel);
    renderLegend(shell.legendEl, L.series.map((s) => ({ label: s.label, color: s.color, shape: "line" })));
    const to = targetState();
    if (mode === "intro" && L.hasData && !reducedMotion()) {
      paint(to);
      shown = to;
      const items = refs.lines.map((p) => {
        let len = 0;
        try {
          len = p.getTotalLength();
        } catch {
          len = 0;
        }
        if (len > 0) {
          p.style.strokeDasharray = `${len} ${len}`;
          p.style.strokeDashoffset = String(len);
        }
        return { p, len };
      });
      refs.areas.forEach((a) => a && (a.style.opacity = "0"));
      refs.endG.style.opacity = "0";
      stop = tween(
        820,
        (t) => {
          const e = easeInOutCubic(t);
          for (const { p, len } of items) if (len > 0) p.style.strokeDashoffset = String(len * (1 - e));
          refs.areas.forEach((a) => a && (a.style.opacity = String(clamp((t - 0.25) / 0.75, 0, 1))));
          refs.endG.style.opacity = String(clamp((t - 0.72) / 0.28, 0, 1));
        },
        resetIntroStyles
      );
    } else if (mode === "morph" && before) {
      const from = { xDom: before.xDom, yDom: before.yDom, counts: {} };
      for (const s of L.series) {
        let c = before.counts[s.id] ?? 0;
        const first = s.points.length ? s.points[0].x : null;
        if (prevFirst.get(s.id) !== first || c > s.points.length) c = 0;
        from.counts[s.id] = c;
      }
      stop = tween(460, (t) => {
        const e = easeOutCubic(t);
        const st = {
          xDom: [lerp(from.xDom[0], to.xDom[0], e), lerp(from.xDom[1], to.xDom[1], e)],
          yDom: [lerp(from.yDom[0], to.yDom[0], e), lerp(from.yDom[1], to.yDom[1], e)],
          counts: {},
        };
        for (const s of L.series) st.counts[s.id] = lerp(from.counts[s.id], to.counts[s.id], e);
        paint(st);
        shown = st;
      });
    } else {
      paint(to);
      shown = to;
    }
    for (const s of L.series) prevFirst.set(s.id, s.points.length ? s.points[0].x : null);
  }

  const size = sizer(el, render);
  return {
    update(next = {}) {
      o = { ...o, ...next };
      size.invalidate();
    },
    destroy() {
      stop();
      size.destroy();
      clearTimeout(touchTimer);
      shell.destroyBase();
    },
  };
}

/* ================================================================== curveChart */

/**
 * curveChart(el, {kind:'pr'|'roc', series:[{label,color,x:[],y:[],auc?,thresholds?}],
 *   baseline?, point?:{x,y,label}, height=260, xLabel?, yLabel?, ariaLabel})
 * PR curves are drawn as steps (sklearn's display); ROC shows the chance diagonal.
 * update({point}) morphs the operating-point marker; new series draw in.
 */
export function curveChart(el, opts = {}) {
  let o = { kind: "pr", height: 260, ...opts };
  const shell = createShell(el, o, { cls: "viz-curve" });
  let L = null;
  let refs = null;
  let stop = noop;
  let stopPt = noop;
  let shownPt = null;
  let sig = "";
  let touchTimer = 0;

  const names = () =>
    o.kind === "roc"
      ? { x: o.xLabel || "False positive rate", y: o.yLabel || "True positive rate", auc: "AUC" }
      : { x: o.xLabel || "Recall", y: o.yLabel || "Precision", auc: "AP" };

  function seriesSig() {
    return (o.series || [])
      .map((s) => {
        const n = (s.x || []).length;
        const mid = n >> 1;
        return `${s.label}|${s.color}|${n}|${s.x?.[mid]}|${s.y?.[mid]}|${s.y?.[n - 1]}|${s.auc}`;
      })
      .join(";") + `|${o.kind}`;
  }

  function layout(W) {
    const H = o.height;
    const nm = names();
    const m = { t: 14, r: 14, b: 42, l: 50 };
    const plotW = Math.max(60, W - m.l - m.r);
    const plotH = Math.max(60, H - m.t - m.b);
    const tickStep = Math.min(plotW, plotH) < 150 ? 0.5 : 0.2;
    const ticks = ticksInDomain(0, 1, 1 / tickStep);
    const series = (o.series || []).map((s, i) => {
      const xsArr = Array.from(s.x || []);
      const ysArr = Array.from(s.y || []);
      const n = Math.min(xsArr.length, ysArr.length);
      const pts = [];
      // NaN / null points are skipped (tooltips keep the raw values; drawing clamps to the fixed [0, 1] frame)
      for (let k = 0; k < n; k++) if (Number.isFinite(xsArr[k]) && Number.isFinite(ysArr[k])) pts.push([xsArr[k], ysArr[k], k]);
      return { label: s.label ?? `Model ${i + 1}`, color: resolveColor(s.color, i), pts, auc: s.auc, thresholds: s.thresholds ? Array.from(s.thresholds) : null };
    });
    const x = linearScale([0, 1], [m.l, m.l + plotW]);
    const y = linearScale([0, 1], [m.t + plotH, m.t]);
    return { W, H, m, plotW, plotH, ticks, tickStep, series, x, y, nm };
  }

  function build() {
    const { m, plotW, plotH, W, H, x, y, ticks, tickStep, nm } = L;
    shell.size(W, H);
    const g = shell.root;
    g.replaceChildren();
    const ax = sv("g", null, g);
    const fmtT = (v) => formatTick(v, tickStep);
    drawYAxis(ax, { ticks, fmt: fmtT, y, m, plotW, plotH });
    for (const t of ticks) {
      const px = x(t);
      if (t > 0) sv("line", { class: "viz-grid", x1: crisp(px), x2: crisp(px), y1: m.t, y2: m.t + plotH }, ax);
      sv("text", { class: "viz-tick", x: r2(px), y: m.t + plotH + 16, "text-anchor": "middle", text: fmtT(t) }, ax);
    }
    sv("line", { class: "viz-axis", x1: m.l, x2: m.l + plotW, y1: crisp(m.t + plotH), y2: crisp(m.t + plotH) }, ax);
    sv("line", { class: "viz-axis", x1: crisp(m.l), x2: crisp(m.l), y1: m.t, y2: m.t + plotH }, ax);
    drawAxisLabels(ax, { xLabel: nm.x, yLabel: nm.y, m, plotW, plotH, H });
    if (o.kind === "roc") {
      sv("line", { class: "viz-ref", x1: x(0), y1: y(0), x2: x(1), y2: y(1) }, ax);
      const ang = (-Math.atan2(plotH, plotW) * 180) / Math.PI;
      sv("text", { class: "viz-ref-label", transform: `translate(${r2(x(0.7))} ${r2(y(0.7) - 6)}) rotate(${r2(ang)})`, "text-anchor": "middle", text: "Chance" }, ax);
    } else if (Number.isFinite(o.baseline)) {
      const py = y(clamp(o.baseline, 0, 1));
      sv("line", { class: "viz-ref", x1: m.l, x2: m.l + plotW, y1: crisp(py), y2: crisp(py) }, ax);
      const label = o.baselineLabel || `No-skill baseline (prevalence ${formatAuto(o.baseline)})`;
      fitText({ class: "viz-ref-label", x: m.l + plotW - 4, y: r2(o.baseline > 0.88 ? py + 13 : py - 6), "text-anchor": "end" }, ax, label, plotW - 8, 10.5, 560);
    }
    refs = { areas: [], lines: [], pix: [] };
    const areaG = sv("g", null, g);
    const lineG = sv("g", null, g);
    const single = L.series.length === 1;
    L.series.forEach((s) => {
      const pp = s.pts.map(([a, b]) => [x(clamp(a, 0, 1)), y(clamp(b, 0, 1))]);
      refs.pix.push(pp);
      const d = o.kind === "roc" ? linePath(pp) : stepPath(pp);
      if (single && pp.length > 1) {
        const first = pp[0];
        const last = pp[pp.length - 1];
        const y0 = r2(y(0));
        refs.areas.push(sv("path", { class: "viz-area", d: `${d}L${r2(last[0])},${y0}L${r2(first[0])},${y0}Z`, style: { fill: s.color } }, areaG));
      }
      refs.lines.push(sv("path", { class: "viz-line-path", d, style: { stroke: s.color } }, lineG));
      // a one-point "curve" has no visible path: mark the point itself
      if (pp.length === 1) sv("circle", { class: "viz-dot", r: 4, cx: r2(pp[0][0]), cy: r2(pp[0][1]), style: { fill: s.color } }, lineG);
    });
    refs.hover = sv("g", { class: "viz-hover" }, g);
    refs.op = sv("g", { class: "viz-op-g" }, g);
    refs.opRing = sv("circle", { class: "viz-op", r: 6 }, refs.op);
    refs.opLabel = sv("text", { class: "viz-direct-label", dy: "0.32em" }, refs.op);
    refs.overlay = sv("rect", { class: "viz-overlay", x: m.l, y: m.t, width: plotW, height: plotH }, g);
    refs.overlay.addEventListener("pointermove", onPointer);
    refs.overlay.addEventListener("pointerdown", onPointer);
    refs.overlay.addEventListener("pointerleave", (e) => e.pointerType === "mouse" && clearHover());
    shell.svg.setAttribute("tabindex", L.series.some((s) => s.pts.length) ? "0" : "-1");
  }

  function paintPoint(p, scale = 1) {
    if (!p) {
      refs.op.style.display = "none";
      return;
    }
    const { x, y, m, plotW } = L;
    const px = x(clamp(p.x, 0, 1));
    const py = y(clamp(p.y, 0, 1));
    refs.op.style.display = "";
    refs.opRing.setAttribute("cx", r2(px));
    refs.opRing.setAttribute("cy", r2(py));
    refs.opRing.setAttribute("r", r2(6 * scale));
    const label = o.point?.label ?? "";
    refs.opLabel.textContent = label;
    const right = px + 12 + textWidth(label, 11.5, 600) < m.l + plotW;
    refs.opLabel.setAttribute("x", r2(right ? px + 12 : px - 12));
    refs.opLabel.setAttribute("text-anchor", right ? "start" : "end");
    refs.opLabel.setAttribute("y", r2(py < m.t + 14 ? py + 14 : py - 12));
    refs.opLabel.style.opacity = String(scale);
  }

  function nearest(px, py) {
    let best = null;
    let bd = Infinity;
    refs.pix.forEach((pp, si) => {
      for (let k = 0; k < pp.length; k++) {
        const dx = pp[k][0] - px;
        const dy = pp[k][1] - py;
        const d = dx * dx + dy * dy;
        if (d < bd) {
          bd = d;
          best = { si, k };
        }
      }
    });
    return best;
  }

  let hoverSel = null;
  function drawHover(sel) {
    const g = refs.hover;
    g.replaceChildren();
    hoverSel = sel;
    if (!sel) return null;
    const s = L.series[sel.si];
    const [px, py] = refs.pix[sel.si][sel.k];
    const { m, plotH } = L;
    sv("line", { class: "viz-crosshair", x1: crisp(px), x2: crisp(px), y1: m.t, y2: m.t + plotH }, g);
    sv("line", { class: "viz-crosshair", x1: m.l, x2: r2(px), y1: crisp(py), y2: crisp(py) }, g);
    sv("circle", { class: "viz-dot", r: 4, cx: r2(px), cy: r2(py), style: { fill: s.color } }, g);
    const [xv, yv, idx] = s.pts[sel.k];
    const rows = [
      { label: L.nm.y, value: formatAuto(yv) },
      { label: L.nm.x, value: formatAuto(xv) },
    ];
    if (s.thresholds && idx < s.thresholds.length) {
      const t = s.thresholds[idx];
      rows.push({ label: "Threshold", value: Number.isFinite(t) ? formatAuto(t) : "above every score" });
    }
    return { title: s.label, rows: rows.map((r, i) => (i === 0 ? { ...r, color: s.color, shape: "line" } : r)) };
  }

  function onPointer(e) {
    if (!refs) return;
    const rect = shell.svg.getBoundingClientRect();
    const sel = nearest(e.clientX - rect.left, e.clientY - rect.top);
    const c = drawHover(sel);
    if (c) showTooltip(c, e.clientX, e.clientY, shell);
    if (e.pointerType && e.pointerType !== "mouse") {
      clearTimeout(touchTimer);
      touchTimer = setTimeout(clearHover, 2600);
    }
  }

  function clearHover() {
    hoverSel = null;
    if (refs) refs.hover.replaceChildren();
    hideTooltip(shell);
  }

  shell.svg.addEventListener("keydown", (e) => {
    if (!refs || !L.series.length) return;
    const cur = hoverSel || { si: 0, k: 0 };
    const n = refs.pix[cur.si]?.length || 0;
    if (!n) return;
    let next = null;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = { si: cur.si, k: Math.min(n - 1, cur.k + Math.max(1, Math.round(n / 40))) };
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = { si: cur.si, k: Math.max(0, cur.k - Math.max(1, Math.round(n / 40))) };
    else if (e.key === "PageDown") next = { si: (cur.si + 1) % L.series.length, k: 0 };
    else if (e.key === "Escape") return clearHover();
    if (!next || !refs.pix[next.si]?.length) return;
    e.preventDefault();
    const c = drawHover(next);
    const rect = shell.svg.getBoundingClientRect();
    const [px, py] = refs.pix[next.si][next.k];
    if (c) showTooltip(c, rect.left + px, rect.top + py, shell);
  });
  shell.svg.addEventListener("blur", clearHover);

  let key = "";
  function render(W, mode) {
    stopPt();
    const newSig = seriesSig();
    const seriesChanged = newSig !== sig;
    sig = newSig;
    const k = `${W}|${o.height}|${newSig}|${o.baseline}|${o.baselineLabel}|${o.xLabel}|${o.yLabel}`;
    const rebuild = k !== key || !refs;
    if (rebuild) {
      stop();
      L = layout(W);
      build();
      key = k;
    }
    shell.setAria(o.ariaLabel);
    if (rebuild) {
      const nm = L.nm;
      const showLegend = L.series.length >= 2 || L.series.some((s) => Number.isFinite(s.auc));
      renderLegend(
        shell.legendEl,
        L.series.map((s) => ({ label: s.label, color: s.color, shape: "line", value: Number.isFinite(s.auc) ? `${nm.auc} ${formatAuto(s.auc)}` : null })),
        showLegend
      );
    }
    const target = o.point && Number.isFinite(o.point.x) && Number.isFinite(o.point.y) ? { x: o.point.x, y: o.point.y } : null;
    const drawIn = rebuild && (mode === "intro" || (mode === "morph" && seriesChanged)) && !reducedMotion();
    if (drawIn) {
      const items = refs.lines.map((p) => {
        let len = 0;
        try {
          len = p.getTotalLength();
        } catch {
          len = 0;
        }
        if (len > 0) {
          p.style.strokeDasharray = `${len} ${len}`;
          p.style.strokeDashoffset = String(len);
        }
        return { p, len };
      });
      refs.areas.forEach((a) => (a.style.opacity = "0"));
      stop = tween(
        mode === "intro" ? 820 : 620,
        (t) => {
          const e = easeInOutCubic(t);
          for (const { p, len } of items) if (len > 0) p.style.strokeDashoffset = String(len * (1 - e));
          refs.areas.forEach((a) => (a.style.opacity = String(clamp((t - 0.3) / 0.7, 0, 1))));
        },
        () => {
          for (const { p } of items) {
            p.style.strokeDasharray = "";
            p.style.strokeDashoffset = "";
          }
          refs.areas.forEach((a) => (a.style.opacity = ""));
        }
      );
    }
    if (!target) {
      paintPoint(null);
      shownPt = null;
    } else if (mode === "snap" || !shownPt || reducedMotion()) {
      if (!shownPt && mode !== "snap" && !reducedMotion()) {
        paintPoint(target, 0);
        const delay = drawIn ? 0.55 : 0;
        stopPt = tween(drawIn ? 900 : 320, (t) => {
          const lt = clamp((t - delay) / (1 - delay), 0, 1);
          paintPoint(target, lt < 1 ? easeOutCubic(lt) * (1 + 0.25 * Math.sin(lt * Math.PI)) : 1);
        });
      } else paintPoint(target);
      shownPt = target;
    } else {
      const from = { ...shownPt };
      stopPt = tween(360, (t) => {
        const e = easeOutCubic(t);
        const p = { x: lerp(from.x, target.x, e), y: lerp(from.y, target.y, e) };
        paintPoint(p);
        shownPt = p;
      });
    }
    if (rebuild && hoverSel && refs.pix[hoverSel.si]?.[hoverSel.k]) drawHover(hoverSel);
  }

  const size = sizer(el, render);
  return {
    update(next = {}) {
      o = { ...o, ...next };
      size.invalidate();
    },
    destroy() {
      stop();
      stopPt();
      clearTimeout(touchTimer);
      size.destroy();
      shell.destroyBase();
    },
  };
}

/* ================================================================== barChart */

function ensureHatch(svg, id) {
  let defs = svg.querySelector("defs");
  if (!defs) defs = sv("defs", null, svg);
  if (!defs.querySelector(`#${id}`)) {
    const p = sv("pattern", { id, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
    sv("line", { class: "viz-hatch-line", x1: 0, y1: 0, x2: 0, y2: 6 }, p);
  }
  return `url(#${id})`;
}

/**
 * barChart(el, {bars:[{label, value, color?, flagged?, note?}], yDomain?, yFormat?,
 *   refLine?:{value,label}, height=240, horizontal?, valueLabels:'none'|'ends'|'flagged'='ends',
 *   valueName?, flaggedLabel?, ariaLabel})
 * Bars grow from the zero baseline (negative values grow downwards / leftwards).
 * Flagged bars are hatched and lighter (e.g. low-support folds excluded from the mean).
 */
export function barChart(el, opts = {}) {
  let o = { valueLabels: "ends", ...opts };
  const shell = createShell(el, o, { cls: "viz-bars", legend: "bottom" });
  let L = null;
  let refs = null;
  let key = "";
  let shown = null;
  let stop = noop;
  const hatchId = `${shell.id}-hatch`;

  function layout(W) {
    const horizontal = !!o.horizontal;
    const bars = (o.bars || []).map((b, i) => ({
      key: String(b.id ?? b.label ?? i),
      label: String(b.label ?? ""),
      value: Number.isFinite(b.value) ? b.value : null,
      color: resolveColor(b.color ?? "--series-1", 0),
      flagged: !!b.flagged,
      note: b.note || "",
    }));
    const n = bars.length;
    const H = o.height ?? (horizontal ? Math.max(110, n * 34 + 40) : 240);
    const vals = bars.filter((b) => b.value != null).map((b) => b.value);
    if (o.refLine && Number.isFinite(o.refLine.value)) vals.push(o.refLine.value);
    const valFmt = o.yFormat || formatAuto;
    const m = { t: 10, r: 12, b: 26, l: 0 };
    let dom = validDomain(o.yDomain);
    let ticks;
    let step;
    const count = horizontal ? Math.max(2, Math.floor((W * 0.6) / 90)) : Math.max(2, Math.round((H - 50) / 46));
    // vertical: ~one tick per 36 px of plot height; horizontal: the x labels need ~60 px each
    const maxTicks = horizontal ? tickBudget(W * 0.6, 60) : tickBudget(H - 50);
    // counts (all whole numbers) get whole-number ticks: never 0, 0.5, 1
    const integer = !dom && vals.length > 0 && vals.every((v) => Number.isInteger(v));
    if (dom) {
      ticks = ticksInDomain(dom[0], dom[1], count, { maxTicks });
      step = tickStep(ticks);
    } else {
      // bars grow from zero, so zero is always in the domain; all zeros -> [0, 1]
      const nt = autoDomain(vals, { count, maxTicks, includeZero: true, integer });
      dom = [nt.min, nt.max];
      ticks = nt.ticks;
      step = nt.step;
    }
    const tickFmt = pickTickFormat(o.yFormat, ticks, step);
    const labelled = (b) => o.valueLabels === "ends" || (o.valueLabels === "flagged" && b.flagged);
    let rotate = false;
    let plotW;
    let plotH;
    if (horizontal) {
      const maxLab = Math.max(0, ...bars.map((b) => textWidth(b.label, 12, 540)));
      m.l = Math.ceil(Math.min(W * 0.4, maxLab + 14));
      const maxVal = Math.max(0, ...bars.filter(labelled).map((b) => textWidth(b.value == null ? "n/a" : valFmt(b.value), 11, 600)));
      m.r = 12 + (o.valueLabels !== "none" ? Math.ceil(maxVal) + 8 : 0);
      m.t = 6;
      m.b = 24;
      plotW = Math.max(40, W - m.l - m.r);
      plotH = Math.max(30, H - m.t - m.b);
    } else {
      m.l = axisLeftWidth(ticks.map(tickFmt));
      m.t = o.valueLabels !== "none" ? 20 : 10;
      plotW = Math.max(40, W - m.l - m.r);
      const band = plotW / Math.max(1, n);
      const maxLab = Math.max(0, ...bars.map((b) => textWidth(b.label, 11)));
      rotate = maxLab > band - 6;
      m.b = rotate ? Math.min(84, Math.ceil(Math.min(maxLab, 92) * 0.72) + 18) : 26;
      plotH = Math.max(30, H - m.t - m.b);
    }
    const bandLen = (horizontal ? plotH : plotW) / Math.max(1, n);
    const thick = Math.max(2, Math.min(24, bandLen * 0.68, bandLen - 2));
    return { W, H, m, plotW, plotH, bars, n, dom, ticks, tickFmt, valFmt, horizontal, rotate, bandLen, thick, labelled };
  }

  function build() {
    const { m, plotW, plotH, W, H, bars, horizontal, rotate, bandLen } = L;
    shell.size(W, H);
    const g = shell.root;
    g.replaceChildren();
    const hatch = bars.some((b) => b.flagged) ? ensureHatch(shell.svg, hatchId) : null;
    refs = { grid: sv("g", null, g), bars: [] };
    const cats = sv("g", null, g);
    refs.ref = sv("g", null, g);
    const barsG = sv("g", null, g);
    bars.forEach((b, i) => {
      const c = m.l + (i + 0.5) * bandLen;
      const cy = m.t + (i + 0.5) * bandLen;
      // long category labels are ellipsised; the full text is in the <title>, the tooltip and the aria-label
      if (horizontal) {
        fitText({ class: "viz-cat", x: m.l - 10, y: r2(cy), dy: "0.32em", "text-anchor": "end" }, cats, b.label, m.l - 14, 12, 540);
      } else if (rotate) {
        fitText({ class: "viz-tick", transform: `translate(${r2(c + 3)} ${m.t + plotH + 12}) rotate(-40)`, "text-anchor": "end" }, cats, b.label, 92, 11);
      } else {
        fitText({ class: "viz-tick", x: r2(c), y: m.t + plotH + 16, "text-anchor": "middle" }, cats, b.label, Math.max(12, bandLen - 4), 11);
      }
      const grp = sv("g", { class: "viz-bar-g", tabindex: 0 }, barsG);
      const hit = horizontal
        ? sv("rect", { class: "viz-hit", x: m.l, y: r2(m.t + i * bandLen), width: plotW, height: r2(bandLen) }, grp)
        : sv("rect", { class: "viz-hit", x: r2(m.l + i * bandLen), y: m.t, width: r2(bandLen), height: plotH }, grp);
      const path = sv("path", { class: "viz-bar" + (b.flagged ? " is-flagged" : ""), style: { fill: b.color } }, grp);
      const hatchPath = b.flagged ? sv("path", { class: "viz-bar-hatch", fill: hatch }, grp) : null;
      const label = sv("text", { class: "viz-value-label", dy: horizontal ? "0.32em" : null }, grp);
      const rec = { b, path, hatchPath, label, hit, grp };
      bindMarkTooltip(grp, shell, () => {
        const cur = rec.b;
        return {
          title: cur.label,
          rows: [{ label: o.valueName || "Value", value: cur.value == null ? "not defined" : L.valFmt(cur.value), color: cur.color, shape: "rect" }],
          note: cur.note || (cur.flagged ? o.flaggedLabel || "Flagged" : ""),
        };
      });
      refs.bars.push(rec);
    });
  }

  function paint(st) {
    const { m, plotW, plotH, horizontal, bandLen, thick } = L;
    const sc = horizontal ? linearScale(st.dom, [m.l, m.l + plotW]) : linearScale(st.dom, [m.t + plotH, m.t]);
    const g = refs.grid;
    g.replaceChildren();
    if (horizontal) {
      drawXTicks(g, { ticks: L.ticks, fmt: L.tickFmt, x: sc, m, plotW, y: m.t + plotH + 16, grid: { when: () => true, y1: m.t, y2: m.t + plotH } });
    } else {
      drawYAxis(g, { ticks: L.ticks, fmt: L.tickFmt, y: sc, m, plotW, plotH });
    }
    const dLo = Math.min(...st.dom);
    const dHi = Math.max(...st.dom);
    const zero = sc(clamp(0, dLo, dHi));
    // a fixed domain narrower than the data: bars stop at the frame (labels still print the true value)
    const inDom = (v) => clamp(v, dLo, dHi);
    if (horizontal) sv("line", { class: "viz-axis", x1: crisp(zero), x2: crisp(zero), y1: m.t, y2: m.t + plotH }, g);
    else sv("line", { class: "viz-axis", x1: m.l, x2: m.l + plotW, y1: crisp(zero), y2: crisp(zero) }, g);
    refs.bars.forEach((r, i) => {
      const v = st.values[i];
      const alpha = st.alpha ? st.alpha[i] : 1;
      const target = r.b.value;
      let d = "";
      if (v != null && target != null) {
        const pv = sc(inDom(v));
        if (horizontal) {
          const y0 = m.t + i * bandLen + (bandLen - thick) / 2;
          d = v >= 0 ? roundedBarPath(zero, y0, pv - zero, thick, 4, "right") : roundedBarPath(pv, y0, zero - pv, thick, 4, "left");
        } else {
          const x0 = m.l + i * bandLen + (bandLen - thick) / 2;
          d = v >= 0 ? roundedBarPath(x0, pv, thick, zero - pv, 4, "top") : roundedBarPath(x0, zero, thick, pv - zero, 4, "bottom");
        }
      }
      r.path.setAttribute("d", d);
      if (r.hatchPath) r.hatchPath.setAttribute("d", d);
      const lab = r.label;
      if (L.labelled(r.b)) {
        lab.style.display = "";
        lab.style.opacity = String(alpha);
        lab.textContent = target == null ? "n/a" : L.valFmt(target);
        const pv = sc(inDom(target == null ? 0 : v ?? 0));
        if (horizontal) {
          const cy = m.t + (i + 0.5) * bandLen;
          const neg = (v ?? 0) < 0;
          lab.setAttribute("x", r2(neg ? Math.min(pv, zero) - 6 : Math.max(pv, zero) + 6));
          lab.setAttribute("y", r2(cy));
          lab.setAttribute("text-anchor", neg ? "end" : "start");
        } else {
          const cx = m.l + (i + 0.5) * bandLen;
          const neg = (v ?? 0) < 0;
          lab.setAttribute("x", r2(cx));
          lab.setAttribute("y", r2(neg ? Math.max(pv, zero) + 13 : Math.min(pv, zero) - 6));
          lab.setAttribute("text-anchor", "middle");
        }
      } else lab.style.display = "none";
    });
    const rl = o.refLine;
    refs.ref.replaceChildren();
    if (rl && Number.isFinite(rl.value) && rl.value >= dLo && rl.value <= dHi) {
      const p = sc(rl.value);
      if (horizontal) {
        sv("line", { class: "viz-ref", x1: crisp(p), x2: crisp(p), y1: m.t - 2, y2: m.t + plotH }, refs.ref);
      } else {
        sv("line", { class: "viz-ref", x1: m.l, x2: m.l + plotW, y1: crisp(p), y2: crisp(p) }, refs.ref);
        if (rl.label) sv("text", { class: "viz-ref-label viz-ref-label-bg", x: m.l + plotW, y: r2(p - 6), "text-anchor": "end", text: rl.label }, refs.ref);
      }
    }
  }

  function render(W, mode) {
    stop();
    const before = shown;
    L = layout(W);
    const k = `${W}|${L.H}|${L.horizontal}|${L.bars.map((b) => `${b.key}:${b.flagged}:${b.color}`).join(",")}`;
    if (k !== key || !refs) {
      build();
      key = k;
    } else {
      L.bars.forEach((b, i) => {
        refs.bars[i].b = b;
      });
    }
    refs.bars.forEach(({ b, grp }) => {
      grp.setAttribute("aria-label", `${b.label}: ${b.value == null ? "not available" : L.valFmt(b.value)}${b.note ? ` (${b.note})` : ""}`);
    });
    shell.setAria(o.ariaLabel);
    const flagged = L.bars.find((b) => b.flagged);
    const legend = [];
    if (flagged) legend.push({ label: o.flaggedLabel || flagged.note || "Flagged", color: flagged.color, shape: "hatch" });
    if (o.horizontal && o.refLine?.label) legend.push({ label: o.refLine.label, color: "var(--text-3)", shape: "ref" });
    renderLegend(shell.legendEl, legend, legend.length > 0);
    const keys = L.bars.map((b) => b.key);
    const to = { dom: L.dom.slice(), values: L.bars.map((b) => b.value), keys };
    if (mode === "intro" && !reducedMotion()) {
      const n = L.bars.length;
      const per = 620;
      const gap = Math.min(50, 400 / Math.max(1, n));
      const total = per + gap * (n - 1);
      stop = tween(total, (t) => {
        const ms = t * total;
        const values = [];
        const alpha = [];
        L.bars.forEach((b, i) => {
          const lt = clamp((ms - i * gap) / per, 0, 1);
          values.push(b.value == null ? null : b.value * easeOutCubic(lt));
          alpha.push(clamp((lt - 0.55) / 0.45, 0, 1));
        });
        paint({ dom: to.dom, values, alpha });
        shown = { dom: to.dom, values, keys };
      });
    } else if (mode === "morph" && before) {
      const prev = new Map();
      before.keys.forEach((kk, i) => prev.set(kk, before.values[i]));
      const fromVals = L.bars.map((b) => prev.get(b.key) ?? 0);
      const fromDom = before.dom;
      stop = tween(520, (t) => {
        const e = easeOutCubic(t);
        const st = {
          dom: [lerp(fromDom[0], to.dom[0], e), lerp(fromDom[1], to.dom[1], e)],
          values: to.values.map((v, i) => (v == null ? null : lerp(fromVals[i], v, e))),
        };
        paint(st);
        shown = { ...st, keys };
      });
    } else {
      paint(to);
      shown = to;
    }
  }

  const size = sizer(el, render);
  return {
    update(next = {}) {
      o = { ...o, ...next };
      size.invalidate();
    },
    destroy() {
      stop();
      size.destroy();
      shell.destroyBase();
    },
  };
}

/* ================================================================== histogram */

/**
 * histogram(el, {edges:[], series:[{label,color,counts:[]}], threshold?, onThreshold?(value),
 *   thresholdStep=0.01, xLabel='Score', height=200, logY?, emptyText?, ariaLabel})
 * Series sit side by side in each bin. With onThreshold the threshold line has a
 * draggable, keyboard-operable handle (role="slider").
 */
export function histogram(el, opts = {}) {
  let o = { height: 200, thresholdStep: 0.01, xLabel: "Score", ...opts };
  const shell = createShell(el, o, { cls: "viz-hist" });
  let L = null;
  let refs = null;
  let key = "";
  let shown = null;
  let stop = noop;
  let stopThr = noop;
  let thr = Number.isFinite(o.threshold) ? o.threshold : null;
  let dragging = false;
  let emitRaf = 0;
  let handle = null;

  function layout(W) {
    const H = o.height;
    // bin edges must all be finite numbers; otherwise the chart shows its empty state
    const rawEdges = Array.from(o.edges || []);
    const edgesOk = rawEdges.length >= 2 && rawEdges.every((e) => typeof e === "number" && Number.isFinite(e));
    const edges = edgesOk ? rawEdges : [];
    const nb = Math.max(0, edges.length - 1);
    const series = (o.series || []).map((s, i) => ({
      label: s.label ?? `Series ${i + 1}`,
      color: resolveColor(s.color, i),
      // counts: NaN / null -> 0, negatives -> 0 (a count cannot be below zero)
      counts: Array.from({ length: nb }, (_, b) => Math.max(0, Number((s.counts || [])[b]) || 0)),
    }));
    let maxC = 0;
    let allInt = true;
    for (const s of series) for (const c of s.counts) {
      if (c > maxC) maxC = c;
      if (!Number.isInteger(c)) allInt = false;
    }
    const m = { t: Number.isFinite(thr) || Number.isFinite(o.threshold) ? 24 : 10, r: 12, b: 26 + (o.xLabel ? 18 : 0), l: 0 };
    const plotH = Math.max(40, H - m.t - m.b);
    let yTicks;
    let yMax;
    let yStep;
    if (o.logY) {
      yMax = Math.max(1, maxC);
      yTicks = log1pTicks(yMax);
      yMax = Math.max(yMax, yTicks[yTicks.length - 1]);
    } else {
      // whole-number counts get whole-number ticks (a max count of 1 -> 0, 1; never 0, 0.5, 1)
      const nt = autoDomain([0, maxC], { count: Math.max(2, Math.round(plotH / 44)), maxTicks: tickBudget(plotH), includeZero: true, integer: allInt, bounds: [0, Infinity] });
      yTicks = nt.ticks;
      yMax = nt.max;
      yStep = nt.step;
    }
    const yTickFmt = o.logY ? (v) => (v >= 1e4 ? formatCompact(v) : v.toLocaleString("en-US")) : (v) => formatTick(v, yStep);
    m.l = axisLeftWidth(yTicks.map(yTickFmt), o.yLabel || o.logY ? 18 : 0);
    const plotW = Math.max(40, W - m.l - m.r);
    let xDom = nb ? [Math.min(edges[0], edges[nb]), Math.max(edges[0], edges[nb])] : [0, 1];
    if (xDom[0] === xDom[1]) xDom = [xDom[0] - 0.5, xDom[0] + 0.5];
    const x = linearScale(xDom, [m.l, m.l + plotW]);
    const xTicks = ticksInDomain(xDom[0], xDom[1], Math.max(2, Math.floor(plotW / 70)), { maxTicks: tickBudget(plotW, 48) });
    const hasData = nb > 0 && series.length > 0;
    return { W, H, m, plotW, plotH, edges, nb, series, yTicks, yMax, yTickFmt, x, xTicks, xDom, hasData, xLabelText: o.xLabel || "Score" };
  }

  function yScale(yMax) {
    const { m, plotH } = L;
    return o.logY ? log1pScale([0, yMax], [m.t + plotH, m.t]) : linearScale([0, yMax], [m.t + plotH, m.t]);
  }

  function build() {
    const { m, plotW, plotH, W, H, series, nb, x, xTicks } = L;
    shell.size(W, H);
    const g = shell.root;
    g.replaceChildren();
    refs = { grid: sv("g", null, g), region: sv("rect", { class: "viz-thr-region" }, g), bins: [] };
    const ax = sv("g", null, g);
    const xStep = tickStep(xTicks);
    drawXTicks(ax, { ticks: xTicks, fmt: (t) => formatTick(t, xStep), x, m, plotW, y: m.t + plotH + 16 });
    drawAxisLabels(ax, { xLabel: o.xLabel, yLabel: o.yLabel || (o.logY ? "Windows (log scale)" : null), m, plotW, plotH, H });
    if (!L.hasData) {
      sv("text", { class: "viz-empty", x: r2(m.l + plotW / 2), y: r2(m.t + plotH / 2), "text-anchor": "middle", text: o.emptyText || "No scores to show" }, ax);
    }
    const barsG = sv("g", null, g);
    const K = Math.max(1, series.length);
    for (let b = 0; b < nb; b++) {
      const x0 = x(L.edges[b]);
      const x1 = x(L.edges[b + 1]);
      const inner = x1 - x0 - 2;
      const sw = Math.max(1, (inner - (K - 1) * 2) / K);
      const grp = sv("g", { class: "viz-bin-g" }, barsG);
      const hit = sv("rect", { class: "viz-hit", x: r2(x0), y: m.t, width: r2(Math.max(1, x1 - x0)), height: plotH }, grp);
      const paths = series.map((s, k) => ({ path: sv("path", { class: "viz-bar", style: { fill: s.color } }, grp), x: x0 + 1 + k * (sw + 2), w: sw, k }));
      const lo = L.edges[b];
      const hi = L.edges[b + 1];
      bindMarkTooltip(grp, shell, () => ({
        title: `${L.xLabelText} ${formatAuto(lo)}–${formatAuto(hi)}`,
        rows: L.series.map((s) => ({ label: s.label, value: s.counts[b].toLocaleString("en-US"), color: s.color, shape: "rect" })),
      }));
      refs.bins.push({ hit, paths });
    }
    refs.axisLine = sv("line", { class: "viz-axis", x1: m.l, x2: m.l + plotW, y1: crisp(m.t + plotH), y2: crisp(m.t + plotH) }, g);
    refs.thrG = sv("g", { class: "viz-thr" }, g);
    refs.thrLine = sv("line", { class: "viz-thr-line", y1: m.t - 2, y2: m.t + plotH }, refs.thrG);
    refs.thrLabel = sv("text", { class: "viz-thr-label", y: m.t - 9 }, refs.thrG);
    refs.thrHit = sv("rect", { class: "viz-thr-hit", y: m.t, width: 18, height: plotH }, refs.thrG);
    if (o.onThreshold) {
      refs.thrHit.style.cursor = "ew-resize";
      bindDrag(refs.thrHit);
    } else refs.thrHit.style.pointerEvents = "none";
    ensureHandle();
  }

  function ensureHandle() {
    if (!o.onThreshold) {
      if (handle) handle.remove();
      handle = null;
      return;
    }
    if (!handle) {
      handle = hd("div", "viz-handle");
      handle.setAttribute("role", "slider");
      handle.setAttribute("tabindex", "0");
      handle.setAttribute("aria-label", o.thresholdLabel || "Decision threshold");
      handle.appendChild(hd("span", "viz-handle-grip"));
      bindDrag(handle);
      handle.addEventListener("keydown", (e) => {
        const stp = o.thresholdStep || 0.01;
        const big = stp * 10;
        const cur = Number.isFinite(thr) ? thr : 0.5;
        const map = { ArrowLeft: cur - stp, ArrowDown: cur - stp, ArrowRight: cur + stp, ArrowUp: cur + stp, PageDown: cur - big, PageUp: cur + big, Home: L.xDom[0], End: L.xDom[1] };
        if (e.key in map) {
          e.preventDefault();
          setThreshold(map[e.key], true);
        }
      });
      el.appendChild(handle);
    }
    handle.setAttribute("aria-valuemin", String(L.xDom[0]));
    handle.setAttribute("aria-valuemax", String(L.xDom[1]));
  }

  function bindDrag(node) {
    node.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      dragging = true;
      stopThr();
      try {
        node.setPointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
      el.classList.add("is-dragging");
      if (node === handle) handle.focus({ preventScroll: true });
    });
    node.addEventListener("pointermove", (e) => {
      if (!dragging || !L) return;
      const rect = shell.svg.getBoundingClientRect();
      setThreshold(L.x.invert(e.clientX - rect.left), true);
    });
    const end = () => {
      if (!dragging) return;
      dragging = false;
      el.classList.remove("is-dragging");
      emit(true);
    };
    node.addEventListener("pointerup", end);
    node.addEventListener("pointercancel", end);
  }

  function emit(now = false) {
    if (!o.onThreshold || !Number.isFinite(thr)) return;
    if (now) {
      cancelAnimationFrame(emitRaf);
      emitRaf = 0;
      o.onThreshold(thr);
      return;
    }
    if (emitRaf) return;
    emitRaf = requestAnimationFrame(() => {
      emitRaf = 0;
      o.onThreshold(thr);
    });
  }

  function setThreshold(v, fromUser) {
    if (!L) return;
    const stp = o.thresholdStep || 0.01;
    const d = stepDecimals(stp);
    const nv = roundTo(clamp(Math.round(v / stp) * stp, L.xDom[0], L.xDom[1]), d);
    const changed = nv !== thr;
    thr = nv;
    paintThreshold(thr);
    if (fromUser && changed) {
      o = { ...o, threshold: thr };
      emit(!dragging);
    }
  }

  function paintThreshold(v) {
    if (!refs) return;
    const { m, plotW, plotH, x } = L;
    if (!Number.isFinite(v)) {
      refs.thrG.style.display = "none";
      refs.region.style.display = "none";
      if (handle) handle.style.display = "none";
      return;
    }
    const px = x(clamp(v, L.xDom[0], L.xDom[1]));
    refs.thrG.style.display = "";
    refs.region.style.display = "";
    refs.thrLine.setAttribute("x1", r2(px));
    refs.thrLine.setAttribute("x2", r2(px));
    refs.thrHit.setAttribute("x", r2(px - 9));
    refs.region.setAttribute("x", r2(px));
    refs.region.setAttribute("y", m.t);
    refs.region.setAttribute("width", r2(Math.max(0, m.l + plotW - px)));
    refs.region.setAttribute("height", plotH);
    const text = `${o.thresholdLabel || "Threshold"} ${v.toFixed(Math.max(2, stepDecimals(o.thresholdStep || 0.01)))}`;
    refs.thrLabel.textContent = text;
    refs.thrLabel.setAttribute("x", r2(px));
    refs.thrLabel.setAttribute("text-anchor", edgeAnchor(px, m.l, m.l + plotW, textWidth(text, 11, 600) / 2 + 4));
    if (handle) {
      handle.style.display = "";
      handle.style.left = `${r2(px)}px`;
      const svgTop = shell.svg.getBoundingClientRect().top - el.getBoundingClientRect().top;
      handle.style.top = `${r2(svgTop + m.t + plotH / 2)}px`;
      handle.setAttribute("aria-valuenow", String(v));
      handle.setAttribute("aria-valuetext", `${v.toFixed(2)}: windows scoring at or above this are flagged as attacks`);
    }
  }

  function paint(st) {
    const { m, plotW, plotH } = L;
    const y = yScale(st.yMax);
    const g = refs.grid;
    g.replaceChildren();
    drawYAxis(g, { ticks: L.yTicks, fmt: L.yTickFmt, y, m, plotW, plotH });
    const base = m.t + plotH;
    refs.bins.forEach((bin, b) => {
      for (const p of bin.paths) {
        const c = st.counts[p.k]?.[b] ?? 0;
        const top = y(c);
        p.path.setAttribute("d", c > 0 ? roundedBarPath(p.x, top, p.w, Math.max(c > 0 ? 1 : 0, base - top), 4, "top") : "");
      }
    });
  }

  function render(W, mode) {
    stop();
    const before = shown;
    L = layout(W);
    const k = `${W}|${L.H}|${L.edges.join(",")}|${L.series.map((s) => s.label + s.color).join(",")}|${o.logY ? 1 : 0}|${o.xLabel}|${!!o.onThreshold}|${L.m.t}`;
    if (k !== key || !refs) {
      build();
      key = k;
    }
    shell.setAria(o.ariaLabel);
    renderLegend(shell.legendEl, L.series.map((s) => ({ label: s.label, color: s.color, shape: "rect" })));
    const to = { yMax: L.yMax, counts: L.series.map((s) => s.counts) };
    if (mode === "intro" && !reducedMotion()) {
      stop = tween(720, (t) => {
        const st = { yMax: to.yMax, counts: [] };
        to.counts.forEach((cs) => {
          st.counts.push(
            cs.map((c, b) => {
              const lt = clamp((t * 720 - (b * 260) / Math.max(1, L.nb)) / 460, 0, 1);
              return c * easeOutCubic(lt);
            })
          );
        });
        paint(st);
        shown = st;
      });
    } else if (mode === "morph" && before) {
      const from = before;
      stop = tween(480, (t) => {
        const e = easeOutCubic(t);
        const st = {
          yMax: lerp(from.yMax, to.yMax, e),
          counts: to.counts.map((cs, k2) => cs.map((c, b) => lerp(from.counts[k2]?.[b] ?? 0, c, e))),
        };
        paint(st);
        shown = st;
      });
    } else {
      paint(to);
      shown = to;
    }
    // threshold
    const target = Number.isFinite(o.threshold) ? o.threshold : null;
    stopThr();
    if (dragging) paintThreshold(thr);
    else if (target == null) {
      thr = null;
      paintThreshold(null);
    } else if (mode === "morph" && Number.isFinite(thr) && thr !== target && !reducedMotion()) {
      const from = thr;
      stopThr = tween(260, (t) => {
        thr = lerp(from, target, easeOutCubic(t));
        paintThreshold(thr);
      }, () => {
        thr = target;
        paintThreshold(thr);
      });
    } else {
      thr = target;
      paintThreshold(thr);
    }
  }

  const size = sizer(el, render);
  return {
    update(next = {}) {
      o = { ...o, ...next };
      size.invalidate();
    },
    destroy() {
      stop();
      stopThr();
      cancelAnimationFrame(emitRaf);
      size.destroy();
      if (handle) handle.remove();
      shell.destroyBase();
    },
  };
}

/* ================================================================== stackedBar */

/**
 * stackedBar(el, {segments:[{id,label,value,color}], height=34, showPercent=true, format?, ariaLabel})
 * One horizontal 100% bar (e.g. the train / validation / test split). Segment
 * widths animate on update; the legend carries labels, counts and percentages.
 */
export function stackedBar(el, opts = {}) {
  let o = { height: 34, showPercent: true, ...opts };
  const shell = createShell(el, o, { cls: "viz-stacked", legend: "bottom" });
  let L = null;
  let refs = null;
  let key = "";
  let shown = null;
  let stop = noop;
  const clipId = `${shell.id}-clip`;

  function layout(W) {
    const segs = (o.segments || []).map((s, i) => ({
      id: String(s.id ?? s.label ?? i),
      label: s.label ?? `Part ${i + 1}`,
      value: Math.max(0, Number(s.value) || 0),
      color: resolveColor(s.color, i),
    }));
    const total = segs.reduce((a, s) => a + s.value, 0);
    return { W, H: o.height, segs, total, fmtV: o.format || ((v) => Math.round(v).toLocaleString("en-US")) };
  }

  function build() {
    const { W, H, segs } = L;
    shell.size(W, H);
    const g = shell.root;
    g.replaceChildren();
    const defs = sv("defs", null, g);
    const cp = sv("clipPath", { id: clipId }, defs);
    const rr = Math.min(8, H / 2);
    sv("rect", { x: 0, y: 0, width: W, height: H, rx: rr, ry: rr }, cp);
    sv("rect", { class: "viz-track", x: 0, y: 0, width: W, height: H, rx: rr, ry: rr }, g);
    const inner = sv("g", { "clip-path": `url(#${clipId})` }, g);
    refs = { segs: [] };
    segs.forEach((s, i) => {
      const grp = sv("g", { class: "viz-bar-g", tabindex: 0 }, inner);
      const rect = sv("rect", { class: "viz-seg", y: 0, height: H, style: { fill: s.color } }, grp);
      bindMarkTooltip(grp, shell, () => {
        const seg = L.segs[i];
        const pct = L.total > 0 ? seg.value / L.total : 0;
        return { title: seg.label, rows: [{ label: "Count", value: L.fmtV(seg.value), color: seg.color, shape: "rect" }, { label: "Share", value: `${(pct * 100).toFixed(1)}%` }] };
      });
      refs.segs.push({ grp, rect });
    });
  }

  function paint(vals) {
    const { W } = L;
    const total = vals.reduce((a, v) => a + v, 0);
    const GAP = 2;
    const weights = vals.map((v) => (total > 0 ? clamp((v / total) * 60, 0, 1) : 0));
    let nGaps = 0;
    let seen = false;
    const gapBefore = vals.map((v, i) => {
      const gw = seen ? GAP * weights[i] : 0;
      if (weights[i] > 0) seen = true;
      nGaps += gw;
      return gw;
    });
    const avail = Math.max(0, W - nGaps);
    let xpos = 0;
    vals.forEach((v, i) => {
      xpos += gapBefore[i];
      const w = total > 0 ? (v / total) * avail : 0;
      const r = refs.segs[i];
      r.rect.setAttribute("x", r2(xpos));
      r.rect.setAttribute("width", r2(Math.max(0, w)));
      r.grp.style.display = w > 0.1 ? "" : "none";
      xpos += w;
    });
  }

  function render(W, mode) {
    stop();
    const before = shown;
    L = layout(W);
    const k = `${W}|${L.H}|${L.segs.map((s) => s.id + s.color).join(",")}`;
    if (k !== key || !refs) {
      build();
      key = k;
    }
    shell.setAria(o.ariaLabel);
    L.segs.forEach((sg, i) => {
      const pct = L.total > 0 ? ((sg.value / L.total) * 100).toFixed(1) : "0.0";
      refs.segs[i].grp.setAttribute("aria-label", `${sg.label}: ${L.fmtV(sg.value)} (${pct}%)`);
    });
    renderLegend(
      shell.legendEl,
      L.segs.map((s) => ({
        label: s.label,
        color: s.color,
        shape: "rect",
        value: [L.fmtV(s.value), o.showPercent && L.total > 0 ? `${((s.value / L.total) * 100).toFixed(1)}%` : null].filter(Boolean).join(" · "),
      })),
      L.segs.length > 0
    );
    const to = L.segs.map((s) => s.value);
    const fromMap = before ? before : null;
    if ((mode === "intro" || (mode === "morph" && fromMap)) && !reducedMotion()) {
      const from = L.segs.map((s) => (mode === "intro" ? 0 : fromMap.get(s.id) ?? 0));
      const introEmpty = mode === "intro";
      stop = tween(introEmpty ? 780 : 560, (t) => {
        const e = easeInOutCubic(t);
        let vals;
        if (introEmpty) {
          // sweep in from the left: reveal the final layout progressively
          const total = to.reduce((a, v) => a + v, 0);
          const cut = total * e;
          let acc = 0;
          vals = to.map((v) => {
            const take = clamp(cut - acc, 0, v);
            acc += v;
            return take;
          });
          vals.push(Math.max(0, total - cut));
          paintWithTail(vals);
        } else {
          vals = to.map((v, i) => lerp(from[i], v, e));
          paint(vals);
        }
        shown = new Map(L.segs.map((s, i) => [s.id, introEmpty ? to[i] : vals[i]]));
      });
    } else {
      paint(to);
      shown = new Map(L.segs.map((s, i) => [s.id, to[i]]));
    }
  }

  // intro helper: the trailing empty part keeps segments at their final scale
  function paintWithTail(valsWithTail) {
    const tail = valsWithTail[valsWithTail.length - 1];
    const vals = valsWithTail.slice(0, -1);
    const total = vals.reduce((a, v) => a + v, 0) + tail;
    const { W } = L;
    const GAP = 2;
    let xpos = 0;
    let seen = false;
    const nVisible = vals.filter((v) => v > 0).length;
    const avail = Math.max(0, W - GAP * Math.max(0, nVisible - 1));
    vals.forEach((v, i) => {
      if (v > 0 && seen) xpos += GAP;
      if (v > 0) seen = true;
      const w = total > 0 ? (v / total) * avail : 0;
      const r = refs.segs[i];
      r.rect.setAttribute("x", r2(xpos));
      r.rect.setAttribute("width", r2(Math.max(0, w)));
      r.grp.style.display = w > 0.1 ? "" : "none";
      xpos += w;
    });
  }

  const size = sizer(el, render);
  return {
    update(next = {}) {
      o = { ...o, ...next };
      size.invalidate();
    },
    destroy() {
      stop();
      size.destroy();
      shell.destroyBase();
    },
  };
}

/* ================================================================== confusionMatrix */

/**
 * confusionMatrix(el, {tp, fp, tn, fn, labels?:{pos:'Attack', neg:'Benign'}, height?, ariaLabel})
 * Rows = actual class, columns = predicted class. Cells use a single-hue ramp by
 * share of all windows; counts count up and fills cross-fade on update.
 */
export function confusionMatrix(el, opts = {}) {
  let o = { ...opts };
  const shell = createShell(el, o, { cls: "viz-cm" });
  let L = null;
  let refs = null;
  let key = "";
  let shown = null;
  let stop = noop;

  const CELLS = [
    { k: "tp", row: 0, col: 0, name: "True positive", plural: "True positives", desc: (p) => `${p} windows correctly flagged` },
    { k: "fn", row: 0, col: 1, name: "False negative", plural: "False negatives", desc: (p) => `${p} windows missed` },
    { k: "fp", row: 1, col: 0, name: "False positive", plural: "False positives", desc: (_, n) => `${n} windows wrongly flagged (false alarms)` },
    { k: "tn", row: 1, col: 1, name: "True negative", plural: "True negatives", desc: (_, n) => `${n} windows correctly left alone` },
  ];

  function layout(W) {
    const pos = String(o.labels?.pos || "Attack");
    const neg = String(o.labels?.neg || "Benign");
    // long class names: the row-label gutter is capped (the cells keep most of the width) and names are ellipsised
    const rowLabW = Math.ceil(Math.min(Math.max(textWidth(pos, 12, 600), textWidth(neg, 12, 600)), Math.max(48, W * 0.26))) + 12;
    const m = { t: 40, l: 22 + rowLabW, r: 0, b: 2 };
    const gw = Math.max(120, W - m.l - m.r);
    const cellW = (gw - 2) / 2;
    const cellH = o.height ? Math.max(48, (o.height - m.t - m.b - 2) / 2) : clamp(cellW * 0.56, 64, 104);
    const H = Math.round(m.t + cellH * 2 + 2 + m.b);
    // counts: NaN / null -> 0, negatives -> 0
    const cnt = (v) => Math.max(0, Number(v) || 0);
    const vals = { tp: cnt(o.tp), fp: cnt(o.fp), tn: cnt(o.tn), fn: cnt(o.fn) };
    const total = vals.tp + vals.fp + vals.tn + vals.fn;
    return { W, H, m, cellW, cellH, pos, neg, vals, total, rowLabW, big: cellW >= 120 ? 22 : 18 };
  }

  function build() {
    const { W, H, m, cellW, cellH, pos, neg, rowLabW } = L;
    shell.size(W, H);
    const g = shell.root;
    g.replaceChildren();
    sv("text", { class: "viz-axis-label", x: r2(m.l + cellW + 1), y: 12, "text-anchor": "middle", text: "Predicted" }, g);
    fitText({ class: "viz-cm-head", x: r2(m.l + cellW / 2), y: 31, "text-anchor": "middle" }, g, pos, cellW - 8, 12, 600);
    fitText({ class: "viz-cm-head", x: r2(m.l + cellW * 1.5 + 2), y: 31, "text-anchor": "middle" }, g, neg, cellW - 8, 12, 600);
    sv("text", { class: "viz-axis-label", transform: `translate(11 ${r2(m.t + cellH + 1)}) rotate(-90)`, "text-anchor": "middle", text: "Actual" }, g);
    fitText({ class: "viz-cm-head", x: m.l - 10, y: r2(m.t + cellH / 2), dy: "0.32em", "text-anchor": "end" }, g, pos, rowLabW - 10, 12, 600);
    fitText({ class: "viz-cm-head", x: m.l - 10, y: r2(m.t + cellH * 1.5 + 2), dy: "0.32em", "text-anchor": "end" }, g, neg, rowLabW - 10, 12, 600);
    refs = {};
    for (const c of CELLS) {
      const x = m.l + c.col * (cellW + 2);
      const y = m.t + c.row * (cellH + 2);
      const grp = sv("g", { class: "viz-cm-cell", tabindex: 0 }, g);
      const rect = sv("rect", { class: "viz-cm-rect", x: r2(x), y: r2(y), width: r2(cellW), height: r2(cellH), rx: 8, ry: 8 }, grp);
      const cap = sv("text", { class: "viz-cm-cap", x: r2(x + 10), y: r2(y + 17), text: c.name }, grp);
      const cnt = sv("text", { class: "viz-cm-count", x: r2(x + cellW / 2), y: r2(y + cellH * 0.58), "text-anchor": "middle", style: { fontSize: `${L.big}px` } }, grp);
      const pct = sv("text", { class: "viz-cm-pct", x: r2(x + cellW / 2), y: r2(y + cellH * 0.58 + 17), "text-anchor": "middle" }, grp);
      if (cellH < 70) cap.setAttribute("y", r2(y + 15));
      bindMarkTooltip(grp, shell, () => {
        const v = L.vals[c.k];
        return {
          title: c.plural,
          rows: [
            { label: "Windows", value: Math.round(v).toLocaleString("en-US"), color: rampFill(L.total ? v / L.total : 0).swatch, shape: "rect" },
            { label: "Share of all", value: L.total ? `${((v / L.total) * 100).toFixed(1)}%` : "—" },
          ],
          note: c.desc(L.pos, L.neg),
        };
      });
      refs[c.k] = { grp, rect, cnt, pct };
    }
  }

  function paint(v) {
    const total = v.tp + v.fp + v.tn + v.fn;
    for (const c of CELLS) {
      const r = refs[c.k];
      const share = total > 0 ? v[c.k] / total : 0;
      const { mix, fill, fillOpacity } = rampFill(share);
      r.rect.style.fill = fill;
      r.rect.style.fillOpacity = fillOpacity == null ? "" : String(fillOpacity);
      r.grp.classList.toggle("strong", mix >= 52);
      r.cnt.textContent = Math.round(v[c.k]).toLocaleString("en-US");
      r.pct.textContent = total > 0 ? `${(share * 100).toFixed(1)}%` : "—";
      const tv = L.vals[c.k];
      r.grp.setAttribute("aria-label", `${c.plural}: ${Math.round(tv).toLocaleString("en-US")} (${L.total ? ((tv / L.total) * 100).toFixed(1) : "0"}% of windows)`);
    }
  }

  function render(W, mode) {
    stop();
    const before = shown;
    L = layout(W);
    const k = `${W}|${L.H}|${L.pos}|${L.neg}`;
    if (k !== key || !refs) {
      build();
      key = k;
    }
    shell.setAria(o.ariaLabel);
    const to = { ...L.vals };
    if (((mode === "morph" && before) || mode === "intro") && !reducedMotion()) {
      const from = mode === "intro" ? { tp: 0, fp: 0, tn: 0, fn: 0 } : before;
      stop = tween(mode === "intro" ? 800 : 520, (t) => {
        const e = easeOutCubic(t);
        const st = {};
        for (const c of CELLS) st[c.k] = lerp(from[c.k], to[c.k], e);
        paint(st);
        shown = st;
      });
    } else {
      paint(to);
      shown = to;
    }
  }

  const size = sizer(el, render);
  return {
    update(next = {}) {
      o = { ...o, ...next };
      size.invalidate();
    },
    destroy() {
      stop();
      size.destroy();
      shell.destroyBase();
    },
  };
}

/* ================================================================== ring */

/**
 * ring(el, {value 0..1 | null, size=120, stroke=10, label, sublabel, color?, format?, ariaLabel})
 * If `label` is a numeric string (e.g. "0.812") it counts up with the sweep;
 * format(v) overrides the centre text while animating.
 */
export function ring(el, opts = {}) {
  let o = { size: 120, stroke: 10, ...opts };
  const shell = createShell(el, o, { cls: "viz-ring" });
  let refs = null;
  let key = "";
  let shown = null;
  let stop = noop;

  function build() {
    const S = o.size;
    const sw = o.stroke;
    shell.size(S, S);
    const g = shell.root;
    g.replaceChildren();
    const r = (S - sw) / 2 - 1;
    refs = {
      r,
      track: sv("circle", { class: "viz-ring-track", cx: S / 2, cy: S / 2, r: r2(r), style: { strokeWidth: `${sw}px` } }, g),
      arc: sv("path", { class: "viz-ring-arc", style: { strokeWidth: `${sw}px` } }, g),
      label: sv("text", { class: "viz-ring-label", x: S / 2, y: S / 2, "text-anchor": "middle", dy: o.sublabel ? "0.05em" : "0.35em", style: { fontSize: `${Math.round(S * 0.2)}px` } }, g),
      sub: sv("text", { class: "viz-ring-sub", x: S / 2, y: r2(S / 2 + S * 0.15), "text-anchor": "middle", style: { fontSize: `${Math.max(10, Math.round(S * 0.095))}px` } }, g),
    };
    bindMarkTooltip(shell.svg, shell, () => ({
      title: o.ariaLabel || o.sublabel || "Value",
      rows: [{ label: o.sublabel || "Value", value: o.label ?? (Number.isFinite(o.value) ? formatAuto(o.value) : "—"), color: resolveColor(o.color ?? "--accent"), shape: "line" }],
    }));
  }

  function labelFor(v, final) {
    if (o.format && Number.isFinite(v)) return o.format(v);
    const lab = o.label;
    if (lab == null) return Number.isFinite(final) ? formatAuto(v) : "—";
    const str = String(lab);
    const mnum = /^(-?\d+(?:\.(\d+))?)(%?)$/.exec(str.trim());
    if (mnum && Number.isFinite(final) && final > 0 && v !== final) {
      const target = Number(mnum[1]);
      const dec = mnum[2] ? mnum[2].length : 0;
      return (target * (v / final)).toFixed(dec) + mnum[3];
    }
    return str;
  }

  function paint(v) {
    const S = o.size;
    const val = Number.isFinite(v) ? clamp(v, 0, 1) : 0;
    refs.arc.setAttribute("d", val > 0 ? arcPath(S / 2, S / 2, refs.r, 0, val * Math.PI * 2) : "");
    refs.arc.style.stroke = resolveColor(o.color ?? "--accent");
    refs.label.textContent = labelFor(v, o.value);
    refs.sub.textContent = o.sublabel ? truncateText(o.sublabel, Math.max(20, (refs.r - o.stroke / 2) * 2 - 12), Math.max(10, Math.round(S * 0.095)), 560) : "";
    refs.label.setAttribute("dy", o.sublabel ? "0.05em" : "0.35em");
  }

  function render(W, mode) {
    stop();
    const k = `${o.size}|${o.stroke}`;
    if (k !== key || !refs) {
      build();
      key = k;
    }
    shell.setAria(o.ariaLabel || [o.sublabel, o.label].filter(Boolean).join(": ") || "Ring gauge");
    const target = Number.isFinite(o.value) ? clamp(o.value, 0, 1) : null;
    if (target == null) {
      paint(null);
      shown = 0;
      return;
    }
    const from = mode === "intro" ? 0 : mode === "morph" ? shown ?? 0 : target;
    if (from === target || reducedMotion()) {
      paint(target);
      shown = target;
      return;
    }
    stop = tween(mode === "intro" ? 900 : 560, (t) => {
      const v = lerp(from, target, easeOutCubic(t));
      paint(v);
      shown = v;
    });
  }

  const size = sizer(el, render);
  return {
    update(next = {}) {
      o = { ...o, ...next };
      size.invalidate();
    },
    destroy() {
      stop();
      size.destroy();
      shell.destroyBase();
    },
  };
}

/* ================================================================== sparkline */

/** sparkline(el, {values, color?, height=28, format?, ariaLabel}) - no axes; hover shows the value. */
export function sparkline(el, opts = {}) {
  let o = { height: 28, ...opts };
  const shell = createShell(el, o, { cls: "viz-spark" });
  let refs = null;
  let stop = noop;

  function render(W, mode) {
    stop();
    const H = o.height;
    shell.size(W, H);
    // missing values (null / NaN) are skipped but keep their slot, so point i stays point i
    const raw = Array.from(o.values || []);
    const idx = [];
    const vals = [];
    raw.forEach((v, i) => {
      if (typeof v === "number" && Number.isFinite(v)) {
        idx.push(i);
        vals.push(v);
      }
    });
    const color = resolveColor(o.color ?? "--accent");
    const g = shell.root;
    g.replaceChildren();
    shell.setAria(o.ariaLabel || "Trend");
    if (!vals.length) return;
    const [lo, hi] = extent(vals);
    const x = linearScale([0, Math.max(1, raw.length - 1)], [3, W - 5]);
    // a flat series sits mid-height instead of on an axis edge
    const pad = lo === hi ? Math.max(Math.abs(lo) * 0.05, 1e-9) : 0;
    const y = linearScale([lo - pad, hi + pad], [H - 4, 4]);
    const pp = vals.map((v, k) => [x(raw.length === 1 ? 0.5 : idx[k]), y(v)]);
    const d = linePath(pp);
    const area = sv("path", { class: "viz-area", d: `${d}L${r2(pp[pp.length - 1][0])},${H}L${r2(pp[0][0])},${H}Z`, style: { fill: color } }, g);
    const line = sv("path", { class: "viz-line-path viz-spark-line", d, style: { stroke: color } }, g);
    const last = pp[pp.length - 1];
    const dot = sv("circle", { class: "viz-dot", r: 3, cx: r2(last[0]), cy: r2(last[1]), style: { fill: color } }, g);
    refs = { hover: sv("g", null, g) };
    const ov = sv("rect", { class: "viz-overlay", x: 0, y: 0, width: W, height: H }, g);
    const fmtV = o.format || formatAuto;
    const onMove = (e) => {
      const rect = shell.svg.getBoundingClientRect();
      const k = nearestIndex(idx, x.invert(e.clientX - rect.left));
      if (k < 0) return;
      const i = idx[k];
      refs.hover.replaceChildren();
      sv("circle", { class: "viz-dot", r: 3, cx: r2(pp[k][0]), cy: r2(pp[k][1]), style: { fill: color } }, refs.hover);
      showTooltip({ title: o.label || `Point ${i + 1}`, rows: [{ label: o.pointLabel ? o.pointLabel(i) : `#${i + 1}`, value: fmtV(vals[k]), color, shape: "line" }] }, e.clientX, e.clientY, shell);
    };
    ov.addEventListener("pointermove", onMove);
    ov.addEventListener("pointerdown", onMove);
    ov.addEventListener("pointerleave", () => {
      refs.hover.replaceChildren();
      hideTooltip(shell);
    });
    if ((mode === "intro" || mode === "morph") && !reducedMotion()) {
      let len = 0;
      try {
        len = line.getTotalLength();
      } catch {
        len = 0;
      }
      if (len > 0) {
        line.style.strokeDasharray = `${len} ${len}`;
        line.style.strokeDashoffset = String(len);
        area.style.opacity = "0";
        dot.style.opacity = "0";
        stop = tween(
          700,
          (t) => {
            line.style.strokeDashoffset = String(len * (1 - easeInOutCubic(t)));
            area.style.opacity = String(t);
            dot.style.opacity = String(clamp((t - 0.7) / 0.3, 0, 1));
          },
          () => {
            line.style.strokeDasharray = "";
            line.style.strokeDashoffset = "";
            area.style.opacity = "";
            dot.style.opacity = "";
          }
        );
      }
    }
  }

  const size = sizer(el, render);
  return {
    update(next = {}) {
      o = { ...o, ...next };
      size.invalidate();
    },
    destroy() {
      stop();
      size.destroy();
      shell.destroyBase();
    },
  };
}

/* ================================================================== rangeChart */

/**
 * rangeChart(el, {domain:[min,max], log?, intervals:[{label, from, to, color}],
 *   markers:[{label, value, color?}], height?, xLabel, format?, ariaLabel})
 * Horizontal range bars (e.g. the interval a learning rate can reach) with
 * vertical markers (e.g. the true optimum) across all rows.
 */
export function rangeChart(el, opts = {}) {
  let o = { ...opts };
  const shell = createShell(el, o, { cls: "viz-range" });
  let stop = noop;

  function render(W, mode) {
    stop();
    const intervals = (o.intervals || []).map((iv, i) => ({ ...iv, label: String(iv.label ?? `Row ${i + 1}`), color: resolveColor(iv.color ?? "--series-1", 0) }));
    // a marker at zero or below cannot sit on a log axis (it would be pinned to the edge and misread)
    const markers = (o.markers || [])
      .filter((mk) => Number.isFinite(mk.value) && (!o.log || mk.value > 0))
      .map((mk, i) => ({ ...mk, color: mk.color ? resolveColor(mk.color, i) : "var(--text)" }));
    const n = intervals.length;
    const labW = Math.ceil(Math.min(W * 0.36, Math.max(0, ...intervals.map((iv) => textWidth(iv.label, 12, 540))) + 14));
    const fmtV = o.format || (o.log ? formatLogTick : formatAuto);
    const vals = [...intervals.flatMap((iv) => [iv.from, iv.to]), ...markers.map((mk) => mk.value)].filter((v) => typeof v === "number" && Number.isFinite(v));
    let dom = validDomain(o.domain);
    if (dom && o.log && !(dom[0] > 0)) {
      // log axis with a zero / negative lower bound: start one decade below the smallest positive value
      const pos = extent(vals.filter((v) => v > 0));
      dom = niceLogDomain(pos ? pos[0] : dom[1] / 1000, dom[1]);
    }
    if (!dom) {
      if (o.log) {
        const pos = extent(vals.filter((v) => v > 0));
        dom = pos ? niceLogDomain(pos[0], pos[1]) : [1, 10];
      } else {
        const nt = autoDomain(vals, { count: 5, bounds: "none" });
        dom = [nt.min, nt.max];
      }
    }
    const m = { t: 0, r: 16, b: 24 + (o.xLabel ? 18 : 0), l: labW };
    const plotW = Math.max(60, W - m.l - m.r);
    const x = o.log ? logScale(dom, [m.l, m.l + plotW]) : linearScale(dom, [m.l, m.l + plotW]);
    // marker labels: up to two rows at the top to avoid collisions
    const mk = markers
      .map((mm) => ({ ...mm, px: x(mm.value), w: textWidth(mm.label || "", 11, 600) }))
      .sort((a, b) => a.px - b.px);
    let rowEnd = [-Infinity, -Infinity];
    for (const mm of mk) {
      const left = mm.px - mm.w / 2;
      const rowIdx = left > rowEnd[0] + 6 ? 0 : 1;
      mm.row = rowIdx;
      rowEnd[rowIdx] = mm.px + mm.w / 2;
    }
    const rows = mk.some((mm) => mm.row === 1) ? 2 : mk.length ? 1 : 0;
    m.t = 8 + rows * 15;
    const rowH = 30;
    const H = o.height ?? Math.round(m.t + n * rowH + m.b + 4);
    const plotH = Math.max(rowH, H - m.t - m.b);
    const bandH = plotH / Math.max(1, n);
    shell.size(W, H);
    shell.setAria(o.ariaLabel);
    const g = shell.root;
    g.replaceChildren();
    const ticks = o.log ? logTicks(dom[0], dom[1], { maxTicks: Math.max(3, Math.floor(plotW / 64)) }) : ticksInDomain(dom[0], dom[1], Math.max(2, Math.floor(plotW / 80)));
    const tStep = !o.log ? tickStep(ticks) : undefined;
    const tFmt = o.log ? formatLogTick : pickTickFormat(o.format, ticks, tStep);
    drawXTicks(g, { ticks, fmt: tFmt, x, m, plotW, y: m.t + plotH + 16, grid: { when: () => true, y1: m.t, y2: m.t + plotH } });
    sv("line", { class: "viz-axis", x1: m.l, x2: m.l + plotW, y1: crisp(m.t + plotH), y2: crisp(m.t + plotH) }, g);
    if (o.xLabel) sv("text", { class: "viz-axis-label", x: r2(m.l + plotW / 2), y: H - 4, "text-anchor": "middle", text: o.xLabel }, g);
    const bars = [];
    intervals.forEach((iv, i) => {
      const cy = m.t + (i + 0.5) * bandH;
      fitText({ class: "viz-cat", x: m.l - 10, y: r2(cy), dy: "0.32em", "text-anchor": "end" }, g, iv.label, labW - 14, 12, 540);
      const a = x(Math.min(iv.from, iv.to));
      const b = x(Math.max(iv.from, iv.to));
      const grp = sv("g", { class: "viz-bar-g", tabindex: 0, "aria-label": `${iv.label}: ${fmtV(iv.from)} to ${fmtV(iv.to)}` }, g);
      sv("rect", { class: "viz-hit", x: m.l, y: r2(m.t + i * bandH), width: plotW, height: r2(bandH) }, grp);
      const th = Math.min(12, bandH * 0.5);
      const w = Math.max(3, b - a);
      const x0 = w > b - a ? (a + b) / 2 - w / 2 : a;
      const path = sv("path", { class: "viz-bar", style: { fill: iv.color } }, grp);
      bars.push({ path, x0, y0: cy - th / 2, w, th });
      bindMarkTooltip(grp, shell, () => ({
        title: iv.label,
        rows: [
          { label: "From", value: fmtV(iv.from), color: iv.color, shape: "rect" },
          { label: "To", value: fmtV(iv.to) },
        ],
        note: iv.note || "",
      }));
    });
    const mkG = sv("g", { class: "viz-range-markers" }, g);
    for (const mm of mk) {
      const grp = sv("g", { class: "viz-bar-g", tabindex: 0, "aria-label": `${mm.label || "Marker"}: ${fmtV(mm.value)}` }, mkG);
      sv("rect", { class: "viz-hit", x: r2(mm.px - 8), y: m.t - 4, width: 16, height: plotH + 4 }, grp);
      sv("line", { class: "viz-range-marker", x1: r2(mm.px), x2: r2(mm.px), y1: m.t - 2, y2: m.t + plotH, style: { stroke: mm.color } }, grp);
      if (mm.label) {
        sv("text", { class: "viz-marker-label", x: r2(mm.px), y: r2(m.t - 7 - (rows - 1 - mm.row) * 15), "text-anchor": edgeAnchor(mm.px, m.l - labW, m.l + plotW + m.r, mm.w / 2), text: mm.label }, grp);
      }
      bindMarkTooltip(grp, shell, () => ({ title: mm.label || "Marker", rows: [{ label: "Value", value: fmtV(mm.value), color: mm.color, shape: "line" }], note: mm.note || "" }));
    }
    const paintBars = (lt) => {
      bars.forEach((b, i) => {
        const t = lt(i);
        b.path.setAttribute("d", pillPath(b.x0, b.y0, Math.max(t > 0 ? Math.min(b.w, b.th) * t : 0, b.w * t), b.th));
      });
    };
    if ((mode === "intro" || mode === "morph") && !reducedMotion()) {
      const per = 560;
      const gap = Math.min(60, 360 / Math.max(1, bars.length));
      const total = per + gap * Math.max(0, bars.length - 1);
      mkG.style.opacity = "0";
      stop = tween(total, (t) => {
        paintBars((i) => easeOutCubic(clamp((t * total - i * gap) / per, 0, 1)));
        mkG.style.opacity = String(clamp((t - 0.5) / 0.5, 0, 1));
      }, () => (mkG.style.opacity = ""));
    } else paintBars(() => 1);
  }

  const size = sizer(el, render);
  return {
    update(next = {}) {
      o = { ...o, ...next };
      size.invalidate();
    },
    destroy() {
      stop();
      size.destroy();
      shell.destroyBase();
    },
  };
}

/* ================================================================== tables */

/** dataTable(el, {columns:[{key,label,num?,format?}], rows:[...], caption?}) -> {update, destroy} */
export function dataTable(el, opts = {}) {
  let o = { ...opts };
  function render() {
    el.replaceChildren();
    const wrap = hd("div", "table-wrap");
    const table = hd("table", "table");
    if (o.caption) table.appendChild(hd("caption", "sr-only", o.caption));
    const thead = hd("thead");
    const trh = hd("tr");
    for (const c of o.columns || []) {
      const th = hd("th", c.num ? "num" : "", c.label ?? c.key);
      th.setAttribute("scope", "col");
      trh.appendChild(th);
    }
    thead.appendChild(trh);
    const tbody = hd("tbody");
    for (const row of o.rows || []) {
      const tr = hd("tr", row._flagged ? "is-flagged" : "");
      for (const c of o.columns || []) {
        const v = row[c.key];
        const text = c.format ? c.format(v, row) : v == null ? "—" : typeof v === "number" ? formatAuto(v) : String(v);
        tr.appendChild(hd("td", c.num ? "num" : "", text));
      }
      tbody.appendChild(tr);
    }
    table.append(thead, tbody);
    wrap.appendChild(table);
    el.appendChild(wrap);
  }
  render();
  return {
    update(next = {}) {
      o = { ...o, ...next };
      render();
    },
    destroy() {
      el.replaceChildren();
    },
  };
}

/**
 * withTableToggle(container, {render(el) -> chart, table:{columns, rows}, title?})
 * -> {chart, table, showTable(bool), update({table}), destroy()}
 * Shows the chart with a "View as table" / "View as chart" toggle.
 */
export function withTableToggle(container, { render, table, title } = {}) {
  container.replaceChildren();
  container.classList.add("viz-toggle");
  const head = hd("div", "viz-toggle-head");
  if (title) head.appendChild(hd("div", "viz-toggle-title", title));
  else head.appendChild(hd("span"));
  const btn = hd("button", "btn sm ghost viz-toggle-btn");
  btn.type = "button";
  btn.setAttribute("aria-pressed", "false");
  head.appendChild(btn);
  const chartHost = hd("div", "viz-toggle-chart");
  const tableHost = hd("div", "viz-toggle-table");
  tableHost.hidden = true;
  const idBase = `viztt${++uid}`;
  chartHost.id = `${idBase}-chart`;
  tableHost.id = `${idBase}-table`;
  btn.setAttribute("aria-controls", `${chartHost.id} ${tableHost.id}`);
  container.append(head, chartHost, tableHost);
  const chart = render ? render(chartHost) : null;
  let tableSpec = table || { columns: [], rows: [] };
  let tbl = null;
  let showing = false;
  const TABLE_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M3 15h18M9 4v16"/></svg>';
  const CHART_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M22 20H2"/></svg>';
  function sync() {
    btn.innerHTML = showing ? `${CHART_ICON}<span>View as chart</span>` : `${TABLE_ICON}<span>View as table</span>`;
    btn.setAttribute("aria-pressed", showing ? "true" : "false");
    chartHost.hidden = showing;
    tableHost.hidden = !showing;
    const shownHost = showing ? tableHost : chartHost;
    shownHost.classList.remove("viz-swap-in");
    void shownHost.offsetWidth;
    shownHost.classList.add("viz-swap-in");
  }
  function showTable(v) {
    showing = !!v;
    if (showing && !tbl) tbl = dataTable(tableHost, { ...tableSpec, caption: title });
    sync();
  }
  btn.addEventListener("click", () => showTable(!showing));
  sync();
  chartHost.classList.remove("viz-swap-in");
  return {
    chart,
    get table() {
      return tbl;
    },
    showTable,
    update({ table: t } = {}) {
      if (t) {
        tableSpec = { ...tableSpec, ...t };
        if (tbl) tbl.update(tableSpec);
      }
    },
    destroy() {
      if (chart && chart.destroy) chart.destroy();
      if (tbl) tbl.destroy();
      container.replaceChildren();
      container.classList.remove("viz-toggle");
    },
  };
}
