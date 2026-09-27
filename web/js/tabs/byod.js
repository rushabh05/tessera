// "Score your own data" panel (Live Detector tab, #byod-mount).
//
// The viewer drops a CSV of 60-second windows in TESSERA's 42-feature format;
// every row is scored in this page by the ACTIVE model (the pretrained real
// model by default, or the viewer's Lab model) through the same hand-written
// forward pass the Live Detector uses (js/forward.js, parity-tested against
// PyTorch: within 4.4e-7 on 40 reference vectors). The file is read with the File API and never leaves the browser.
//
// The first half of this file is pure (CSV parsing, header validation, value
// parsing, scoring, CSV writing) and is exported for web/tests/byod.test.mjs.
// Nothing at module top level touches the DOM, so importing it under Node is safe.

import { forward } from "../forward.js";
import { h, fmt, countUp, stagger, reducedMotion, downloadFile, syncRangeFill, debounce, nextFrame, escapeHtml, toast as domToast } from "../ui/dom.js";
import { icon } from "../ui/icons.js";
import { confusionMatrix, curveChart, histogram as histogramChart, barChart, withTableToggle } from "../ui/charts.js";
import { evaluateAll, metricsAt, histogram as scoreHistogram, MIN_SUPPORT_FOR_RATES } from "../lab/metrics.js";

/* ====================================================================== constants */

export const MAX_ROWS = 50_000;
export const MAX_FILE_BYTES = 64 * 1024 * 1024;
export const N_FEATURES = 42;
export const TEMPLATE_EXAMPLES = 10;
export const TABLE_ROWS = 200;
export const MODALITY_IDS = ["m1_log", "m2_metrics", "m3_identity", "m4_graph"];
export const MODALITY_LABELS = {
  m1_log: "Log templates",
  m2_metrics: "Network metrics",
  m3_identity: "Host identity",
  m4_graph: "Graph structure",
};
/** [start, stop) of each modality inside the 42-vector (M1, M2, M3, M4). */
export const MODALITY_SLICES = [
  [0, 8],
  [8, 32],
  [32, 34],
  [34, 42],
];
/** Accepted names for the optional ground-truth column ("label" preferred). */
export const LABEL_ALIASES = ["label", "y", "is_attack", "target"];
/** Columns this panel appends to the scored download (replaced if already present). */
export const OUTPUT_COLUMNS = ["score", "verdict", ...MODALITY_IDS.map((m) => `attribution_${m}`)];
const ID_COLUMN = /^(window_id|window|id|row_id|name|window_name)$/;

/* ====================================================================== CSV parsing */

/** Column-name normalisation used for matching: trim, lower-case, spaces/dashes/dots -> "_". */
export function normalizeColumnName(name) {
  return String(name ?? "")
    .replace(/^﻿/, "")
    .trim()
    .toLowerCase()
    .replace(/[\s\-.]+/g, "_");
}

/** Guess the delimiter from the first line (comma unless ';' or tab is clearly used). */
export function sniffDelimiter(text) {
  const counts = { ",": 0, ";": 0, "\t": 0 };
  let inQuotes = false;
  const lim = Math.min(text.length, 65536);
  for (let i = 0; i < lim; i++) {
    const c = text[i];
    if (c === '"') inQuotes = !inQuotes;
    else if (!inQuotes && (c === "\n" || c === "\r")) break;
    else if (!inQuotes && c in counts) counts[c]++;
  }
  let best = ",";
  for (const d of [";", "\t"]) if (counts[d] > counts[best]) best = d;
  return best;
}

function countNewlines(src, a, b) {
  let n = 0;
  for (let i = a; i < b; i++) {
    const c = src.charCodeAt(i);
    if (c === 10) n++;
    else if (c === 13 && src.charCodeAt(i + 1) !== 10) n++;
  }
  return n;
}

function looksBinary(text) {
  if (text.startsWith("PK\u0003\u0004")) return "zip";
  const head = text.slice(0, 4096);
  if (head.includes("\u0000")) return "nul";
  return null;
}

/**
 * RFC 4180 CSV parser (plus the usual real-world leniencies).
 * parseCsv(text, {delimiter = auto, maxRows = 50,000}) ->
 *   {header: string[], rows: string[][], rowLines: number[] (1-based line each row starts on),
 *    delimiter, truncated: bool (more data rows than maxRows), fatal: string|null}
 * - quoted fields may contain the delimiter, doubled quotes ("") and line breaks;
 * - CRLF, LF and lone CR line endings; a leading UTF-8 BOM is dropped;
 * - blank lines (and lines whose every cell is empty, e.g. ",,,") are skipped anywhere;
 * - header cells are trimmed; data cells are kept verbatim (value parsers trim).
 */
export function parseCsv(text, { delimiter = null, maxRows = MAX_ROWS } = {}) {
  const out = { header: [], rows: [], rowLines: [], delimiter: ",", truncated: false, fatal: null };
  if (typeof text !== "string") {
    out.fatal = "The file could not be read as text.";
    return out;
  }
  const src = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  const bin = looksBinary(src);
  if (bin) {
    out.fatal =
      bin === "zip"
        ? "This looks like a spreadsheet file (.xlsx), not a CSV. In your spreadsheet app use “Save as” → CSV, then drop that file here."
        : "This does not look like a plain-text CSV (it contains binary data). Export it as a UTF-8 CSV and try again.";
    return out;
  }
  const delim = delimiter ?? sniffDelimiter(src);
  out.delimiter = delim;
  const D = delim.charCodeAt(0);
  const n = src.length;
  let i = 0;
  let line = 1;
  let haveHeader = false;

  while (i < n) {
    const recLine = line;
    const fields = [];
    for (;;) {
      // one field starting at i
      let j = i;
      while (j < n && (src.charCodeAt(j) === 32 || (src.charCodeAt(j) === 9 && D !== 9))) j++;
      if (j < n && src.charCodeAt(j) === 34) {
        const quoteLine = line;
        let value = "";
        let k = j + 1;
        for (;;) {
          const q = src.indexOf('"', k);
          if (q === -1) {
            out.fatal = `Line ${quoteLine}: a quoted value starts here but its closing quote (") is missing.`;
            return out;
          }
          value += src.slice(k, q);
          line += countNewlines(src, k, q);
          if (src.charCodeAt(q + 1) === 34) {
            value += '"';
            k = q + 2;
            continue;
          }
          k = q + 1;
          break;
        }
        // Lenient: characters between the closing quote and the delimiter are kept.
        let m = k;
        while (m < n) {
          const c = src.charCodeAt(m);
          if (c === D || c === 10 || c === 13) break;
          m++;
        }
        const stray = src.slice(k, m);
        if (stray.trim() !== "") value += stray;
        fields.push(value);
        i = m;
      } else {
        let m = i;
        while (m < n) {
          const c = src.charCodeAt(m);
          if (c === D || c === 10 || c === 13) break;
          m++;
        }
        fields.push(src.slice(i, m));
        i = m;
      }
      if (i >= n) break;
      const c = src.charCodeAt(i);
      if (c === D) {
        i++;
        if (i >= n) {
          fields.push("");
          break;
        }
        continue;
      }
      // line break ends the record
      i += c === 13 && src.charCodeAt(i + 1) === 10 ? 2 : 1;
      line++;
      break;
    }
    let blank = true;
    for (const f of fields) {
      if (f.trim() !== "") {
        blank = false;
        break;
      }
    }
    if (blank) continue;
    if (!haveHeader) {
      out.header = fields.map((f) => f.trim());
      haveHeader = true;
      continue;
    }
    if (out.rows.length >= maxRows) {
      out.truncated = true;
      break;
    }
    out.rows.push(fields);
    out.rowLines.push(recLine);
  }
  if (!haveHeader) out.fatal = "The file is empty: there is no header row and no data.";
  return out;
}

/* ====================================================================== header + values */

const NUM_RE = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

/** Strict decimal number (dot decimal, optional exponent). Anything else -> NaN. */
export function parseNumber(raw) {
  if (raw === null || raw === undefined) return NaN;
  const t = String(raw).trim();
  if (!NUM_RE.test(t)) return NaN;
  const v = Number(t);
  return Number.isFinite(v) ? v : NaN;
}

const LABEL_POS = new Set(["1", "1.0", "true", "yes", "attack", "malicious", "anomaly"]);
const LABEL_NEG = new Set(["0", "0.0", "false", "no", "benign", "normal"]);

/** 1 (attack) | 0 (benign) | null (blank = unknown) | undefined (not a label). */
export function parseLabel(raw) {
  const t = String(raw ?? "").trim().toLowerCase();
  if (t === "") return null;
  if (LABEL_POS.has(t)) return 1;
  if (LABEL_NEG.has(t)) return 0;
  return undefined;
}

function listNames(names, max = 8) {
  const shown = names.slice(0, max).join(", ");
  return names.length > max ? `${shown} and ${names.length - max} more` : shown;
}

/**
 * validateHeader(header, featureNames) ->
 * {ok, errors: [plain-language strings], notes: [strings], featureIndex: Int32Array (column of
 *  each feature), labelIndex (-1 if none), labelName, idIndex (-1 if none), missing: [names],
 *  unknown: [names], duplicates: [names], replaced: [names], renamed: [{from, to}],
 *  keep: [column indices carried into the scored download]}
 * Names match ignoring case and spaces; columns may be in any order. Missing or
 * duplicated feature columns are errors; unknown columns are kept and ignored.
 */
export function validateHeader(header, featureNames, { labelAliases = LABEL_ALIASES } = {}) {
  const res = {
    ok: false,
    errors: [],
    notes: [],
    featureIndex: new Int32Array(featureNames.length).fill(-1),
    labelIndex: -1,
    labelName: null,
    idIndex: -1,
    missing: [],
    unknown: [],
    duplicates: [],
    replaced: [],
    renamed: [],
    keep: [],
  };
  const cols = Array.from(header || [], (x) => String(x ?? "").trim());
  if (!cols.length || cols.every((c) => c === "")) {
    res.errors.push("The file has no header row. The first line must list the column names; download the template to see them.");
    return res;
  }
  const want = new Map(featureNames.map((f, k) => [normalizeColumnName(f), k]));
  const norms = cols.map(normalizeColumnName);
  let labelCol = norms.indexOf("label");
  if (labelCol === -1) labelCol = norms.findIndex((nm) => labelAliases.includes(nm));
  const outputs = new Set(OUTPUT_COLUMNS);
  const dupSeen = new Set();
  cols.forEach((name, c) => {
    const nm = norms[c];
    if (want.has(nm)) {
      const k = want.get(nm);
      if (res.featureIndex[k] !== -1) {
        if (!dupSeen.has(k)) res.duplicates.push(featureNames[k]);
        dupSeen.add(k);
        return;
      }
      res.featureIndex[k] = c;
      if (name !== featureNames[k]) res.renamed.push({ from: name, to: featureNames[k] });
      res.keep.push(c);
      return;
    }
    if (c === labelCol) {
      res.labelIndex = c;
      res.labelName = name;
      res.keep.push(c);
      return;
    }
    if (outputs.has(nm)) {
      res.replaced.push(name);
      return;
    }
    res.unknown.push(name || `(unnamed column ${c + 1})`);
    if (res.idIndex === -1 && ID_COLUMN.test(nm)) res.idIndex = c;
    res.keep.push(c);
  });
  res.missing = featureNames.filter((_, k) => res.featureIndex[k] === -1);

  if (res.missing.length) {
    const numeric = cols.filter((c) => Number.isFinite(parseNumber(c))).length;
    if (res.missing.length === featureNames.length && numeric >= cols.length / 2) {
      res.errors.push("The first line looks like numbers, not column names. Add the header row from the template as the first line.");
    } else {
      res.errors.push(
        `Missing ${res.missing.length} of the ${featureNames.length} required feature columns: ${listNames(res.missing)}. ` +
          "Names must match the template (case and spaces are ignored)."
      );
    }
  }
  for (const d of res.duplicates) res.errors.push(`Column “${d}” appears more than once, so it is unclear which one to use.`);

  if (res.renamed.length) {
    res.notes.push(`Matched ignoring case and spaces: ${listNames(res.renamed.map((r) => `“${r.from}” → ${r.to}`), 3)}.`);
  }
  if (res.unknown.length) {
    res.notes.push(
      `${res.unknown.length === 1 ? "1 extra column is" : `${res.unknown.length} extra columns are`} kept in the scored download but not used by the model: ${listNames(res.unknown)}.`
    );
  }
  if (res.replaced.length) {
    res.notes.push(`Columns from an earlier TESSERA run (${listNames(res.replaced)}) will be replaced with fresh values.`);
  }
  if (res.labelIndex === -1) {
    res.notes.push("No label column, so only scores are shown. Add a “label” column (1 = attack, 0 = benign) to also get AP, ROC-AUC and MCC.");
  }
  res.ok = res.errors.length === 0;
  return res;
}

/** Availability of the 4 modalities from a 42-vector: any non-zero value in the slice; M3 always 1. */
export function availabilityOf(x, slices = MODALITY_SLICES) {
  return slices.map(([a, b], m) => {
    if (m === 2) return 1;
    for (let i = a; i < b; i++) if (x[i] !== 0) return 1;
    return 0;
  });
}

function clip(str, max = 24) {
  const s = String(str);
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

/**
 * "Line 8, n_flow: “abc” is not a number". Lines count from the header (line 1), so
 * they equal the row numbers a spreadsheet shows.
 */
export function formatRowError(e) {
  const where = `Line ${e.line ?? e.row + 1}`;
  return e.column ? `${where}, ${e.column}: ${e.message}` : `${where}: ${e.message}`;
}

/**
 * parseRows(parsed, validation, {featureNames, featureMeta?, knownHostBuckets?, maxErrors = 5}) ->
 * {n, X: Float64Array(n*42), labels: Int8Array(n) (-1 = no label), source: Int32Array(n) (index into
 *  parsed.rows), errors: first maxErrors [{row, line, column, value, message}], nErrorRows,
 *  nLabelled, nPositive, drift: {aboveUpper, aboveUpperRows, topAbove: [[name, count]], negative,
 *  negativeRows, unknownHostRows}}
 * A row with any problem is skipped (not scored) and counted.
 */
export function parseRows(parsed, v, { featureNames, featureMeta = null, knownHostBuckets = null, maxErrors = 5 } = {}) {
  const nF = featureNames.length;
  const rows = parsed.rows;
  const N = rows.length;
  const headerLen = parsed.header.length;
  const X = new Float64Array(N * nF);
  const labels = new Int8Array(N);
  const source = new Int32Array(N);
  const errors = [];
  let nErrorRows = 0;
  let n = 0;
  let nLabelled = 0;
  let nPositive = 0;
  const upper = featureMeta ? featureMeta.map((m) => (m.kind === "continuous" && Number.isFinite(m.upper) ? m.upper : Infinity)) : null;
  const hbCol = featureMeta ? featureMeta.findIndex((m) => m.kind === "host_bucket") : -1;
  const hostSet = knownHostBuckets ? new Set(knownHostBuckets) : null;
  const drift = { aboveUpper: 0, aboveUpperRows: 0, topAbove: [], negative: 0, negativeRows: 0, unknownHostRows: 0 };
  const aboveBy = new Map();

  for (let r = 0; r < N; r++) {
    const row = rows[r];
    let err = null;
    if (row.length > headerLen) {
      for (let c = headerLen; c < row.length; c++) {
        if (row[c].trim() !== "") {
          err = { column: null, value: null, message: `this row has ${row.length} values but the header has ${headerLen} columns (is there an unquoted comma inside a value?)` };
          break;
        }
      }
    }
    const base = n * nF;
    if (!err) {
      for (let k = 0; k < nF; k++) {
        const c = v.featureIndex[k];
        const present = c < row.length;
        const raw = present ? row[c] : "";
        const x = parseNumber(raw);
        if (Number.isNaN(x)) {
          const t = String(raw).trim();
          let message;
          if (!present) message = `missing (the row has only ${row.length} of ${headerLen} values)`;
          else if (t === "") message = "empty value (every feature needs a number; use 0 for none)";
          else if (/^[+-]?\d+,\d+$/.test(t)) message = `“${clip(t)}” uses a decimal comma; write it with a dot (${t.replace(",", ".")})`;
          else message = `“${clip(t)}” is not a number`;
          err = { column: featureNames[k], value: raw, message };
          break;
        }
        X[base + k] = x;
      }
    }
    let lab = -1;
    if (!err && v.labelIndex >= 0) {
      const raw = v.labelIndex < row.length ? row[v.labelIndex] : "";
      const l = parseLabel(raw);
      if (l === undefined) err = { column: v.labelName || "label", value: raw, message: `“${clip(String(raw).trim())}” is not a label (use 1 = attack, 0 = benign, or leave it blank)` };
      else if (l !== null) lab = l;
    }
    if (err) {
      nErrorRows++;
      if (errors.length < maxErrors) errors.push({ row: r + 1, line: parsed.rowLines ? parsed.rowLines[r] : null, ...err });
      continue;
    }
    if (upper) {
      let above = false;
      let neg = false;
      for (let k = 0; k < nF; k++) {
        const x = X[base + k];
        if (x < 0) {
          drift.negative++;
          neg = true;
        } else if (x > upper[k] * (1 + 1e-6)) {
          drift.aboveUpper++;
          above = true;
          aboveBy.set(featureNames[k], (aboveBy.get(featureNames[k]) || 0) + 1);
        }
      }
      if (above) drift.aboveUpperRows++;
      if (neg) drift.negativeRows++;
    }
    if (hostSet && hbCol >= 0 && !hostSet.has(X[base + hbCol])) drift.unknownHostRows++;
    labels[n] = lab;
    if (lab >= 0) {
      nLabelled++;
      if (lab === 1) nPositive++;
    }
    source[n] = r;
    n++;
  }
  drift.topAbove = [...aboveBy.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3);
  return {
    n,
    nFeatures: nF,
    X: X.slice(0, n * nF),
    labels: labels.slice(0, n),
    source: source.slice(0, n),
    errors,
    nErrorRows,
    nLabelled,
    nPositive,
    drift,
  };
}

/* ====================================================================== scoring */

const nowMs = () => (typeof performance !== "undefined" && performance.now ? performance.now() : Date.now());

/**
 * Score n rows of X (row-major, nFeatures per row) with forward.js, in time slices that
 * yield to the event loop so the page stays responsive.
 * -> {scores: Float64Array(n), attribution: Float64Array(n*4), cancelled, done}
 */
export async function scoreRows(X, n, weights, { nFeatures = N_FEATURES, onProgress = null, shouldCancel = null, sliceMs = 12 } = {}) {
  const scores = new Float64Array(n);
  const attribution = new Float64Array(n * 4);
  let r = 0;
  while (r < n) {
    const t0 = nowMs();
    do {
      const end = Math.min(n, r + 64);
      for (; r < end; r++) {
        const x = X.subarray(r * nFeatures, (r + 1) * nFeatures);
        const out = forward(x, availabilityOf(x), weights);
        scores[r] = out.score;
        for (let m = 0; m < 4; m++) attribution[r * 4 + m] = out.attribution[m];
      }
    } while (r < n && nowMs() - t0 < sliceMs);
    if (onProgress) onProgress(r / n, r);
    if (r < n) {
      await new Promise((res) => setTimeout(res, 0));
      if (shouldCancel && shouldCancel()) return { scores, attribution, cancelled: true, done: r };
    }
  }
  return { scores, attribution, cancelled: false, done: n };
}

/* ====================================================================== CSV writing */

export function csvEscape(value, delimiter = ",") {
  const s = value === null || value === undefined ? "" : String(value);
  if (s.includes('"') || s.includes(delimiter) || s.includes("\n") || s.includes("\r") || /^\s|\s$/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

export function toCsvLine(cells, delimiter = ",") {
  return cells.map((c) => csvEscape(c, delimiter)).join(delimiter);
}

/** Shortest text that round-trips the number exactly (no -0). */
export function formatCsvNumber(v) {
  if (!Number.isFinite(v)) return "";
  return Object.is(v, -0) ? "0" : String(v);
}

/** The template's example rows: every flagged demo window (up to a quarter) plus the first others, in file order. */
export function pickTemplateWindows(windows, n = TEMPLATE_EXAMPLES) {
  const list = Array.from(windows || []);
  const flagged = list.filter((w) => w.predicted_attack);
  const rest = list.filter((w) => !w.predicted_attack);
  const nFlag = Math.min(flagged.length, Math.max(1, Math.floor(n / 4)));
  const chosen = [...flagged.slice(0, nFlag), ...rest.slice(0, n - nFlag)];
  return chosen.sort((a, b) => (a.index ?? 0) - (b.index ?? 0)).slice(0, n);
}

/**
 * Template CSV: the 42 feature names + "label", then example rows from demo_windows.json
 * (synthetic). The demo windows carry no ground truth, so their label cells are blank.
 */
export function buildTemplateCsv(featureNames, windows, { n = TEMPLATE_EXAMPLES } = {}) {
  const lines = [toCsvLine([...featureNames, "label"])];
  for (const w of pickTemplateWindows(windows, n)) {
    lines.push(toCsvLine([...Array.from(w.features, formatCsvNumber), ""]));
  }
  return lines.join("\r\n") + "\r\n";
}

/**
 * Scored download: every kept original column (verbatim, original order), then score,
 * verdict (at `threshold`, attack iff score >= threshold) and the 4 attribution values.
 * Rows that could not be scored keep their cells with verdict "not scored".
 */
export function buildScoredCsv(parsed, validation, data, scored, { threshold = 0.5 } = {}) {
  const keep = validation.keep;
  const lines = [toCsvLine([...keep.map((c) => parsed.header[c]), ...OUTPUT_COLUMNS])];
  const map = new Int32Array(parsed.rows.length).fill(-1);
  for (let i = 0; i < data.n; i++) map[data.source[i]] = i;
  for (let r = 0; r < parsed.rows.length; r++) {
    const row = parsed.rows[r];
    const cells = keep.map((c) => (c < row.length ? row[c] : ""));
    const i = map[r];
    if (i >= 0) {
      const s = scored.scores[i];
      cells.push(s.toFixed(6), s >= threshold ? "attack" : "benign");
      for (let m = 0; m < 4; m++) cells.push(scored.attribution[i * 4 + m].toFixed(6));
    } else {
      cells.push("", "not scored", "", "", "", "");
    }
    lines.push(toCsvLine(cells));
  }
  return lines.join("\r\n") + "\r\n";
}

/** Labelled synthetic sample (from datagen.js's corpus) as CSV: window_id, host, 42 features, label. */
export function buildSampleCsv(corpus) {
  const nF = corpus.nFeatures || N_FEATURES;
  const lines = [toCsvLine(["window_id", "host", ...corpus.featureNames, "label"])];
  const hostCount = new Map();
  for (let r = 0; r < corpus.n; r++) {
    const host = corpus.hostIds[corpus.host[r]];
    const k = (hostCount.get(host) || 0) + 1;
    hostCount.set(host, k);
    const id = `synth-${corpus.replicaIds[corpus.replica[r]]}-${host}-${String(k).padStart(4, "0")}`;
    const feats = [];
    for (let j = 0; j < nF; j++) feats.push(formatCsvNumber(Number(corpus.X[r * nF + j].toPrecision(9))));
    lines.push(toCsvLine([id, host, ...feats, String(corpus.y[r])]));
  }
  return lines.join("\r\n") + "\r\n";
}

/* ====================================================================== UI */

const VERDICT = {
  attack: `<span class="pill bad">${icon("alert")}Flagged</span>`,
  benign: `<span class="pill good">${icon("check")}Benign</span>`,
};
const OUTCOME = {
  tp: `<span class="pill good">${icon("check")}Caught</span>`,
  tn: `<span class="pill neutral">${icon("check")}Correct</span>`,
  fn: `<span class="pill bad">${icon("x")}Missed</span>`,
  fp: `<span class="pill warn">${icon("alert")}False alarm</span>`,
};

function fmtScore(s) {
  if (!Number.isFinite(s)) return "—";
  if (s < 0.001) return "<0.001";
  if (s > 0.999) return ">0.999";
  return s.toFixed(3);
}
const fmtBytes = (b) => (b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`);

function btn(iconName, text, onClick, cls = "btn sm") {
  return h("button", { type: "button", class: cls, html: `${icon(iconName)}<span>${escapeHtml(text)}</span>`, onClick });
}

function notice(kind, title, items = [], extra = null) {
  const iconName = kind === "bad" || kind === "warn" ? "alert" : kind === "good" ? "check" : "info";
  const body = h("div", {}, title ? h("strong", {}, title) : null);
  if (items.length === 1) body.appendChild(document.createTextNode(title ? ` ${items[0]}` : items[0]));
  else if (items.length) {
    const ul = h("ul", { class: "byod-list" });
    for (const it of items) ul.appendChild(h("li", {}, it));
    body.appendChild(ul);
  }
  if (extra) body.appendChild(extra);
  return h("div", { class: `notice ${kind === "info" ? "" : kind} byod-notice`, role: kind === "bad" ? "alert" : null, html: icon(iconName, { cls: "notice-icon" }) }, body);
}

function statTile({ label, sub = "", primary = false, cls = "" }) {
  const labelEl = h("div", { class: "stat-label" }, label);
  const value = h("div", { class: "stat-value tabular" }, "—");
  const subEl = h("div", { class: "stat-sub" }, sub);
  const el = h("div", { class: `stat byod-tile ${primary ? "primary" : ""} ${cls}` }, labelEl, value, subEl);
  return { el, labelEl, value, sub: subEl, last: 0, cancel: () => {} };
}

/** Set a tile's number: counts up from the value it showed before (instant when animate is false). */
function setTile(tile, v, format, { animate = true } = {}) {
  tile.cancel();
  tile.cancel = () => {};
  if (v === null || v === undefined || !Number.isFinite(v)) {
    tile.value.textContent = "—";
    tile.last = 0;
    return;
  }
  if (animate && v !== tile.last) tile.cancel = countUp(tile.value, v, { from: tile.last, duration: 700, format });
  else tile.value.textContent = format(v);
  tile.last = v;
}

function panel(title, sub, ...children) {
  return h(
    "div",
    { class: "byod-panel" },
    h("div", { class: "byod-panel-head" }, h("div", { class: "card-title" }, title), sub ? h("div", { class: "card-sub" }, sub) : null),
    ...children
  );
}

function thinCurve(pr, max = 2500) {
  const m = pr.recall.length;
  if (m <= max) return { x: pr.recall, y: pr.precision, thresholds: pr.thresholds };
  const step = Math.ceil(m / max);
  const keep = [];
  for (let i = 0; i < m; i += step) keep.push(i);
  if (keep[keep.length - 1] !== m - 1) keep.push(m - 1);
  return { x: keep.map((i) => pr.recall[i]), y: keep.map((i) => pr.precision[i]), thresholds: keep.map((i) => pr.thresholds[i]) };
}

/**
 * Mount the panel into #byod-mount. ctx = {store, navigate, toast, isActive}.
 */
/** 1 -> "1st", 2 -> "2nd", 20 -> "20th", 23 -> "23rd". */
export function ordinal(n) {
  const v = Math.abs(Math.round(n)) % 100;
  const suf = v >= 11 && v <= 13 ? "th" : { 1: "st", 2: "nd", 3: "rd" }[v % 10] || "th";
  return `${Math.round(n)}${suf}`;
}

export async function mount(el, ctx) {
  const store = ctx.store;
  const toast = ctx.toast || domToast;
  el.replaceChildren(
    h("section", { class: "card byod-card", "data-tour": "byod", "aria-busy": "true" }, h("div", { class: "skeleton", style: { height: "22px", width: "40%" } }), h("div", { class: "skeleton", style: { height: "150px" } }))
  );
  let stats;
  let demo;
  try {
    [stats, demo] = await Promise.all([store.loadJSON("data/replica_stats.json"), store.loadJSON("data/demo_windows.json")]);
  } catch (err) {
    el.replaceChildren(h("div", { class: "mount-error", "data-tour": "byod" }, `The “score your own data” panel could not load its data files (${err.message}). The rest of the detector still works.`));
    return null;
  }

  const featureNames = stats.feature_names;
  const featureMeta = stats.feature_meta;
  const hostByBucket = new Map(stats.hosts.map((x) => [x.host_bucket, x.label]));
  const knownHostBuckets = stats.hosts.map((x) => x.host_bucket);
  const hbCol = featureMeta.findIndex((m) => m.kind === "host_bucket");

  const S = { job: 0, threshold: 0.5, sort: "file", logY: false, dataset: null, charts: {}, raf: 0, histDriven: false };
  const R = {};

  /* ---------------------------------------------------------------- static layout */

  R.model = h("div", { class: "byod-model" });

  R.input = h("input", { type: "file", accept: ".csv,text/csv", class: "sr-only", tabindex: "-1", "aria-hidden": "true" });
  R.input.addEventListener("change", () => {
    const f = R.input.files && R.input.files[0];
    R.input.value = "";
    if (f) handleFile(f);
  });
  const openPicker = () => R.input.click();

  R.drop = h("div", {
    class: "byod-drop",
    role: "group",
    "aria-label": "Drop zone for a CSV file",
    onClick: (e) => {
      if (e.target.closest("button, a, input")) return;
      openPicker();
    },
  });

  R.templateBtn = btn("download", "Download template CSV", downloadTemplate);
  R.exampleBtn = btn("play", "Try it with the example file", loadExample, "btn sm primary");
  R.sampleBtn = btn("sparkles", "Labelled synthetic sample", loadSample);

  R.status = h("div", { class: "byod-status", hidden: true });
  R.statusText = h("span", { class: "byod-status-text", "aria-live": "polite" });
  R.statusCount = h("span", { class: "byod-status-count tabular", "aria-hidden": "true" });
  R.fill = h("div", { class: "progress-fill" });
  R.bar = h("div", { class: "progress", role: "progressbar", "aria-label": "Scoring progress", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": "0" }, R.fill);
  R.status.append(h("div", { class: "byod-status-line" }, R.statusText, R.statusCount), R.bar);

  R.messages = h("div", { class: "byod-messages" });
  R.results = h("div", { class: "byod-results", hidden: true, "aria-label": "Scoring results" });

  const columnsHelp = h(
    "details",
    { class: "acc byod-columns" },
    h("summary", {}, "Which columns does the file need?"),
    h(
      "div",
      { class: "acc-body stack-sm" },
      h("p", {}, "One row per 60-second window, one column per feature. Columns can be in any order and names are matched ignoring case and spaces. Any extra columns (an id, a timestamp) are carried into the download untouched."),
      ...stats.modalities.map((m, i) =>
        h(
          "div",
          { class: "byod-mod" },
          h("div", { class: "byod-mod-head" }, h("span", { class: `swatch byod-mod-${i + 1}`, "aria-hidden": "true" }), h("strong", {}, m.label), h("span", { class: "tiny faint" }, `${m.slice[1] - m.slice[0]} columns`)),
          h("div", { class: "byod-chip-list" }, ...featureNames.slice(m.slice[0], m.slice[1]).map((f) => h("code", { class: "byod-chip" }, f)))
        )
      ),
      h("p", {}, h("code", { class: "byod-chip" }, "label"), " (optional): 1 = attack, 0 = benign, blank = unknown. With labels you also get AP, ROC-AUC, MCC and a confusion matrix."),
      h(
        "p",
        { class: "small" },
        "A feature group counts as present when any of its columns is non-zero, the rule this site's aggregate statistics use (the real pipeline records presence directly, so a present group whose values are all zero reads as absent here); host identity is always present. ",
        h("code", {}, "host_bucket"),
        ` is ${stats.hosts.map((x) => `${x.host_bucket} (${x.label})`).join(", ")} for the hosts the model knows.`
      )
    )
  );

  const card = h(
    "section",
    { class: "card byod-card", "data-tour": "byod", "aria-labelledby": "byod-title" },
    h(
      "div",
      { class: "byod-head" },
      h(
        "div",
        { class: "byod-head-main" },
        h("div", { class: "byod-head-icon", html: icon("upload") }),
        h(
          "div",
          { class: "byod-head-text" },
          h("div", { class: "eyebrow" }, "Bring your own data"),
          h("h2", { class: "byod-title", id: "byod-title" }, "Score your own windows (CSV)"),
          h("p", { class: "card-sub" }, "Load a CSV of windows in TESSERA’s 42-feature format and the active model scores every row, right here in the page.")
        )
      ),
      R.model
    ),
    h("p", { class: "byod-privacy", html: `${icon("lock")}<span>Your file is read locally by your browser and never uploaded anywhere. Nothing is stored: reload the page and it is gone.</span>` }),
    R.drop,
    h(
      "div",
      { class: "byod-actions" },
      h("div", { class: "row" }, R.exampleBtn, R.sampleBtn, R.templateBtn),
      h(
        "p",
        { class: "tiny faint" },
        `Example: the first ${TEMPLATE_EXAMPLES} rows of the template, synthetic windows from the list above (unlabelled). Sample: about 1,000 synthetic windows with the generator’s labels, so the metrics appear.`
      )
    ),
    columnsHelp,
    R.status,
    R.messages,
    R.results,
    h(
      "div",
      { class: "card-foot" },
      "Scores come from the model named at the top of this panel, running in this page with the same code as the Live Detector. A model trained on the AIT testbed may not transfer to another network: rows with values outside anything it saw are pointed out after loading."
    ),
    R.input
  );
  el.replaceChildren(card);
  stagger(card);
  renderDrop();
  renderModel();

  // Drag and drop anywhere on the card; the drop zone lights up.
  let dragDepth = 0;
  const hasFiles = (e) => Array.from(e.dataTransfer?.types || []).includes("Files");
  card.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth++;
    R.drop.classList.add("is-dragover");
  });
  card.addEventListener("dragover", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });
  card.addEventListener("dragleave", (e) => {
    if (!hasFiles(e)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) R.drop.classList.remove("is-dragover");
  });
  card.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth = 0;
    R.drop.classList.remove("is-dragover");
    const files = e.dataTransfer.files;
    if (files.length > 1) toast("Several files were dropped; scoring the first one.", { kind: "info" });
    if (files[0]) handleFile(files[0]);
  });

  const onModelChange = debounce(() => {
    renderModel();
    if (S.dataset) rescore();
  }, 60);
  store.on("activeModel", onModelChange);
  store.on("labModel", onModelChange);

  /* ---------------------------------------------------------------- drop zone + model */

  function renderDrop() {
    const ds = S.dataset;
    R.drop.classList.toggle("is-compact", !!ds);
    const choose = h("button", { type: "button", class: ds ? "btn sm" : "btn primary", html: `${icon("upload")}<span>${ds ? "Choose another file" : "Choose file"}</span>`, onClick: openPicker });
    if (!ds) {
      R.drop.replaceChildren(
        h("div", { class: "byod-drop-icon", html: icon("upload") }),
        h("div", { class: "byod-drop-title" }, h("span", { class: "when-idle" }, "Drop a CSV file here"), h("span", { class: "when-over" }, "Release to score it")),
        h("div", { class: "small muted" }, "or pick one from your computer"),
        choose,
        h("div", { class: "hint" }, `Up to ${fmt.int(MAX_ROWS)} rows · the 42 feature columns, plus an optional label column`)
      );
    } else {
      const m = ds.meta;
      R.drop.replaceChildren(
        h("div", { class: "byod-drop-icon sm", html: icon("file") }),
        h(
          "div",
          { class: "byod-file" },
          h("div", { class: "byod-file-name" }, m.name),
          h("div", { class: "hint" }, [m.size != null ? fmtBytes(m.size) : null, `${fmt.int(ds.parsed.rows.length)} data rows`, "drop another file to replace it"].filter(Boolean).join(" · "))
        ),
        choose
      );
    }
    R.drop.firstChild.classList.add("pop");
  }

  function modelName(kind) {
    return kind === "lab" ? "your Lab model" : "the pretrained model";
  }

  function renderModel() {
    const kind = store.getActiveModelKind();
    const lab = store.getLabModel();
    R.model.replaceChildren();
    const text = kind === "lab" ? `Model: your Lab model${lab?.label ? ` (${lab.label})` : ""}` : "Model: pretrained on real AIT data";
    R.model.appendChild(h("span", { class: `badge byod-model-badge is-${kind}`, title: kind === "lab" ? "Trained in your browser on synthetic data" : "Trained on 7 real AIT replicas, santos held out" }, h("span", { class: "dot" }), text));
    if (lab) {
      const seg = h("div", { class: "seg", role: "group", "aria-label": "Model to score with" });
      for (const [k, label] of [
        ["pretrained", "Pretrained"],
        ["lab", "Your Lab model"],
      ]) {
        seg.appendChild(h("button", { type: "button", "aria-pressed": String(kind === k), onClick: () => store.setActiveModelKind(k) }, label));
      }
      R.model.appendChild(seg);
    }
  }

  /* ---------------------------------------------------------------- status + messages */

  function setStatus(text, { progress = null, count = "", state = "running" } = {}) {
    R.status.hidden = false;
    if (text !== null && R.statusText.textContent !== text) R.statusText.textContent = text;
    R.statusCount.textContent = count;
    R.bar.classList.toggle("is-running", state === "running");
    R.bar.classList.toggle("is-done", state === "done");
    R.bar.classList.toggle("is-error", state === "error");
    R.status.dataset.state = state;
    if (progress !== null) {
      const pct = Math.max(0, Math.min(100, progress * 100));
      R.fill.style.width = `${pct}%`;
      R.bar.setAttribute("aria-valuenow", String(Math.round(pct)));
    }
  }

  function showMessages(list) {
    R.messages.replaceChildren(...list);
    stagger(R.messages);
  }

  function fail(title, items, extra = []) {
    S.job++; // cancel anything still running for an earlier file
    setStatus("Could not score this file: see the problem below.", { progress: 1, state: "error" });
    showMessages([notice("bad", title, items), ...extra]);
    clearResults();
    S.dataset = null;
    renderDrop();
  }

  /* ---------------------------------------------------------------- loading */

  async function handleFile(file) {
    if (/\.(xlsx|xls|numbers|ods)$/i.test(file.name)) {
      fail("This is a spreadsheet file, not a CSV.", ["In your spreadsheet app choose “Save as” → CSV (comma separated), then drop the .csv here."]);
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      fail("This file is too large.", [`It is ${fmtBytes(file.size)}; the limit is ${fmtBytes(MAX_FILE_BYTES)} (about ${fmt.int(MAX_ROWS)} rows is plenty for a demo). Split it and score the parts one at a time.`]);
      return;
    }
    const job = ++S.job;
    setStatus(`Reading ${file.name}…`, { progress: 0.02 });
    let text;
    try {
      text = await file.text();
    } catch (err) {
      if (job === S.job) fail("The browser could not read this file.", [err.message || String(err)]);
      return;
    }
    if (job !== S.job) return;
    await loadText(text, { name: file.name, size: file.size, source: "file" }, job);
  }

  async function loadText(text, meta, job = ++S.job) {
    setStatus(`Checking the columns and values in ${meta.name}…`, { progress: 0.05 });
    await nextFrame();
    if (job !== S.job) return;
    const parsed = parseCsv(text, { maxRows: MAX_ROWS });
    if (parsed.fatal) return fail("This file could not be read as a CSV.", [parsed.fatal]);
    const v = validateHeader(parsed.header, featureNames);
    if (!v.ok) return fail("The header row does not match the template.", v.errors, v.notes.length ? [notice("info", "Also noticed", v.notes)] : []);
    if (!parsed.rows.length) return fail("The file has a header but no data rows.", ["Add one row per window under the header, like the template."]);
    const data = parseRows(parsed, v, { featureNames, featureMeta, knownHostBuckets });
    if (data.n === 0) {
      return fail(
        "None of the rows could be scored.",
        data.errors.map(formatRowError).concat(data.nErrorRows > data.errors.length ? [`…and ${fmt.int(data.nErrorRows - data.errors.length)} more rows with problems.`] : [])
      );
    }
    const msgs = [];
    if (parsed.truncated) msgs.push(notice("warn", `Only the first ${fmt.int(MAX_ROWS)} rows are scored.`, [`The file has more rows than the ${fmt.int(MAX_ROWS)}-row limit for this in-browser demo; the rest were not read.`]));
    if (data.nErrorRows) {
      const items = data.errors.map(formatRowError);
      if (data.nErrorRows > items.length) items.push(`…and ${fmt.int(data.nErrorRows - items.length)} more.`);
      const which = data.nErrorRows <= data.errors.length ? "" : ` (the first ${data.errors.length} are listed)`;
      msgs.push(notice("warn", `${fmt.int(data.nErrorRows)} ${data.nErrorRows === 1 ? "row was" : "rows were"} skipped because of a problem${which}:`, items));
    }
    const d = data.drift;
    const drift = [];
    if (d.aboveUpperRows) {
      const k = stats.min_support ?? 20;
      drift.push(
        `${fmt.int(d.aboveUpperRows)} ${d.aboveUpperRows === 1 ? "row has" : "rows have"} values above the ${ordinal(k)}-largest value seen in the real AIT data the model learned from (most often ${d.topAbove.map(([c]) => c).join(", ")}). The bound is not the real maximum, which is never shipped, so a flagged value may still be inside the real range; treat those scores as possible extrapolation.`
      );
    }
    if (d.negativeRows) drift.push(`${fmt.int(d.negativeRows)} ${d.negativeRows === 1 ? "row has" : "rows have"} negative values. Every TESSERA feature is a count, size, entropy or fraction, so negatives are outside what the model has seen.`);
    if (d.unknownHostRows) drift.push(`${fmt.int(d.unknownHostRows)} ${d.unknownHostRows === 1 ? "row has" : "rows have"} a host_bucket the model never saw (it knows ${knownHostBuckets.join(", ")}).`);
    if (drift.length && meta.source === "example") {
      drift.push("Expected for this example: the Live Detector’s demo windows were sampled from per-feature log-normal fits that ignore the real bounds, so a fraction can come out above 1. The check is doing its job.");
    }
    if (drift.length) msgs.push(notice("warn", "Range check: some inputs look unlike the training data.", drift));
    const notes = [...v.notes];
    if (v.labelIndex >= 0 && data.nLabelled === 0 && meta.source === "file") {
      notes.push(`The “${v.labelName}” column is present but every cell is blank, so only scores are shown.`);
    }
    if (notes.length) msgs.push(notice("info", null, notes));
    showMessages(msgs);

    S.dataset = { meta, parsed, v, data, scored: null, eval: null, summary: null };
    clearResults();
    renderDrop();
    await score(job, { fresh: true });
  }

  function loadExample() {
    const text = buildTemplateCsv(featureNames, demo.windows);
    loadText(text, { name: "tessera-example.csv", size: text.length, source: "example" });
  }

  async function loadSample() {
    const job = ++S.job;
    setStatus("Generating about 1,000 synthetic windows from the santos statistics…", { progress: 0.02 });
    try {
      const { generateCorpus } = await import("../lab/datagen.js");
      await nextFrame();
      if (job !== S.job) return;
      const corpus = generateCorpus(stats, { scale: 0.05, seed: 0, replicas: ["santos"] });
      const text = buildSampleCsv(corpus);
      await loadText(text, { name: "tessera-synthetic-santos.csv", size: text.length, source: "sample" }, job);
    } catch (err) {
      console.error(err);
      if (job === S.job) fail("The synthetic sample could not be generated.", [err.message || String(err)]);
    }
  }

  function downloadTemplate() {
    downloadFile("tessera-template.csv", buildTemplateCsv(featureNames, demo.windows), "text/csv");
    toast("Template downloaded: 42 feature columns, an optional label column and 10 synthetic example rows.", { kind: "good" });
  }

  function downloadScored() {
    const ds = S.dataset;
    if (!ds?.scored) return;
    const csv = buildScoredCsv(ds.parsed, ds.v, ds.data, ds.scored, { threshold: S.threshold });
    const base = ds.meta.name.replace(/\.[^.]+$/, "") || "windows";
    downloadFile(`${base}-scored-t${S.threshold.toFixed(2)}.csv`, csv, "text/csv");
    toast(`Downloaded ${fmt.int(ds.parsed.rows.length)} rows with score, verdict (threshold ${S.threshold.toFixed(2)}) and attribution.`, { kind: "good" });
  }

  function clearAll() {
    S.job++;
    S.dataset = null;
    clearResults();
    R.messages.replaceChildren();
    R.status.hidden = true;
    renderDrop();
  }

  /* ---------------------------------------------------------------- scoring */

  async function score(job, { fresh = false } = {}) {
    const ds = S.dataset;
    let model;
    try {
      model = await store.getActiveModel();
      if (!model?.weights) throw new Error("no model weights were returned");
    } catch (err) {
      console.error(err);
      if (job === S.job)
        fail("The model could not be loaded, so nothing was scored.", [
          err?.message || String(err),
          "The pretrained model needs data/weights.json. Serve the site over http (see About), reload, and drop the file again.",
        ]);
      return;
    }
    if (job !== S.job || !ds) return;
    try {
      await scoreWith(job, ds, model, fresh);
    } catch (err) {
      console.error(err);
      if (job === S.job) fail("Scoring failed part-way through.", [err?.message || String(err)]);
    }
  }

  async function scoreWith(job, ds, model, fresh) {
    const n = ds.data.n;
    const who = modelName(model.kind);
    setStatus(`${fresh ? "Scoring" : "Re-scoring"} ${fmt.int(n)} ${n === 1 ? "row" : "rows"} with ${who}…`, { progress: 0.08, count: `0 / ${fmt.int(n)}` });
    R.results.classList.toggle("is-stale", !fresh);
    const t0 = nowMs();
    const res = await scoreRows(ds.data.X, n, model.weights, {
      onProgress: (f, r) => setStatus(null, { progress: 0.08 + 0.9 * f, count: `${fmt.int(r)} / ${fmt.int(n)}` }),
      shouldCancel: () => job !== S.job,
    });
    if (res.cancelled || job !== S.job) return;
    const ms = nowMs() - t0;
    ds.scored = { ...res, model: { kind: model.kind, label: model.label }, ms };
    ds.summary = summarise(ds);
    ds.eval = evaluateLabelled(ds);
    setStatus(`Done: ${fmt.int(n)} ${n === 1 ? "row" : "rows"} scored in ${fmt.ms(ms)} with ${who}.`, { progress: 1, count: "", state: "done" });
    R.results.classList.remove("is-stale");
    if (!R.res) buildResults();
    else paintResults({ first: false });
    if (!fresh && !ctx.isActive?.("detector")) return;
    if (!fresh) toast(`Re-scored ${fmt.int(n)} rows with ${who}.`, { kind: "info" });
  }

  function rescore() {
    const job = ++S.job;
    score(job, { fresh: false });
  }

  function summarise(ds) {
    const { scores, attribution } = ds.scored;
    const n = ds.data.n;
    const X = ds.data.X;
    let sum = 0;
    const attr = [0, 0, 0, 0];
    const present = [0, 0, 0, 0];
    for (let i = 0; i < n; i++) {
      sum += scores[i];
      for (let m = 0; m < 4; m++) attr[m] += attribution[i * 4 + m];
      const a = availabilityOf(X.subarray(i * N_FEATURES, (i + 1) * N_FEATURES));
      for (let m = 0; m < 4; m++) present[m] += a[m];
    }
    const sorted = Float64Array.from(scores).sort();
    const median = n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
    return { mean: sum / n, median, attrMean: attr.map((a) => a / n), presentRate: present.map((p) => p / n) };
  }

  function evaluateLabelled(ds) {
    const { labels, n } = ds.data;
    if (!ds.data.nLabelled) return null;
    const s = ds.scored.scores;
    const y = new Uint8Array(ds.data.nLabelled);
    const sl = new Float64Array(ds.data.nLabelled);
    const unl = [];
    let k = 0;
    for (let i = 0; i < n; i++) {
      if (labels[i] >= 0) {
        y[k] = labels[i];
        sl[k] = s[i];
        k++;
      } else unl.push(s[i]);
    }
    const ev = evaluateAll(y, sl, { threshold: S.threshold });
    return { y, s: sl, ev, unlabelled: Float64Array.from(unl) };
  }

  /* ---------------------------------------------------------------- results */

  function destroyCharts() {
    for (const c of Object.values(S.charts)) c?.destroy?.();
    S.charts = {};
  }

  function clearResults() {
    destroyCharts();
    R.res = null;
    R.results.replaceChildren();
    R.results.hidden = true;
  }

  function sourceTag(source) {
    if (source === "example") return h("span", { class: "tag-synthetic" }, "Synthetic example");
    if (source === "sample") return h("span", { class: "tag-synthetic" }, "Synthetic · generator labels");
    return h("span", { class: "pill neutral" }, h("span", { class: "dot" }), "Your file");
  }

  function buildResults() {
    const ds = S.dataset;
    destroyCharts();
    const labelled = !!ds.eval;
    const rr = (R.res = { labelled });

    const head = h(
      "div",
      { class: "byod-res-head" },
      h("div", { class: "byod-res-title" }, h("h3", {}, "Results"), sourceTag(ds.meta.source)),
      h("div", { class: "row byod-res-actions" }, btn("download", "Download scored CSV", downloadScored, "btn sm primary"), btn("reset", "Clear", clearAll, "btn sm ghost"))
    );
    rr.scoredWith = h("p", { class: "small muted byod-scored-with" });
    rr.notes = h("div", { class: "byod-messages" });

    rr.thrInput = h("input", { type: "range", min: "0", max: "1", step: "0.01", value: String(S.threshold), id: "byod-thr", "aria-describedby": "byod-thr-hint" });
    rr.thrOut = h("output", { class: "byod-thr-value", for: "byod-thr" }, S.threshold.toFixed(2));
    syncRangeFill(rr.thrInput);
    rr.thrInput.addEventListener("input", () => setThreshold(Number(rr.thrInput.value), "slider"));
    const thr = h(
      "div",
      { class: "byod-thr" },
      h("label", { class: "label", for: "byod-thr" }, "Decision threshold"),
      rr.thrInput,
      rr.thrOut,
      h("div", { class: "hint byod-thr-hint", id: "byod-thr-hint" }, "Windows scoring at or above this are flagged as attacks. The project’s default is 0.50; the download uses the value set here.")
    );

    rr.tRows = statTile({ label: "Rows scored" });
    rr.tFlag = statTile({ label: "Flagged" });
    rr.tMean = statTile({ label: "Mean score" });
    rr.tSource = statTile({ label: "Leaned on most", cls: "is-text" });
    const tiles = h("div", { class: "grid-4 byod-tiles" }, rr.tRows.el, rr.tFlag.el, rr.tMean.el, rr.tSource.el);

    const blocks = [head, rr.scoredWith, rr.notes, thr, tiles];

    if (labelled) {
      rr.tAp = statTile({ label: "Average precision (AP)", primary: true });
      rr.tAuc = statTile({ label: "ROC-AUC", sub: "0.5 = coin flip, 1 = perfect ranking" });
      rr.tMcc = statTile({ label: "MCC" });
      rr.tRecall = statTile({ label: "Attacks caught (recall)" });
      rr.acc = h("p", { class: "tiny faint byod-acc-note" });
      rr.cmHost = h("div", { class: "byod-chart" });
      rr.prHost = h("div", { class: "byod-chart" });
      blocks.push(
        h("div", { class: "grid-4 byod-tiles" }, rr.tAp.el, rr.tAuc.el, rr.tMcc.el, rr.tRecall.el),
        rr.acc,
        h(
          "div",
          { class: "grid-2 byod-charts" },
          panel("At this threshold", "Rows are what the labels say, columns what the model says.", rr.cmHost),
          panel("Precision–recall curve", "Every threshold at once; the dot is the one set above.", rr.prHost)
        )
      );
    }

    rr.histHost = h("div", { class: "byod-chart" });
    rr.logInput = h("input", { type: "checkbox" });
    rr.logInput.checked = S.logY;
    rr.logInput.addEventListener("change", () => {
      S.logY = rr.logInput.checked;
      S.charts.hist?.chart?.update({ logY: S.logY });
    });
    rr.attrHost = h("div", { class: "byod-chart" });
    rr.presentList = h("div", { class: "byod-present" });
    blocks.push(
      h(
        "div",
        { class: "grid-2 byod-charts" },
        panel(
          "Score distribution",
          labelled ? "How many labelled benign and attack rows land at each score. Drag the line to move the threshold." : "How many rows land at each score. Drag the line to move the threshold.",
          h("label", { class: "switch byod-log" }, rr.logInput, h("span"), "Log scale (makes rare bars visible)"),
          rr.histHost
        ),
        panel(
          "What the model leaned on",
          "Average gate weight per feature group across the scored rows.",
          rr.attrHost,
          rr.presentList,
          h("p", { class: "tiny faint" }, "A hint, not proof: in a real test these weights understated how much the model needed log templates. The ablation study in ", h("a", { href: "#results" }, "Real Results"), " is the reliable measure.")
        )
      )
    );

    rr.sortSeg = h("div", { class: "seg", role: "group", "aria-label": "Row order" });
    for (const [k, label] of [
      ["file", "File order"],
      ["desc", "Highest score first"],
      ["asc", "Lowest first"],
    ]) {
      rr.sortSeg.appendChild(
        h("button", {
          type: "button",
          "data-sort": k,
          "aria-pressed": String(S.sort === k),
          onClick: () => {
            S.sort = k;
            rr.sortSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.sort === k)));
            renderTable({ animate: true });
          },
        }, label)
      );
    }
    rr.tableNote = h("div", { class: "small muted" });
    rr.thead = h("thead");
    rr.tbody = h("tbody");
    const table = h("table", { class: "table byod-table" }, h("caption", { class: "sr-only" }, "Scored rows"), rr.thead, rr.tbody);
    blocks.push(
      h(
        "div",
        { class: "byod-table-section" },
        h("div", { class: "byod-table-head" }, h("div", {}, h("div", { class: "card-title" }, "Scored rows"), rr.tableNote), rr.sortSeg),
        h("div", { class: "table-wrap byod-table-wrap" }, table)
      )
    );

    R.results.replaceChildren(...blocks);
    stagger(R.results);
    R.results.hidden = false;

    const D = chartData(ds);
    if (labelled) {
      S.charts.cm = confusionMatrix(rr.cmHost, { ...D.cm, labels: { pos: "Attack", neg: "Benign" }, ariaLabel: "Confusion matrix of the labelled rows at the current threshold" });
      if (D.pr) {
        S.charts.pr = curveChart(rr.prHost, { kind: "pr", height: 260, ...D.pr, ariaLabel: "Precision–recall curve of the labelled rows, with the current threshold marked" });
      } else {
        rr.prHost.replaceChildren(h("p", { class: "small muted byod-empty" }, "A precision–recall curve needs at least one attack and one benign label."));
      }
    }
    S.charts.hist = withTableToggle(rr.histHost, {
      render: (host) =>
        histogramChart(host, {
          edges: D.hist.edges,
          series: D.hist.series,
          threshold: S.threshold,
          thresholdLabel: "Threshold",
          logY: S.logY,
          height: 220,
          xLabel: "Attack score",
          ariaLabel: "Histogram of attack scores with the decision threshold",
          onThreshold: (v) => setThreshold(v, "hist"),
        }),
      table: D.hist.table,
    });
    S.charts.attr = barChart(rr.attrHost, {
      bars: D.attrBars,
      horizontal: true,
      yDomain: [0, 1],
      yFormat: (v) => fmt.pct(v, 0),
      valueName: "Average gate weight",
      height: 170,
      ariaLabel: "Average gate weight per feature group",
    });
    paintResults({ first: true, D });
  }

  const pointAt = (m, t) => (m.confusion.tp + m.confusion.fp > 0 ? { x: m.recall, y: m.precision, label: `threshold ${t.toFixed(2)}` } : null);

  /** Everything the charts show, at the current threshold. */
  function chartData(ds) {
    const sm = ds.summary;
    const model = ds.scored.model;
    const D = {
      hist: histogramData(ds),
      attrBars: MODALITY_IDS.map((id, m) => ({ id, label: MODALITY_LABELS[id], value: sm.attrMean[m], note: `Present in ${fmt.pct(sm.presentRate[m], 0)} of rows` })),
      cm: null,
      pr: null,
    };
    if (ds.eval) {
      const m = metricsAt(ds.eval.y, ds.eval.s, S.threshold);
      D.cm = { ...m.confusion };
      const ev = ds.eval.ev;
      if (ev.pr) {
        const c = thinCurve(ev.pr);
        D.pr = {
          series: [{ label: model.kind === "lab" ? "Your Lab model" : "Pretrained model", color: "--series-1", x: c.x, y: c.y, thresholds: c.thresholds, auc: ev.ap }],
          baseline: ev.prevalence,
          point: pointAt(m, S.threshold),
        };
      }
    }
    return D;
  }

  function paintResults({ first, D = null }) {
    const ds = S.dataset;
    const rr = R.res;
    if (!ds?.scored || !rr) return;
    const { n, nErrorRows } = ds.data;
    const sm = ds.summary;
    const model = ds.scored.model;
    rr.scoredWith.replaceChildren(
      `${fmt.int(n)} ${n === 1 ? "row" : "rows"} from `,
      h("strong", {}, ds.meta.name),
      ` scored in ${fmt.ms(ds.scored.ms)} by ${model.kind === "lab" ? `your Lab model (${model.label}), trained in this browser on synthetic data` : "the pretrained model, trained on 7 real AIT replicas (santos held out)"}.`
    );

    // result-level notes
    const notes = [];
    if (ds.meta.source === "example") {
      notes.push(
        notice("info", "Synthetic example.", [
          `These ${n} rows are synthetic windows from the Live Detector list above. They carry no ground-truth labels, so only scores are shown${model.kind === "pretrained" ? " (they match the list, because it is the same model and the same windows)" : ""}. For metrics, try the labelled synthetic sample.`,
        ])
      );
    } else if (ds.meta.source === "sample") {
      const ref = h("span", { class: "byod-ref" });
      notes.push(
        notice(
          "warn",
          "Synthetic sample, generator labels.",
          [
            "These windows were drawn from aggregate statistics of the santos replica (the one the pretrained model never trained on). The label is the class the generator sampled, so this AP says how realistic the generator is for this model, not how well it catches real attacks.",
          ],
          ref
        )
      );
      fillRealReference(ref, model.kind);
    }
    if (ds.eval) {
      const ev = ds.eval.ev;
      if (ev.ap === null) {
        notes.push(notice("info", "AP and ROC-AUC need both classes.", [`Every labelled row is ${ev.nPositive ? "an attack" : "benign"}, so there is nothing to rank; MCC and the confusion matrix are still shown.`]));
      } else if (ev.nPositive < MIN_SUPPORT_FOR_RATES) {
        notes.push(
          notice("warn", `Low support: only ${ev.nPositive} labelled attack ${ev.nPositive === 1 ? "row" : "rows"}.`, [
            `Below ${MIN_SUPPORT_FOR_RATES} positives the project shows AP and MCC but does not trust them (the same rule keeps low-support folds out of its summary means).`,
          ])
        );
      }
      if (ds.eval.unlabelled.length) {
        const nb = ds.eval.unlabelled.length;
        notes.push(notice("info", null, [`Metrics use the ${fmt.int(ds.data.nLabelled)} labelled ${ds.data.nLabelled === 1 ? "row" : "rows"}; ${fmt.int(nb)} ${nb === 1 ? "row has" : "rows have"} a blank label and ${nb === 1 ? "is" : "are"} only scored.`]));
      }
    }
    rr.notes.replaceChildren(...notes);
    if (first) stagger(rr.notes);

    setTile(rr.tRows, n, (v) => fmt.int(v));
    rr.tRows.sub.textContent = nErrorRows ? `${fmt.int(nErrorRows)} skipped with problems` : "every row was valid";
    setTile(rr.tMean, sm.mean, (v) => v.toFixed(3));
    rr.tMean.sub.textContent = `median ${fmtScore(sm.median)} · 0 = benign, 1 = attack`;
    const top = sm.attrMean.indexOf(Math.max(...sm.attrMean));
    rr.tSource.value.textContent = MODALITY_LABELS[MODALITY_IDS[top]];
    rr.tSource.sub.textContent = `${fmt.pct(sm.attrMean[top], 0)} of the gate weight on average`;

    if (ds.eval) {
      const ev = ds.eval.ev;
      setTile(rr.tAp, ev.ap, (v) => v.toFixed(3));
      rr.tAp.sub.replaceChildren(ev.ap === null ? "needs both classes" : `Primary metric · no-skill baseline ${ev.prevalence.toFixed(3)}`);
      if (ev.ap !== null && ev.nPositive < MIN_SUPPORT_FOR_RATES) rr.tAp.sub.append(" ", h("span", { class: "pill warn byod-mini-pill" }, "low support"));
      setTile(rr.tAuc, ev.rocAuc, (v) => v.toFixed(3));
    }

    if (!first) {
      const d = D || chartData(ds);
      if (d.cm) S.charts.cm?.update(d.cm);
      if (d.pr) S.charts.pr?.update(d.pr);
      S.charts.hist.chart.update({ edges: d.hist.edges, series: d.hist.series, threshold: S.threshold });
      S.charts.hist.update({ table: d.hist.table });
      S.charts.attr.update({ bars: d.attrBars });
    }
    rr.presentList.replaceChildren(
      h("span", { class: "tiny faint" }, "Present in"),
      ...MODALITY_IDS.map((id, m) =>
        h("span", { class: "byod-present-item tiny" }, h("span", { class: `swatch byod-mod-${m + 1}`, "aria-hidden": "true" }), h("span", { class: "muted" }, MODALITY_LABELS[id]), h("strong", { class: "tabular" }, fmt.pct(sm.presentRate[m], 0)))
      ),
      h("span", { class: "tiny faint" }, "of rows")
    );

    paintThreshold({ animate: true, charts: false });
    renderTable({ animate: first });
  }

  function histogramData(ds) {
    const s = ds.scored.scores;
    let series;
    let edges;
    if (!ds.eval) {
      const hh = scoreHistogram(new Uint8Array(s.length), s, 20);
      edges = hh.edges;
      series = [{ label: "Windows", color: "--series-1", counts: hh.benign }];
    } else {
      const hl = ds.eval.ev.histogram;
      edges = hl.edges;
      series = [
        { label: "Benign (labelled)", color: "benign", counts: hl.benign },
        { label: "Attack (labelled)", color: "attack", counts: hl.attack },
      ];
      if (ds.eval.unlabelled.length) {
        const hu = scoreHistogram(new Uint8Array(ds.eval.unlabelled.length), ds.eval.unlabelled, 20);
        series.push({ label: "Unlabelled", color: "unused", counts: hu.benign });
      }
    }
    const columns = [{ key: "range", label: "Score range" }, ...series.map((sr, k) => ({ key: `c${k}`, label: sr.label, num: true, format: (v) => fmt.int(v) }))];
    const rows = [];
    for (let b = 0; b < edges.length - 1; b++) {
      const row = { range: `${edges[b].toFixed(2)}–${edges[b + 1].toFixed(2)}` };
      series.forEach((sr, k) => (row[`c${k}`] = sr.counts[b]));
      rows.push(row);
    }
    return { edges, series, table: { columns, rows } };
  }

  async function fillRealReference(el, kind) {
    try {
      const real = await store.loadJSON("data/real_results.json");
      const tb = real?.tessera_base;
      if (!tb || !Number.isFinite(tb.tessera_ap) || !el.isConnected) return;
      el.replaceChildren(
        h("span", { class: "tag-real" }, "Real"),
        ` For comparison, the pretrained model scored AP ${tb.tessera_ap.toFixed(4)} and MCC ${tb.tessera_mcc.toFixed(4)} on the real held-out ${tb.held_out} windows`,
        kind === "lab" ? " (that number is for the pretrained model, not your Lab model)" : "",
        ". ",
        h("span", { class: "faint" }, `Source: ${String(tb.source).split(";")[0]}. `),
        h("a", { href: "#results" }, "See Real Results"),
        "."
      );
      el.classList.add("byod-ref-ready");
    } catch {
      /* the reference line is optional */
    }
  }

  /* ---------------------------------------------------------------- threshold */

  function setThreshold(v, from) {
    if (!Number.isFinite(v)) return;
    S.threshold = Math.round(Math.min(1, Math.max(0, v)) * 100) / 100;
    if (from === "hist") S.histDriven = true;
    if (from !== "slider" && R.res) {
      R.res.thrInput.value = String(S.threshold);
      R.res.thrInput.style.setProperty("--fill", `${S.threshold * 100}%`);
    }
    if (S.raf) return;
    S.raf = requestAnimationFrame(() => {
      S.raf = 0;
      paintThreshold({ animate: false, charts: true });
    });
  }

  /** Everything that depends on the threshold: tiles, confusion matrix, PR dot, histogram line, table verdicts. */
  function paintThreshold({ animate, charts = true }) {
    const ds = S.dataset;
    const rr = R.res;
    if (!ds?.scored || !rr) return;
    const t = S.threshold;
    rr.thrOut.textContent = t.toFixed(2);
    const s = ds.scored.scores;
    let flagged = 0;
    for (let i = 0; i < s.length; i++) if (s[i] >= t) flagged++;
    rr.tFlag.labelEl.textContent = `Flagged (score ≥ ${t.toFixed(2)})`;
    setTile(rr.tFlag, flagged, (v) => fmt.int(v), { animate });
    rr.tFlag.sub.textContent = `${fmt.pct(flagged / s.length, 1)} of rows`;
    if (charts && !S.histDriven) S.charts.hist?.chart?.update({ threshold: t });
    S.histDriven = false;

    if (ds.eval) {
      const m = metricsAt(ds.eval.y, ds.eval.s, t);
      const c = m.confusion;
      setTile(rr.tMcc, m.mcc, (v) => (v < 0 ? `−${Math.abs(v).toFixed(3)}` : v.toFixed(3)), { animate });
      rr.tMcc.sub.textContent = `at threshold ${t.toFixed(2)} · −1 to 1, 0 = chance`;
      setTile(rr.tRecall, m.recall, (v) => fmt.pct(v, 1), { animate });
      rr.tRecall.sub.textContent = `${fmt.int(c.tp)} of ${fmt.int(c.tp + c.fn)} · ${fmt.int(c.fp)} false ${c.fp === 1 ? "alarm" : "alarms"} · precision ${fmt.pct(m.precision, 1)}`;
      if (charts) {
        S.charts.cm?.update({ ...c });
        S.charts.pr?.update({ point: pointAt(m, t) });
      }
      const prev = ds.eval.ev.prevalence;
      rr.acc.textContent = `Accuracy ${fmt.pct(m.accuracy, 1)}, shown small on purpose: always answering “${prev <= 0.5 ? "benign" : "attack"}” would already score ${fmt.pct(Math.max(prev, 1 - prev), 1)} here, which is why the project leads with AP and MCC.`;
    }
    paintVerdicts();
  }

  /* ---------------------------------------------------------------- table */

  function orderIndices(ds) {
    const n = ds.data.n;
    const idx = new Int32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    if (S.sort === "file") return idx;
    const s = ds.scored.scores;
    return S.sort === "desc" ? idx.sort((a, b) => s[b] - s[a] || a - b) : idx.sort((a, b) => s[a] - s[b] || a - b);
  }

  function renderTable({ animate }) {
    const ds = S.dataset;
    const rr = R.res;
    if (!ds?.scored || !rr) return;
    const hasLabel = ds.v.labelIndex >= 0;
    const idCol = ds.v.idIndex;
    const cols = [
      `<th class="num" scope="col" title="Line number in the file (the header is line 1)">Line</th>`,
      idCol >= 0 ? `<th scope="col">${escapeHtml(ds.parsed.header[idCol])}</th>` : "",
      `<th scope="col">Host</th>`,
      `<th scope="col" aria-sort="${S.sort === "desc" ? "descending" : S.sort === "asc" ? "ascending" : "none"}">Score</th>`,
      `<th scope="col">Verdict</th>`,
      hasLabel ? `<th scope="col">Label</th><th scope="col">Outcome</th>` : "",
      `<th scope="col">Main source</th>`,
    ];
    rr.thead.innerHTML = `<tr>${cols.join("")}</tr>`;
    const order = orderIndices(ds);
    const nShow = Math.min(TABLE_ROWS, order.length);
    const s = ds.scored.scores;
    const a = ds.scored.attribution;
    const X = ds.data.X;
    const motion = animate && !reducedMotion();
    const html = [];
    for (let k = 0; k < nShow; k++) {
      const i = order[k];
      const r = ds.data.source[i];
      const row = ds.parsed.rows[r];
      const hb = X[i * N_FEATURES + hbCol];
      const host = hostByBucket.get(hb) || `Bucket ${formatCsvNumber(hb)}`;
      const w = (s[i] * 100).toFixed(1);
      let top = 0;
      for (let m = 1; m < 4; m++) if (a[i * 4 + m] > a[i * 4 + top]) top = m;
      const lab = ds.data.labels[i];
      html.push(
        `<tr class="${motion && k < 24 ? "byod-row-in" : ""}" style="--i:${k}" data-i="${i}">` +
          `<td class="num faint">${ds.parsed.rowLines[r] ?? r + 2}</td>` +
          (idCol >= 0 ? `<td class="mono byod-id" title="${escapeHtml(row[idCol] ?? "")}">${escapeHtml(clip(String(row[idCol] ?? "").trim(), 34))}</td>` : "") +
          `<td>${escapeHtml(host)}</td>` +
          `<td><span class="byod-meter"><span class="byod-meter-track" aria-hidden="true"><span class="byod-meter-fill" data-w="${w}" style="width:${motion ? 0 : w}%"></span></span><span class="byod-meter-val tabular">${fmtScore(s[i])}</span></span></td>` +
          `<td class="byod-verdict"></td>` +
          (hasLabel ? `<td>${lab === 1 ? "Attack" : lab === 0 ? "Benign" : `<span class="faint">—</span>`}</td><td class="byod-outcome"></td>` : "") +
          `<td class="muted">${MODALITY_LABELS[MODALITY_IDS[top]]} <span class="faint tabular">${fmt.pct(a[i * 4 + top], 0)}</span></td>` +
          `</tr>`
      );
    }
    rr.tbody.innerHTML = html.join("");
    const orderText = S.sort === "file" ? "in file order" : S.sort === "desc" ? "highest score first" : "lowest score first";
    rr.tableNote.textContent = nShow < ds.data.n ? `Showing ${fmt.int(nShow)} of ${fmt.int(ds.data.n)} rows, ${orderText}. The download has every row.` : `All ${fmt.int(nShow)} rows, ${orderText}.`;
    paintVerdicts();
    if (motion) {
      nextFrame().then(() =>
        nextFrame().then(() => {
          rr.tbody.querySelectorAll(".byod-meter-fill").forEach((f) => (f.style.width = `${f.dataset.w}%`));
        })
      );
    }
  }

  function paintVerdicts() {
    const ds = S.dataset;
    const rr = R.res;
    if (!ds?.scored || !rr) return;
    const t = S.threshold;
    const s = ds.scored.scores;
    const labels = ds.data.labels;
    for (const tr of rr.tbody.children) {
      const i = Number(tr.dataset.i);
      const flag = s[i] >= t;
      const key = flag ? "attack" : "benign";
      const vc = tr.querySelector(".byod-verdict");
      if (vc.dataset.v !== key) {
        vc.dataset.v = key;
        vc.innerHTML = VERDICT[key];
        tr.querySelector(".byod-meter").classList.toggle("is-flag", flag);
      }
      const oc = tr.querySelector(".byod-outcome");
      if (oc) {
        const lab = labels[i];
        const o = lab < 0 ? "none" : lab === 1 ? (flag ? "tp" : "fn") : flag ? "fp" : "tn";
        if (oc.dataset.o !== o) {
          oc.dataset.o = o;
          oc.innerHTML = o === "none" ? `<span class="faint">—</span>` : OUTCOME[o];
        }
      }
    }
  }

  card.removeAttribute("aria-busy");
  return {
    onShow() {},
    onHide() {},
  };
}
