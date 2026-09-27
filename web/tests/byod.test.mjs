// Tests for the pure half of web/js/tabs/byod.js ("score your own data"): the CSV
// parser, header validator, value parsing, availability rule, scoring loop and the
// CSV writers. No DOM is needed: importing byod.js under Node must not touch it.
// Run: node --test "web/tests/*.test.mjs"   (Node 24 does not accept a bare directory)
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import * as byod from "../js/tabs/byod.js";
import { forward } from "../js/forward.js";

const {
  parseCsv,
  sniffDelimiter,
  normalizeColumnName,
  validateHeader,
  parseNumber,
  parseLabel,
  parseRows,
  formatRowError,
  availabilityOf,
  scoreRows,
  csvEscape,
  toCsvLine,
  formatCsvNumber,
  pickTemplateWindows,
  buildTemplateCsv,
  buildScoredCsv,
  buildSampleCsv,
  MAX_ROWS,
  OUTPUT_COLUMNS,
} = byod;

const readJson = (rel) => JSON.parse(readFileSync(new URL(rel, import.meta.url), "utf8"));
const stats = readJson("../data/replica_stats.json");
const demo = readJson("../data/demo_windows.json");
const weights = readJson("../data/weights.json");
const FEATURES = stats.feature_names;
const HEADER = FEATURES.join(",");

/** A valid data line: 42 comma-separated numbers (+ optional extra cells). */
const zeros = (over = {}) => FEATURES.map((f) => (f in over ? over[f] : "0")).join(",");

// ---------------------------------------------------------------------------
// module shape

test("byod.js imports under Node without a DOM and exports mount + the pure helpers", () => {
  assert.equal(typeof globalThis.document, "undefined");
  assert.equal(typeof byod.mount, "function");
  for (const fn of [parseCsv, validateHeader, parseRows, parseNumber, parseLabel, scoreRows, buildTemplateCsv, buildScoredCsv]) {
    assert.equal(typeof fn, "function");
  }
  assert.equal(MAX_ROWS, 50000);
  assert.deepEqual(OUTPUT_COLUMNS, ["score", "verdict", "attribution_m1_log", "attribution_m2_metrics", "attribution_m3_identity", "attribution_m4_graph"]);
});

// ---------------------------------------------------------------------------
// parseCsv

test("parseCsv: plain rows, header trimmed, cells kept verbatim", () => {
  const p = parseCsv(" a , b ,c\n1, 2 ,3\n4,5,6\n");
  assert.equal(p.fatal, null);
  assert.deepEqual(p.header, ["a", "b", "c"]);
  assert.deepEqual(p.rows, [
    ["1", " 2 ", "3"],
    ["4", "5", "6"],
  ]);
  assert.deepEqual(p.rowLines, [2, 3]);
  assert.equal(p.truncated, false);
});

test("parseCsv: CRLF, lone CR, UTF-8 BOM and a missing final newline", () => {
  const p = parseCsv("﻿a,b\r\n1,2\r3,4\r\n5,6");
  assert.deepEqual(p.header, ["a", "b"]);
  assert.deepEqual(p.rows, [
    ["1", "2"],
    ["3", "4"],
    ["5", "6"],
  ]);
  assert.deepEqual(p.rowLines, [2, 3, 4]);
});

test("parseCsv: RFC 4180 quotes - delimiters, doubled quotes and line breaks inside quotes", () => {
  const text = 'id,note,x\r\n"w,1","he said ""hi""",1\r\n"w2","two\r\nlines",2\r\n  "w3"  ,"",3\r\n';
  const p = parseCsv(text);
  assert.equal(p.fatal, null);
  assert.deepEqual(p.rows, [
    ["w,1", 'he said "hi"', "1"],
    ["w2", "two\r\nlines", "2"],
    ["w3", "", "3"],
  ]);
  // the multi-line record starts on line 3; the next one on line 5
  assert.deepEqual(p.rowLines, [2, 3, 5]);
});

test("parseCsv: blank lines and all-empty records are skipped anywhere, including the end", () => {
  const p = parseCsv("\n\na,b\n\n1,2\n , \n,,\n3,4\n\n\n  \n");
  assert.deepEqual(p.header, ["a", "b"]);
  assert.deepEqual(p.rows, [
    ["1", "2"],
    ["3", "4"],
  ]);
  assert.deepEqual(p.rowLines, [5, 8]);
});

test("parseCsv: a trailing delimiter yields an empty last cell", () => {
  const p = parseCsv("a,b,\n1,2,\n3,4,");
  assert.deepEqual(p.header, ["a", "b", ""]);
  assert.deepEqual(p.rows, [
    ["1", "2", ""],
    ["3", "4", ""],
  ]);
});

test("parseCsv: an unterminated quote is a fatal error naming the line it starts on", () => {
  const p = parseCsv('a,b\n1,2\n3,"oops\n4,5\n');
  assert.match(p.fatal, /^Line 3: .*closing quote/);
});

test("parseCsv: empty input and binary input are fatal with plain-language messages", () => {
  assert.match(parseCsv("").fatal, /empty/);
  assert.match(parseCsv("\n \n\r\n").fatal, /empty/);
  assert.match(parseCsv("PK\u0003\u0004\u0014\u0000binary").fatal, /xlsx/);
  assert.match(parseCsv("a,b\n1\u0000,2").fatal, /binary/);
  assert.match(parseCsv(null).fatal, /text/);
});

test("parseCsv: delimiter sniffing (comma default, semicolon, tab) and an explicit delimiter", () => {
  assert.equal(sniffDelimiter("a,b,c\n1,2,3"), ",");
  assert.equal(sniffDelimiter("a;b;c\n1;2;3"), ";");
  assert.equal(sniffDelimiter("a\tb\tc\n"), "\t");
  assert.equal(sniffDelimiter('"x;y",b\n'), ","); // ';' inside quotes does not count
  assert.equal(sniffDelimiter("single"), ",");
  const semi = parseCsv("a;b\n1;2\n");
  assert.equal(semi.delimiter, ";");
  assert.deepEqual(semi.rows, [["1", "2"]]);
  const tab = parseCsv("a\tb\n1\t 2\n");
  assert.deepEqual(tab.rows, [["1", " 2"]]);
  const forced = parseCsv("a;b\n1;2\n", { delimiter: "," });
  assert.deepEqual(forced.header, ["a;b"]);
});

test("parseCsv: stops at maxRows and reports truncation only when more data follows", () => {
  const lines = ["a"];
  for (let i = 0; i < 10; i++) lines.push(String(i));
  assert.equal(parseCsv(lines.join("\n"), { maxRows: 10 }).truncated, false);
  const p = parseCsv(lines.join("\n") + "\n10\n", { maxRows: 10 });
  assert.equal(p.rows.length, 10);
  assert.equal(p.truncated, true);
  // trailing blank lines after exactly maxRows rows are not "more data"
  assert.equal(parseCsv(lines.join("\n") + "\n\n\n", { maxRows: 10 }).truncated, false);
});

// ---------------------------------------------------------------------------
// validateHeader

test("validateHeader: the 42 template names (any order) plus label pass", () => {
  const shuffled = [...FEATURES].reverse();
  const v = validateHeader([...shuffled, "label"], FEATURES);
  assert.equal(v.ok, true);
  assert.deepEqual(v.errors, []);
  assert.equal(v.labelIndex, 42);
  FEATURES.forEach((f, k) => assert.equal(shuffled[v.featureIndex[k]], f));
  assert.deepEqual(v.missing, []);
  assert.deepEqual(v.unknown, []);
});

test("validateHeader: missing columns are listed by name, in pipeline order", () => {
  const cols = FEATURES.filter((f) => f !== "n_flow" && f !== "peer_entropy");
  const v = validateHeader(cols, FEATURES);
  assert.equal(v.ok, false);
  assert.deepEqual(v.missing, ["n_flow", "peer_entropy"]);
  assert.equal(v.errors.length, 1);
  assert.match(v.errors[0], /Missing 2 of the 42 required feature columns: n_flow, peer_entropy/);
});

test("validateHeader: many missing columns are summarised, not dumped", () => {
  const v = validateHeader(["n_events", "foo"], FEATURES);
  assert.equal(v.missing.length, 41);
  assert.match(v.errors[0], /Missing 41 of the 42 .* and 33 more\./);
});

test("validateHeader: unknown columns are kept and reported, not errors", () => {
  const v = validateHeader(["window_id", ...FEATURES, "host", ""], FEATURES);
  assert.equal(v.ok, true);
  assert.deepEqual(v.unknown, ["window_id", "host", "(unnamed column 45)"]);
  assert.equal(v.idIndex, 0);
  assert.equal(v.labelIndex, -1);
  assert.deepEqual(v.keep, [...Array(45).keys()]);
  assert.ok(v.notes.some((n) => /3 extra columns .*window_id, host/.test(n)));
  assert.ok(v.notes.some((n) => /No label column/.test(n)));
});

test("validateHeader: duplicated feature columns are an error", () => {
  const v = validateHeader([...FEATURES, "n_flow", " N_FLOW "], FEATURES);
  assert.equal(v.ok, false);
  assert.deepEqual(v.duplicates, ["n_flow"]);
  assert.equal(v.errors.filter((e) => /n_flow.*more than once/.test(e)).length, 1);
});

test("validateHeader: names match ignoring case, spaces and dashes (and say so)", () => {
  assert.equal(normalizeColumnName("  N Flow "), "n_flow");
  assert.equal(normalizeColumnName("frac-proto.tcp"), "frac_proto_tcp");
  const cols = FEATURES.map((f) => (f === "n_flow" ? "N Flow" : f === "peer_entropy" ? "Peer-Entropy" : f));
  const v = validateHeader([...cols, "LABEL"], FEATURES);
  assert.equal(v.ok, true);
  assert.equal(v.labelIndex, 42);
  assert.deepEqual(v.renamed.map((r) => r.to), ["n_flow", "peer_entropy"]);
  assert.ok(v.notes.some((n) => /Matched ignoring case/.test(n)));
});

test("validateHeader: label aliases, 'label' preferred; output columns from a scored file are replaced", () => {
  const withY = validateHeader([...FEATURES, "y"], FEATURES);
  assert.equal(withY.labelIndex, 42);
  const both = validateHeader(["y", ...FEATURES, "label"], FEATURES);
  assert.equal(both.labelIndex, 43);
  assert.deepEqual(both.unknown, ["y"]);
  const rescored = validateHeader([...FEATURES, "label", ...OUTPUT_COLUMNS], FEATURES);
  assert.equal(rescored.ok, true);
  assert.deepEqual(rescored.replaced, OUTPUT_COLUMNS);
  assert.deepEqual(rescored.keep, [...Array(43).keys()]);
});

test("validateHeader: a numeric first line gets a 'this is data, not a header' message", () => {
  const v = validateHeader(FEATURES.map(() => "1.5"), FEATURES);
  assert.equal(v.ok, false);
  assert.match(v.errors[0], /looks like numbers, not column names/);
  assert.match(validateHeader([], FEATURES).errors[0], /no header row/);
});

// ---------------------------------------------------------------------------
// value parsing

test("parseNumber: strict decimals only", () => {
  for (const [s, v] of [
    ["0", 0],
    [" 12 ", 12],
    ["-3.5", -3.5],
    ["+2", 2],
    [".5", 0.5],
    ["5.", 5],
    ["1e-3", 0.001],
    ["2.5E+4", 25000],
    ["15.939504623413086", 15.939504623413086],
  ]) {
    assert.equal(parseNumber(s), v, s);
  }
  for (const s of ["", "  ", "abc", "0x10", "1,5", "1.2.3", "Infinity", "NaN", "1e400", "1 000", "--1", "e5", null, undefined]) {
    assert.ok(Number.isNaN(parseNumber(s)), String(s));
  }
});

test("parseLabel: 1/0 and common words; blank = unknown; anything else is invalid", () => {
  for (const s of ["1", " 1.0 ", "true", "Attack", "YES", "malicious"]) assert.equal(parseLabel(s), 1, s);
  for (const s of ["0", "0.0", "FALSE", "benign", "no", "Normal"]) assert.equal(parseLabel(s), 0, s);
  for (const s of ["", "  ", null, undefined]) assert.equal(parseLabel(s), null);
  for (const s of ["2", "-1", "maybe", "0.5"]) assert.equal(parseLabel(s), undefined, s);
});

test("availabilityOf: any non-zero value marks a modality present; host identity always present", () => {
  const x = new Array(42).fill(0);
  assert.deepEqual(availabilityOf(x), [0, 0, 1, 0]);
  x[7] = 1; // last M1 column
  x[8] = -2; // first M2 column (non-zero counts, whatever the sign)
  x[41] = 0.001; // last M4 column
  assert.deepEqual(availabilityOf(x), [1, 1, 1, 1]);
  x[32] = 0;
  x[33] = 0;
  assert.equal(availabilityOf(x)[2], 1);
});

test("parseRows: first 5 row errors with line numbers, the rest counted, bad rows skipped", () => {
  const lines = [HEADER + ",label"];
  lines.push(zeros() + ",1"); // line 2 ok
  lines.push(zeros({ n_flow: "abc" }) + ",0"); // line 3 bad number
  lines.push(zeros({ n_dns: "" }) + ","); // line 4 empty feature
  lines.push(zeros() + ",2"); // line 5 bad label
  lines.push(zeros({ n_events: "3,5" }).replace("3,5", '"3,5"') + ",0"); // line 6 decimal comma (quoted)
  lines.push(FEATURES.slice(0, 40).map(() => "0").join(",")); // line 7 too short
  lines.push(zeros() + ",0,extra"); // line 8 too long
  lines.push(zeros() + ",0,,"); // line 9 ok: extra EMPTY trailing cells are tolerated
  lines.push(zeros({ n_flow: "7" }) + ","); // line 10 ok, unlabelled
  const parsed = parseCsv(lines.join("\n"));
  const v = validateHeader(parsed.header, FEATURES);
  const d = parseRows(parsed, v, { featureNames: FEATURES });
  assert.equal(d.n, 3);
  assert.equal(d.nErrorRows, 6);
  assert.equal(d.errors.length, 5);
  assert.deepEqual(
    d.errors.map((e) => [e.line, e.column]),
    [
      [3, "n_flow"],
      [4, "n_dns"],
      [5, "label"],
      [6, "n_events"],
      [7, "peer_reappearance_rate"], // the 40-value row first lacks feature index 40
    ]
  );
  const msgs = d.errors.map(formatRowError);
  assert.equal(msgs[0], "Line 3, n_flow: “abc” is not a number");
  assert.match(msgs[1], /^Line 4, n_dns: empty value/);
  assert.match(msgs[2], /^Line 5, label: “2” is not a label/);
  assert.match(msgs[3], /decimal comma.*3\.5/);
  assert.match(msgs[4], /^Line 7, peer_reappearance_rate: missing \(the row has only 40 of 43 values\)/);
  assert.deepEqual(Array.from(d.labels), [1, 0, -1]);
  assert.deepEqual(Array.from(d.source), [0, 7, 8]);
  assert.equal(d.nLabelled, 2);
  assert.equal(d.nPositive, 1);
  assert.equal(d.X.length, 3 * 42);
  assert.equal(d.X[2 * 42 + FEATURES.indexOf("n_flow")], 7);
  // the too-long row's message (6th error) is counted but not listed
  const all = parseRows(parsed, v, { featureNames: FEATURES, maxErrors: 10 });
  assert.match(formatRowError(all.errors[5]), /^Line 8: this row has 44 values but the header has 43 columns/);
});

test("parseRows: range check against the real data's maxima and known host buckets", () => {
  const lines = [HEADER];
  lines.push(zeros({ host_bucket: "10" }));
  lines.push(zeros({ host_bucket: "47", n_flow: "1e9", frac_ipv6: "2" }));
  lines.push(zeros({ host_bucket: "12", n_events: "-1" }));
  const parsed = parseCsv(lines.join("\n"));
  const v = validateHeader(parsed.header, FEATURES);
  const d = parseRows(parsed, v, { featureNames: FEATURES, featureMeta: stats.feature_meta, knownHostBuckets: stats.hosts.map((x) => x.host_bucket) });
  assert.equal(d.n, 3);
  assert.equal(d.drift.aboveUpperRows, 1);
  assert.equal(d.drift.aboveUpper, 2);
  assert.deepEqual(d.drift.topAbove.map(([k]) => k).sort(), ["frac_ipv6", "n_flow"]);
  assert.equal(d.drift.negativeRows, 1);
  assert.equal(d.drift.unknownHostRows, 1);
});

// ---------------------------------------------------------------------------
// template + scoring round trip against the recorded demo windows

test("template: header is the 42 feature names + label, with 10 synthetic demo rows (blank labels)", () => {
  const csv = buildTemplateCsv(FEATURES, demo.windows);
  const p = parseCsv(csv);
  assert.equal(p.fatal, null);
  assert.deepEqual(p.header, [...FEATURES, "label"]);
  assert.equal(p.rows.length, 10);
  assert.ok(p.rows.every((r) => r.length === 43 && r[42] === ""));
  const picked = pickTemplateWindows(demo.windows);
  assert.equal(picked.length, 10);
  // every flagged demo window (up to 2 of 10) is included so both verdicts appear
  const nFlagged = demo.windows.filter((w) => w.predicted_attack).length;
  assert.equal(picked.filter((w) => w.predicted_attack).length, Math.min(nFlagged, 2));
  assert.deepEqual(picked.map((w) => w.index), [...picked.map((w) => w.index)].sort((a, b) => a - b));
  assert.ok(csv.endsWith("\r\n"));
});

test("template rows round-trip exactly and score like demo_windows.json (same model, same windows)", async () => {
  const picked = pickTemplateWindows(demo.windows);
  const p = parseCsv(buildTemplateCsv(FEATURES, demo.windows));
  const v = validateHeader(p.header, FEATURES);
  assert.equal(v.ok, true);
  const d = parseRows(p, v, { featureNames: FEATURES, featureMeta: stats.feature_meta, knownHostBuckets: stats.hosts.map((x) => x.host_bucket) });
  assert.equal(d.n, 10);
  assert.equal(d.nErrorRows, 0);
  assert.equal(d.nLabelled, 0);
  assert.equal(d.drift.unknownHostRows, 0);
  picked.forEach((w, r) => {
    assert.deepEqual(Array.from(d.X.subarray(r * 42, r * 42 + 42)), w.features, `window ${w.window_id}`);
    const recorded = ["m1_log", "m2_metrics", "m3_identity", "m4_graph"].map((k) => (w.availability[k] ? 1 : 0));
    assert.deepEqual(availabilityOf(w.features), recorded, `availability of ${w.window_id}`);
  });
  const res = await scoreRows(d.X, d.n, weights);
  assert.equal(res.cancelled, false);
  let worst = 0;
  picked.forEach((w, r) => {
    worst = Math.max(worst, Math.abs(res.scores[r] - w.score));
    const attr = ["m1_log", "m2_metrics", "m3_identity", "m4_graph"].map((k) => w.attribution[k]);
    for (let m = 0; m < 4; m++) worst = Math.max(worst, Math.abs(res.attribution[r * 4 + m] - attr[m]));
  });
  assert.ok(worst < 1e-5, `max |score - recorded| = ${worst}`);
});

test("scoreRows equals forward.js row by row, reports monotone progress and can be cancelled", async () => {
  const n = 700;
  const X = new Float64Array(n * 42);
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  for (let i = 0; i < X.length; i++) X[i] = rnd() < 0.4 ? 0 : Math.exp(4 * rnd());
  for (let r = 0; r < n; r++) X[r * 42 + 32] = [10, 34, 47][r % 3];
  const seen = [];
  const res = await scoreRows(X, n, weights, { sliceMs: 0, onProgress: (f, done) => seen.push([f, done]) });
  for (let r = 0; r < n; r += 37) {
    const x = Array.from(X.subarray(r * 42, r * 42 + 42));
    const ref = forward(x, availabilityOf(x), weights);
    assert.equal(res.scores[r], ref.score);
    assert.deepEqual(Array.from(res.attribution.subarray(r * 4, r * 4 + 4)), ref.attribution);
  }
  assert.ok(seen.length > 1, "progress was reported more than once");
  for (let i = 1; i < seen.length; i++) assert.ok(seen[i][1] > seen[i - 1][1]);
  assert.deepEqual(seen[seen.length - 1], [1, n]);
  let calls = 0;
  const cancelled = await scoreRows(X, n, weights, { sliceMs: 0, shouldCancel: () => ++calls >= 2 });
  assert.equal(cancelled.cancelled, true);
  assert.ok(cancelled.done < n);
});

// ---------------------------------------------------------------------------
// CSV writing

test("csvEscape / toCsvLine / formatCsvNumber", () => {
  assert.equal(csvEscape("plain"), "plain");
  assert.equal(csvEscape("a,b"), '"a,b"');
  assert.equal(csvEscape('say "hi"'), '"say ""hi"""');
  assert.equal(csvEscape("two\nlines"), '"two\nlines"');
  assert.equal(csvEscape(" padded "), '" padded "');
  assert.equal(csvEscape("a;b", ";"), '"a;b"');
  assert.equal(csvEscape(null), "");
  assert.equal(toCsvLine(["x", 1, "y,z"]), 'x,1,"y,z"');
  assert.equal(formatCsvNumber(-0), "0");
  assert.equal(formatCsvNumber(0.1 + 0.2), "0.30000000000000004");
  assert.equal(formatCsvNumber(1e-7), "1e-7");
  assert.equal(parseNumber(formatCsvNumber(1e-7)), 1e-7);
  assert.equal(formatCsvNumber(NaN), "");
});

test("buildScoredCsv: original columns verbatim + score, verdict, attribution; bad rows marked; re-parsable", async () => {
  const lines = [["window_id", ...FEATURES, "label", "score"].join(",")];
  lines.push(['"w,1"', zeros({ host_bucket: "10", n_flow: "30" }), "1", "0.9"].join(","));
  lines.push(["w2", zeros({ n_flow: "oops" }), "0", "0.1"].join(","));
  lines.push(["w3", zeros({ host_bucket: "47" }), "", ""].join(","));
  const parsed = parseCsv(lines.join("\r\n"));
  const v = validateHeader(parsed.header, FEATURES);
  assert.deepEqual(v.replaced, ["score"]);
  const d = parseRows(parsed, v, { featureNames: FEATURES });
  const scored = await scoreRows(d.X, d.n, weights);
  const out = buildScoredCsv(parsed, v, d, scored, { threshold: 0.5 });
  const back = parseCsv(out);
  assert.equal(back.fatal, null);
  assert.deepEqual(back.header, ["window_id", ...FEATURES, "label", ...OUTPUT_COLUMNS]);
  assert.equal(back.rows.length, 3);
  assert.equal(back.rows[0][0], "w,1");
  const sCol = back.header.indexOf("score");
  const vCol = back.header.indexOf("verdict");
  assert.equal(back.rows[0][sCol], scored.scores[0].toFixed(6));
  assert.equal(back.rows[0][vCol], scored.scores[0] >= 0.5 ? "attack" : "benign");
  assert.equal(back.rows[1][vCol], "not scored");
  assert.equal(back.rows[1][sCol], "");
  assert.equal(back.rows[2][sCol], scored.scores[1].toFixed(6));
  const attr = [0, 1, 2, 3].map((m) => Number(back.rows[0][sCol + 2 + m]));
  assert.ok(Math.abs(attr.reduce((a, b) => a + b, 0) - 1) < 1e-5);
  // the scored file is itself a valid input: old outputs are replaced, not duplicated
  const again = validateHeader(back.header, FEATURES);
  assert.equal(again.ok, true);
  assert.deepEqual(again.replaced, OUTPUT_COLUMNS);
  // threshold changes the verdict column only
  const strict = parseCsv(buildScoredCsv(parsed, v, d, scored, { threshold: 1 }));
  assert.equal(strict.rows[0][vCol], "benign");
});

test("buildSampleCsv: window_id, host, 42 features, label - and every row validates", () => {
  const corpus = {
    n: 2,
    nFeatures: 42,
    X: new Float32Array(84).map((_, i) => (i % 42 === 32 ? 10 : i % 5 === 0 ? 0.30000001192092896 : 0)),
    y: Uint8Array.from([0, 1]),
    host: Uint8Array.from([2, 2]),
    replica: Uint8Array.from([0, 0]),
    hostIds: ["vpn", "intranet_server", "inet-firewall"],
    replicaIds: ["santos"],
    featureNames: FEATURES,
  };
  const p = parseCsv(buildSampleCsv(corpus));
  assert.deepEqual(p.header, ["window_id", "host", ...FEATURES, "label"]);
  assert.deepEqual(p.rows.map((r) => r[0]), ["synth-santos-inet-firewall-0001", "synth-santos-inet-firewall-0002"]);
  assert.equal(p.rows[0][2], "0.300000012");
  const v = validateHeader(p.header, FEATURES);
  assert.equal(v.ok, true);
  assert.equal(v.idIndex, 0);
  const d = parseRows(p, v, { featureNames: FEATURES });
  assert.equal(d.n, 2);
  assert.deepEqual(Array.from(d.labels), [0, 1]);
});
