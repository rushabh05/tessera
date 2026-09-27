// pipeline.js: the whole Training Lab run (synthetic data -> split -> leakage
// certificate -> train -> test -> metrics -> ledger) end to end in Node, on the
// real exported statistics (web/data/replica_stats.json) and the real pretrained
// weights (web/data/weights.json). Every row involved is synthetic.
//
// Run: node --test "web/tests/*.test.mjs"   (Node 24 does not accept a bare directory)

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { webcrypto } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { runPipeline, normaliseConfig, stageWeights, mergeCertificates, STAGES, LEDGER_CAP, LEDGER_SKIPPED_NO_CRYPTO } from "../js/lab/pipeline.js";
import { setSha256 } from "../js/merkle.js";
import { DEFAULT_SPLIT, REPLICA_ORDER, buildSplit, leakageCertificate, loroFolds } from "../js/lab/splits.js";
import { generateCorpus } from "../js/lab/datagen.js";

const here = dirname(fileURLToPath(import.meta.url));
const loadJson = (rel) => JSON.parse(readFileSync(join(here, "..", rel), "utf8"));
const stats = loadJson("data/replica_stats.json");
const pretrainedWeights = loadJson("data/weights.json");

setSha256(async (bytes) => new Uint8Array(await webcrypto.subtle.digest("SHA-256", bytes)));

const STAGE_ORDER = STAGES.map((s) => s.id);
const others = (id) => REPLICA_ORDER.filter((r) => r !== id);

async function run(config, extra = {}) {
  const events = [];
  const result = await runPipeline(config, {
    stats,
    pretrainedWeights: "pretrainedWeights" in extra ? extra.pretrainedWeights : pretrainedWeights,
    ...(extra.sha256 ? { sha256: extra.sha256 } : {}),
    runId: extra.runId ?? "t",
    onEvent: (ev) => {
      events.push(ev);
      if (extra.onEvent) extra.onEvent(ev);
    },
    shouldCancel: extra.shouldCancel ?? null,
  });
  return { result, events };
}

function assertMonotoneProgress(events) {
  const prog = events.filter((e) => e.type === "progress").map((e) => e.overall);
  assert.ok(prog.length > 0, "at least one progress event");
  for (let i = 1; i < prog.length; i++) {
    assert.ok(prog[i] >= prog[i - 1] - 1e-12, `progress went backwards: ${prog[i - 1]} -> ${prog[i]}`);
  }
  for (const p of prog) assert.ok(p >= 0 && p <= 1);
  return prog;
}

/** Stage ids in first-seen order, and every stage opens (running/skipped) before it closes. */
function stageSequence(events) {
  const seq = [];
  for (const e of events) {
    if (e.type !== "stage") continue;
    if (seq[seq.length - 1] !== e.stage) seq.push(e.stage);
  }
  return seq;
}

const isHex64 = (s) => typeof s === "string" && /^[0-9a-f]{64}$/.test(s);

test("default split + TESSERA-base: stages in order, monotone progress, contract-shaped result", async () => {
  const { result: r, events } = await run({
    dataset: { scale: 0.02, seed: 0 },
    split: DEFAULT_SPLIT,
    model: { kind: "tessera", epochs: 3, patience: 5, seed: 0 },
  });

  // events: every one carries runId and tMs
  for (const e of events) {
    assert.equal(e.runId, "t");
    assert.equal(typeof e.tMs, "number");
  }
  assert.deepEqual(stageSequence(events), STAGE_ORDER);
  for (const st of STAGE_ORDER) {
    const evs = events.filter((e) => e.type === "stage" && e.stage === st);
    assert.equal(evs[0].status, "running", `${st} starts running`);
    assert.equal(evs[evs.length - 1].status, "done", `${st} ends done`);
    for (const e of evs) {
      assert.equal(typeof e.detail, "string");
      assert.ok(e.progress >= 0 && e.progress <= 1);
    }
  }
  assert.ok(!events.some((e) => e.type === "error"));
  const prog = assertMonotoneProgress(events);
  assert.equal(prog[prog.length - 1], 1);
  // progress events are throttled (<= ~20/s), plus one forced event per stage end
  const nProgress = events.filter((e) => e.type === "progress").length;
  const nStageEnds = events.filter((e) => e.type === "stage" && e.status !== "running").length;
  assert.ok(nProgress <= r.elapsedMs / 50 + nStageEnds + 2, `${nProgress} progress events in ${r.elapsedMs} ms`);
  assert.equal(events.filter((e) => e.type === "corpus").length, 1);
  assert.equal(events.filter((e) => e.type === "split").length, 1);
  const epochs = events.filter((e) => e.type === "epoch");
  assert.equal(epochs.length, 3);
  assert.deepEqual(epochs.map((e) => e.epoch), [1, 2, 3]);
  for (const e of epochs) {
    assert.ok(Number.isFinite(e.trainLoss));
    assert.ok(e.valAp >= 0 && e.valAp <= 1);
  }
  const resultEvents = events.filter((e) => e.type === "result");
  assert.equal(resultEvents.length, 1);
  assert.equal(resultEvents[0].result, r);
  assert.equal(events[events.length - 1].type, "result");

  // result shape
  assert.equal(r.runId, "t");
  assert.equal(r.mode, "replica");
  assert.equal(r.cancelled, false);
  assert.equal(r.model.kind, "tessera");
  assert.equal(r.model.nParams, 5005);
  assert.ok(r.model.weights && r.model.weights.encoders.length === 4, "weights in weights.json format");
  assert.ok(Array.isArray(r.model.weights.head.linear0.weight[0]), "plain nested arrays");
  assert.equal(r.model.history.epochsRun, 3);
  assert.ok(r.model.trainMs >= 0);
  const t = r.test;
  assert.ok(t.n > 0);
  assert.ok(t.scores instanceof Float32Array && t.scores.length === t.n);
  assert.ok(t.y instanceof Uint8Array && t.y.length === t.n);
  assert.ok(t.replica instanceof Uint8Array && t.host instanceof Uint8Array);
  assert.ok(t.attribution instanceof Float32Array && t.attribution.length === t.n * 4);
  assert.deepEqual(new Set(Array.from(t.replica, (i) => t.replicaIds[i])), new Set(["santos"]));
  assert.equal(typeof r.metrics.ap, "number");
  assert.ok(r.metrics.ap >= 0 && r.metrics.ap <= 1);
  assert.equal(r.metrics.n, t.n);
  assert.deepEqual(r.metricsScope, { kind: "single-split" });
  assert.ok(t.fold instanceof Uint8Array && t.fold.length === t.n && t.fold.every((v) => v === 0));
  assert.ok(t.included instanceof Uint8Array && t.included.length === t.n && t.included.every((v) => v === 1));
  assert.equal(t.nPositive, r.metrics.nPositive);
  assert.equal(r.ledgerStatus, "done");
  assert.ok(r.metrics.pr && r.metrics.roc && r.metrics.histogram && r.metrics.atThreshold);
  assert.deepEqual(r.perReplica.map((p) => p.id), ["santos"]);
  assert.ok(r.perHost.length >= 1 && r.perHost.every((h) => typeof h.label === "string"));
  assert.equal(r.attributionMean.length, 4);
  assert.ok(Math.abs(r.attributionMean.reduce((a, b) => a + b, 0) - 1) < 1e-6, "gates sum to 1");
  assert.ok(r.fidelity && r.fidelity.pretrainedAp >= 0 && r.fidelity.pretrainedAp <= 1);
  assert.equal(typeof r.fidelity.note, "string");
  assert.match(r.fidelity.note, /separates synthetic attacks from synthetic benign windows the way it does on real data/);
  assert.match(r.fidelity.note, /not proof of realism and not a result on real data/);
  assert.ok(Array.isArray(r.certificate.checks));
  assert.equal(r.certificate.checks.find((c) => c.id === "replica-disjoint").severity, "ok");
  assert.equal(r.certificate.checks.find((c) => c.id === "stretch-disjoint").severity, "ok");
  // quiet windows (host identity only) repeat across testbeds; they are reported, not failed
  assert.equal(r.certificate.checks.find((c) => c.id === "duplicate-rows").severity, "ok");
  assert.equal(r.certificate.activeDuplicates, 0);
  assert.equal(r.certificate.overall, "ok");
  assert.equal(r.certificate.support, "ok");
  assert.ok(!r.certificate.checks.some((c) => c.id === "pretrained-contamination"), "only for the pretrained model");
  assert.equal(r.splitSummary.mode, "replica");
  assert.ok(isHex64(r.ledger.rootHex), "ledger root is 64 hex chars");
  assert.equal(r.ledger.nLeaves, Math.min(t.n, LEDGER_CAP));
  assert.equal(r.folds, null);
  assert.equal(r.loroSummary, null);
  for (const st of STAGE_ORDER) assert.equal(typeof r.timings[st], "number");
  assert.ok(r.elapsedMs > 0);
});

test("pretrained model: train stage skipped; contamination check fails on fox, passes on santos", async () => {
  const cfg = (heldOut) => ({
    dataset: { scale: 0.02, seed: 0 },
    split: { mode: "replica", train: others(heldOut), test: [heldOut], valFraction: 0.15, seed: 0 },
    model: { kind: "pretrained" },
    ledger: false,
  });
  const fox = await run(cfg("fox"));
  const trainEvs = fox.events.filter((e) => e.type === "stage" && e.stage === "train");
  assert.deepEqual(trainEvs.map((e) => e.status), ["skipped"]);
  assert.match(trainEvs[0].detail, /pretrained on real AIT data/);
  assert.ok(!fox.events.some((e) => e.type === "epoch"));
  const cFox = fox.result.certificate.checks.find((c) => c.id === "pretrained-contamination");
  assert.ok(cFox, "contamination check present");
  assert.equal(cFox.pass, false);
  assert.equal(cFox.severity, "fail");
  assert.equal(fox.result.model.weights, null);
  assert.equal(fox.result.model.nParams, 5005);
  assert.equal(fox.result.ledger, null);
  assert.equal(fox.events.filter((e) => e.type === "stage" && e.stage === "ledger").at(-1).status, "skipped");
  assert.equal(assertMonotoneProgress(fox.events).at(-1), 1);
  assert.ok(fox.result.fidelity.note.includes("fox"), "fidelity caveat names the contaminated replica");

  const santos = await run(cfg("santos"));
  const cSantos = santos.result.certificate.checks.find((c) => c.id === "pretrained-contamination");
  assert.equal(cSantos.pass, true);
  assert.equal(cSantos.severity, "ok");
  assert.equal(santos.result.fidelity.pretrainedAp, santos.result.metrics.ap);
});

test("random row-level split is flagged by the stretch-disjoint check", async () => {
  const { result } = await run({
    dataset: { scale: 0.02, seed: 0 },
    split: { mode: "random", replicas: [...REPLICA_ORDER], trainPct: 70, valPct: 15, testPct: 15, seed: 0 },
    model: { kind: "pretrained" },
    ledger: false,
  });
  assert.equal(result.mode, "random");
  const c = result.certificate.checks.find((x) => x.id === "stretch-disjoint");
  assert.equal(c.severity, "fail");
  assert.equal(c.pass, false);
  assert.ok(result.certificate.stretchOverlap > 0);
});

test("logistic-regression baseline runs end to end", async () => {
  const { result: r, events } = await run({
    dataset: { scale: 0.02, seed: 0 },
    split: { mode: "chronological", replicas: [...REPLICA_ORDER], trainPct: 70, valPct: 15, testPct: 15 },
    model: { kind: "logreg", seed: 0, logreg: { epochs: 60 } },
  });
  assert.equal(r.cancelled, false);
  assert.equal(r.model.kind, "logreg");
  assert.equal(r.model.nParams, 47);
  assert.equal(r.model.weights, null);
  assert.equal(r.model.history.epochsRun, 60);
  assert.equal(events.filter((e) => e.type === "epoch").length, 60);
  assert.equal(r.test.attribution, null);
  assert.equal(r.attributionMean, null);
  assert.ok(r.metrics.ap >= 0 && r.metrics.ap <= 1);
  assert.ok(typeof r.fidelity.pretrainedAp === "number");
  assert.ok(isHex64(r.ledger.rootHex));
  assert.equal(assertMonotoneProgress(events).at(-1), 1);
});

test("leave-one-replica-out: 8 folds, shaw is low support and excluded from the summary", async () => {
  const { result: r, events } = await run({
    dataset: { scale: 0.02, seed: 0 },
    split: { mode: "loro", replicas: [...REPLICA_ORDER], valFraction: 0.15, seed: 0 },
    model: { kind: "tessera", epochs: 2, seed: 0 },
  });
  assert.equal(r.mode, "loro");
  assert.equal(r.folds.length, 8);
  assert.deepEqual(r.folds.map((f) => f.heldOut), [...REPLICA_ORDER]);
  const foldEvents = events.filter((e) => e.type === "fold");
  assert.equal(foldEvents.length, 8);
  assert.deepEqual(foldEvents.map((e) => e.foldIndex), [0, 1, 2, 3, 4, 5, 6, 7]);
  for (const e of foldEvents) assert.equal(e.nFolds, 8);
  const shaw = r.folds.find((f) => f.heldOut === "shaw");
  assert.equal(shaw.lowSupport, true);
  assert.ok(shaw.nPositive < 20);
  assert.ok(r.loroSummary.excludedLowSupport.includes("shaw"));
  const kept = r.folds.filter((f) => !f.lowSupport);
  assert.equal(r.loroSummary.ap.n, kept.length);
  assert.ok(r.loroSummary.ap.n < 8);
  const keptMean = kept.reduce((a, f) => a + f.ap, 0) / kept.length;
  assert.ok(Math.abs(r.loroSummary.ap.mean - keptMean) < 1e-12);
  assert.ok(r.loroSummary.naiveAp.n >= r.loroSummary.ap.n);
  // pooled out-of-fold predictions cover every generated window once ...
  assert.equal(r.test.n, r.corpus.n);
  // ... but the metrics count only the scoreable folds, exactly like loroSummary
  const excluded = r.folds.filter((f) => f.lowSupport);
  const keptN = kept.reduce((a, f) => a + f.n, 0);
  assert.equal(r.metrics.n, keptN);
  assert.equal(r.metrics.nPositive, kept.reduce((a, f) => a + f.nPositive, 0));
  assert.deepEqual(r.metricsScope, {
    kind: "pooled-folds",
    nFolds: kept.length,
    nModels: kept.length,
    nFoldsTotal: 8,
    excluded: excluded.map((f) => f.heldOut),
    n: keptN,
  });
  assert.deepEqual(r.metricsScope.excluded, r.loroSummary.excludedLowSupport);
  assert.ok(r.test.fold instanceof Uint8Array && r.test.fold.length === r.test.n);
  assert.ok(r.test.included instanceof Uint8Array && r.test.included.length === r.test.n);
  let nIncluded = 0;
  for (let i = 0; i < r.test.n; i++) {
    const f = r.folds[r.test.fold[i]];
    assert.equal(r.test.replicaIds[r.test.replica[i]], f.heldOut, "each row's fold holds out its replica");
    assert.equal(r.test.included[i], f.lowSupport ? 0 : 1);
    nIncluded += r.test.included[i];
  }
  assert.equal(nIncluded, keptN);
  assert.equal(r.test.nPositive, r.folds.reduce((a, f) => a + f.nPositive, 0), "test.nPositive counts every pooled row");
  // per-host rows are the counted ones too
  assert.equal(r.perHost.reduce((a, h) => a + h.n, 0), keptN);
  // the certificate: no fold leaks; shaw's missing support is reported separately
  assert.notEqual(r.certificate.overall, "fail");
  assert.equal(r.certificate.support, "warn");
  assert.equal(r.fidelity, null);
  assert.equal(r.model.weights, null);
  assert.equal(r.splitSummary.nFolds, 8);
  assert.ok(isHex64(r.ledger.rootHex));
  assert.equal(r.ledger.nLeaves, Math.min(r.test.n, LEDGER_CAP));
  // progress is monotone across the folds and repeated stages carry foldIndex
  assert.equal(assertMonotoneProgress(events).at(-1), 1);
  const splitDone = events.filter((e) => e.type === "stage" && e.stage === "split" && e.status === "done");
  assert.deepEqual(splitDone.map((e) => e.foldIndex), [0, 1, 2, 3, 4, 5, 6, 7]);
  const trainProgress = events.filter((e) => e.type === "stage" && e.stage === "train").map((e) => e.progress);
  for (let i = 1; i < trainProgress.length; i++) {
    assert.ok(trainProgress[i] >= trainProgress[i - 1] - 1e-12, "stage progress is cumulative across folds");
  }
  assert.equal(trainProgress.at(-1), 1);
});

test("cancelling mid-training resolves quickly with cancelled = true and no error", async () => {
  let cancelAt = null;
  let epochsSeen = 0;
  const { result, events } = await run(
    { dataset: { scale: 0.02, seed: 0 }, model: { kind: "tessera", epochs: 200, patience: 200 } },
    {
      onEvent: (ev) => {
        if (ev.type === "epoch" && ++epochsSeen === 1) cancelAt = performance.now();
      },
      shouldCancel: () => cancelAt !== null,
    },
  );
  const lag = performance.now() - cancelAt;
  assert.equal(result.cancelled, true);
  assert.equal(result.cancelledAt, "train");
  assert.ok(lag < 1500, `cancel took ${lag.toFixed(0)} ms`);
  assert.ok(!events.some((e) => e.type === "error"));
  const last = events.filter((e) => e.type === "stage").at(-1);
  assert.equal(last.status, "skipped");
  const trainClose = events.filter((e) => e.type === "stage" && e.stage === "train").at(-1);
  assert.equal(trainClose.status, "skipped");
  assert.equal(events.at(-1).type, "result");
  assert.equal(result.metrics, null);
  assert.ok(epochsSeen < 5);
});

test("invalid settings reject with a readable message and an error event", async () => {
  const events = [];
  await assert.rejects(
    runPipeline(
      { split: { mode: "replica", train: ["santos"], test: ["santos"], valFraction: 0.15 } },
      { stats, pretrainedWeights, onEvent: (e) => events.push(e) },
    ),
    /both training and testing/,
  );
  assert.equal(events.filter((e) => e.type === "error").length, 1);
  assert.match(events.find((e) => e.type === "error").message, /cannot run/);
  const { errors } = normaliseConfig({ model: { kind: "nope" }, dataset: { scale: 0 } }, stats);
  assert.equal(errors.length, 2);
});

test("identical settings give an identical run (same ledger root)", async () => {
  const cfg = { dataset: { scale: 0.02, seed: 3 }, model: { kind: "tessera", epochs: 1, seed: 1 } };
  const a = await run(cfg);
  const b = await run(cfg);
  assert.equal(a.result.ledger.rootHex, b.result.ledger.rootHex);
  assert.deepEqual(Array.from(a.result.test.scores), Array.from(b.result.test.scores));
});

// ---------------------------------------------------------------------------
// Nearest-neighbour memoriser, feature sets, stage weights, one-class validation

const allReplicas = (mode) => ({ mode, replicas: [...REPLICA_ORDER], trainPct: 70, valPct: 15, testPct: 15, seed: 0 });

test("stageWeights: every model's shares sum to 100; the weights ride on every stage event", async () => {
  for (const kind of ["tessera", "pretrained", "logreg", "knn"]) {
    const w = stageWeights(kind);
    assert.deepEqual(Object.keys(w), STAGE_ORDER);
    assert.equal(Object.values(w).reduce((a, b) => a + b, 0), 100, kind);
  }
  assert.deepEqual(Object.fromEntries(STAGES.map((st) => [st.id, st.weight])), stageWeights("tessera"));
  assert.equal(stageWeights("pretrained").train, 0, "the pretrained model skips training");
  assert.ok(stageWeights("knn").test > stageWeights("knn").train, "the memoriser's cost is in scoring");
  const w = stageWeights("knn");
  w.test = 0; // a copy: callers cannot change the pipeline's weights
  assert.ok(stageWeights("knn").test > 0);
  const { events } = await run({ dataset: { scale: 0.02, seed: 0 }, model: { kind: "knn" }, ledger: false });
  const plan = events.find((e) => e.type === "plan");
  assert.deepEqual(plan.weights, stageWeights("knn"));
  for (const e of events.filter((x) => x.type === "stage")) assert.deepEqual(e.weights, stageWeights("knn"));
});

test("memoriser (knn): instant fit, scoring reports progress, monotone overall %, no weights or attribution", async () => {
  const { result: r, events } = await run({ dataset: { scale: 0.02, seed: 0 }, split: DEFAULT_SPLIT, model: { kind: "knn" } });
  assert.equal(r.cancelled, false);
  assert.equal(r.model.kind, "knn");
  assert.equal(r.model.k, 5);
  assert.equal(r.model.nParams, null);
  assert.equal(r.model.nStored, r.splitSummary.n.train);
  assert.equal(r.model.weights, null);
  assert.equal(r.model.history, null);
  assert.equal(r.test.attribution, null);
  assert.ok(!events.some((e) => e.type === "epoch"), "nothing to train epoch by epoch");
  assert.deepEqual(stageSequence(events), STAGE_ORDER);
  const trainDone = events.filter((e) => e.type === "stage" && e.stage === "train").at(-1);
  assert.equal(trainDone.status, "done");
  assert.match(trainDone.detail, /Nothing to fit/);
  const testUpdates = events.filter((e) => e.type === "stage" && e.stage === "test" && e.status === "running");
  assert.ok(testUpdates.length >= 1);
  assert.equal(assertMonotoneProgress(events).at(-1), 1);
  assert.ok(r.metrics.ap > r.metrics.prevalence, `AP ${r.metrics.ap} above no-skill ${r.metrics.prevalence}`);
  for (const v of r.test.scores) assert.ok(v >= 0 && v <= 1);
  assert.ok(isHex64(r.ledger.rootHex));
  assert.ok(r.fidelity && Number.isFinite(r.fidelity.pretrainedAp));
  const again = await run({ dataset: { scale: 0.02, seed: 0 }, split: DEFAULT_SPLIT, model: { kind: "knn" } });
  assert.equal(again.result.ledger.rootHex, r.ledger.rootHex, "deterministic");
});

test("memoriser: cancelling while it scores the test set ends with cancelled = true at the test stage", async () => {
  let cancel = false;
  const { result, events } = await run(
    { dataset: { scale: 0.05, seed: 0 }, split: allReplicas("random"), model: { kind: "knn" }, ledger: false },
    { onEvent: (ev) => ev.type === "stage" && ev.stage === "test" && ev.status === "running" && (cancel = true), shouldCancel: () => cancel },
  );
  assert.equal(result.cancelled, true);
  assert.equal(result.cancelledAt, "test");
  assert.ok(!events.some((e) => e.type === "error"));
  assert.equal(events.at(-1).type, "result");
});

test("features 'm2': the certificate compares the 24 network-metric columns; every learnable model runs", async () => {
  for (const kind of ["knn", "logreg", "tessera"]) {
    const { result: r, events } = await run({
      dataset: { scale: 0.02, seed: 0 },
      split: DEFAULT_SPLIT,
      model: { kind, epochs: 2, logreg: { epochs: 30 } },
      features: "m2",
      ledger: false,
    });
    assert.equal(r.cancelled, false, kind);
    assert.equal(r.features, "m2");
    assert.equal(r.config.features, "m2");
    assert.equal(r.model.features, "m2");
    assert.equal(r.certificate.featureSet, "m2");
    assert.ok(r.metrics.ap >= 0 && r.metrics.ap <= 1);
    assert.equal(assertMonotoneProgress(events).at(-1), 1);
    if (kind === "logreg") assert.equal(r.model.nParams, 24 + 4 + 1, "weights for the 24 metric columns only");
    if (kind === "tessera") {
      // only M2 (and the always-present, zeroed M3) can carry gate weight
      assert.ok(r.attributionMean[0] === 0 && r.attributionMean[3] === 0, `gates ${r.attributionMean}`);
    }
  }
  const all = await run({ dataset: { scale: 0.02, seed: 0 }, split: DEFAULT_SPLIT, model: { kind: "knn" }, ledger: false });
  const m2 = await run({ dataset: { scale: 0.02, seed: 0 }, split: DEFAULT_SPLIT, model: { kind: "knn" }, features: "m2", ledger: false });
  assert.equal(all.result.features, "all");
  assert.notDeepEqual(Array.from(all.result.test.scores), Array.from(m2.result.test.scores), "the feature set changes what the model sees");
  assert.ok(m2.result.certificate.testRowsDuplicatingTrain >= all.result.certificate.testRowsDuplicatingTrain, "fewer columns, at least as many exact copies");
  assert.match(normaliseConfig({ features: "m3" }, stats).errors.join(" "), /Unknown feature set/);
});

test("features 'm2' with the pretrained model is overridden to 'all', and the run says so", async () => {
  const { warnings, config } = normaliseConfig({ model: { kind: "pretrained" }, features: "m2" }, stats);
  assert.equal(config.features, "all");
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /all 42 features/);
  const { result: r, events } = await run({ dataset: { scale: 0.02, seed: 0 }, model: { kind: "pretrained" }, features: "m2", ledger: false });
  assert.equal(r.features, "all");
  assert.equal(r.certificate.featureSet, "all");
  assert.deepEqual(r.notes, warnings);
  assert.deepEqual(events.find((e) => e.type === "plan").notes, warnings);
  const train = events.filter((e) => e.type === "stage" && e.stage === "train").at(-1);
  assert.equal(train.status, "skipped");
  assert.match(train.detail, /network-metrics-only setting does not apply/);
  assert.deepEqual(normaliseConfig({ model: { kind: "pretrained" } }, stats).warnings, []);
});

test("a validation set with no attack windows: early stopping uses validation loss, no NaN anywhere", async () => {
  // wheeler's timeline at 2% scale: 40% train (39 attacks), 15% validation (no attacks), 45% test (54 attacks)
  const split = { mode: "chronological", replicas: ["wheeler"], trainPct: 40, valPct: 15, testPct: 45 };
  for (const kind of ["tessera", "logreg"]) {
    const { result: r, events } = await run({ dataset: { scale: 0.02, seed: 0 }, split, model: { kind, epochs: 6, patience: 2, logreg: { epochs: 40 } }, ledger: false });
    assert.equal(events.find((e) => e.type === "split").summary.positives.val, 0, "the premise: no attacks in validation");
    assert.equal(r.cancelled, false);
    const h = r.model.history;
    assert.equal(h.earlyStopMetric, "val_loss", kind);
    assert.deepEqual(h.valAp, []);
    for (const v of h.valLoss) assert.ok(Number.isFinite(v));
    const epochs = events.filter((e) => e.type === "epoch");
    for (const e of epochs) {
      assert.equal(e.valAp, null);
      assert.ok(Number.isFinite(e.valLoss) && Number.isFinite(e.trainLoss));
    }
    const done = events.filter((e) => e.type === "stage" && e.stage === "train").at(-1);
    assert.match(done.detail, /validation AP is undefined/);
    assert.doesNotMatch(done.detail, /NaN|n\/a/);
    assert.ok(Number.isFinite(r.metrics.ap));
  }
  // no validation rows at all: the final epoch is kept
  const none = await run({
    dataset: { scale: 0.02, seed: 0 },
    split: { mode: "chronological", replicas: [...REPLICA_ORDER], trainPct: 80, valPct: 0, testPct: 20 },
    model: { kind: "tessera", epochs: 2 },
    ledger: false,
  });
  assert.equal(none.result.model.history.earlyStopMetric, "none");
  assert.equal(none.result.model.history.bestEpoch, 1);
  assert.match(none.events.filter((e) => e.type === "stage" && e.stage === "train").at(-1).detail, /no validation windows/);
});

test("experiment: the memoriser on network metrics scores higher on a random split than a chronological one", async (t) => {
  // The Lab's headline experiment (scale 0.05, seed 0, all 8 replicas, 70/15/15, M2 only).
  // This is a measurement, not a tuned outcome: if it ever stops holding, report it.
  const t0 = performance.now();
  const out = {};
  for (const mode of ["random", "chronological"]) {
    const { result } = await run({ dataset: { scale: 0.05, seed: 0 }, split: allReplicas(mode), model: { kind: "knn" }, features: "m2", ledger: false });
    out[mode] = result;
  }
  const ms = performance.now() - t0;
  const R = out.random.metrics;
  const C = out.chronological.metrics;
  const msg = `memoriser, M2 only: random AP ${R.ap.toFixed(3)} (prevalence ${R.prevalence.toFixed(3)}, ${R.nPositive} attacks) vs chronological AP ${C.ap.toFixed(3)} (prevalence ${C.prevalence.toFixed(3)}, ${C.nPositive} attacks); ${ms.toFixed(0)} ms for both runs`;
  t.diagnostic(msg);
  assert.equal(out.random.certificate.overall, "fail");
  assert.ok(out.random.certificate.activeDuplicates > 0, "the random split puts active copies of test windows in training");
  assert.ok(R.ap > C.ap, msg);
});

// ---------------------------------------------------------------------------
// Certificate semantics, LORO metrics scope, and robustness without Web Crypto
// or without the pretrained weights

test("LORO at the default size (scale 0.05, seed 0): no leak, support 'warn' naming shaw, metrics over the scoreable folds", async () => {
  const { result: r } = await run({
    dataset: { scale: 0.05, seed: 0 },
    split: { mode: "loro", replicas: [...REPLICA_ORDER], valFraction: 0.15, seed: 0 },
    model: { kind: "knn" },
    ledger: false,
  });
  const cert = r.certificate;
  const ch = (id) => cert.checks.find((c) => c.id === id);
  assert.notEqual(cert.overall, "fail", cert.checks.map((c) => `${c.id}=${c.severity}`).join(", "));
  assert.equal(cert.overall, "ok");
  assert.equal(cert.support, "warn");
  assert.equal(ch("test-support").severity, "warn");
  assert.equal(ch("test-support").group, "support");
  assert.match(ch("test-support").detail, /^shaw has fewer than 20 attack windows/);
  assert.match(ch("test-support").detail, /shaw: 0 attack windows, AP undefined/);
  assert.match(ch("test-support").detail, /7 of 8 folds are summarised/);
  assert.match(ch("test-support").detail, /not a leak/);
  assert.deepEqual(ch("test-support").flaggedFolds, ["shaw"]);
  assert.deepEqual(cert.excludedLowSupport, ["shaw"]);
  assert.equal(cert.lowSupport, false, "7 folds are scoreable");
  // ok-in-every-fold checks carry an aggregate, not the first fold's numbers
  const dup = ch("duplicate-rows");
  assert.equal(dup.severity, "ok");
  assert.doesNotMatch(dup.detail, /^All 8 folds:/);
  const perFoldQuiet = r.folds.map((f) => f.certificate.quietDuplicates);
  const sumQuiet = perFoldQuiet.reduce((a, b) => a + b, 0);
  assert.equal(cert.quietDuplicates, sumQuiet);
  assert.equal(cert.activeDuplicates, 0);
  assert.equal(cert.testRowsDuplicatingTrain, r.folds.reduce((a, f) => a + f.certificate.testRowsDuplicatingTrain, 0));
  assert.equal(cert.nTest, r.test.n);
  assert.match(dup.detail, new RegExp(`^Across the 8 folds, ${sumQuiet.toLocaleString("en")} test windows`));
  assert.match(dup.detail, new RegExp(`${Math.min(...perFoldQuiet)} to ${Math.max(...perFoldQuiet)} per fold`));
  assert.match(ch("replica-disjoint").detail, /^In each of the 8 folds/);
  assert.equal(cert.perFold.find((f) => f.heldOut === "shaw").support, "fail");
  assert.equal(cert.perFold.find((f) => f.heldOut === "shaw").overall, "ok");
  // metrics: pooled over the 7 scoreable folds' out-of-fold predictions
  assert.equal(r.metricsScope.kind, "pooled-folds");
  assert.equal(r.metricsScope.nFolds, 7);
  assert.equal(r.metricsScope.nModels, 7);
  assert.deepEqual(r.metricsScope.excluded, ["shaw"]);
  const shaw = r.folds.find((f) => f.heldOut === "shaw");
  assert.equal(r.metrics.n, r.test.n - shaw.n);
  assert.equal(r.loroSummary.ap.n, 7);
});

test("mergeCertificates: sums counts, fails support only when no fold is scoreable, accepts preview pairs", () => {
  const c = generateCorpus(stats, { scale: 0.02, seed: 0 });
  const folds = loroFolds({ mode: "loro", replicas: [...REPLICA_ORDER] }, c.replicaIds).map((fc) => {
    const certificate = leakageCertificate(c, buildSplit(c, fc));
    return { heldOut: fc.test[0], certificate };
  });
  const merged = mergeCertificates(folds);
  const viaPreview = mergeCertificates(folds.map((f) => ({ heldOut: f.heldOut, cert: f.certificate })));
  assert.deepEqual(viaPreview, merged);
  for (const k of ["testRowsDuplicatingTrain", "quietDuplicates", "activeDuplicates", "stretchOverlap", "testPositives", "nTest"]) {
    assert.equal(merged[k], folds.reduce((a, f) => a + f.certificate[k], 0), k);
  }
  assert.equal(merged.overall, "ok");
  // every fold unscoreable: support fails, still no leak
  const none = mergeCertificates(folds.map((f) => ({ heldOut: f.heldOut, certificate: { ...f.certificate, lowSupport: true, testPositives: 0 } })));
  assert.equal(none.support, "fail");
  assert.equal(none.overall, "ok");
  assert.equal(none.lowSupport, true);
  assert.match(none.checks.find((x) => x.id === "test-support").detail, /no summary to report/);
  // a leaking fold is named, with its own detail
  const leak = folds.map((f, k) =>
    k !== 2 ? f : { ...f, certificate: { ...f.certificate, overall: "fail", checks: f.certificate.checks.map((x) => (x.id === "stretch-disjoint" ? { ...x, severity: "fail", pass: false, detail: "cut" } : x)) } },
  );
  const m = mergeCertificates(leak);
  assert.equal(m.overall, "fail");
  assert.equal(m.checks.find((x) => x.id === "stretch-disjoint").detail, `1 of 8 folds flagged (${folds[2].heldOut}). Worst, fold ${folds[2].heldOut}: cut`);
});

test("without Web Crypto the ledger stage is skipped, not failed, and the results are kept", async () => {
  const desc = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  Object.defineProperty(globalThis, "crypto", { value: undefined, configurable: true, writable: true });
  try {
    assert.equal(globalThis.crypto, undefined, "the premise: no Web Crypto");
    const { result: r, events } = await run({ dataset: { scale: 0.02, seed: 0 }, model: { kind: "knn" } });
    assert.ok(!events.some((e) => e.type === "error"));
    assert.equal(r.cancelled, false);
    assert.equal(r.ledger, null);
    assert.equal(r.ledgerStatus, "skipped-no-crypto");
    assert.ok(r.metrics && Number.isFinite(r.metrics.ap));
    const led = events.filter((e) => e.type === "stage" && e.stage === "ledger");
    assert.equal(led.at(-1).status, "skipped");
    assert.equal(led.at(-1).detail, LEDGER_SKIPPED_NO_CRYPTO);
    assert.match(LEDGER_SKIPPED_NO_CRYPTO, /Web Crypto needs https:\/\/ or localhost/);
    assert.equal(assertMonotoneProgress(events).at(-1), 1);
    // an explicit sha256 still builds the ledger
    const withHash = await run(
      { dataset: { scale: 0.02, seed: 0 }, model: { kind: "knn" } },
      { sha256: async (b) => new Uint8Array(await webcrypto.subtle.digest("SHA-256", b)) },
    );
    assert.ok(isHex64(withHash.result.ledger.rootHex));
  } finally {
    if (desc) Object.defineProperty(globalThis, "crypto", desc);
    else delete globalThis.crypto;
  }
  assert.ok(globalThis.crypto?.subtle, "restored");
});

test("without the pretrained weights every trainable model still runs; the pretrained model fails clearly", async () => {
  for (const kind of ["tessera", "logreg", "knn"]) {
    const { result: r, events } = await run(
      { dataset: { scale: 0.02, seed: 0 }, model: { kind, epochs: 1, logreg: { epochs: 10 } }, ledger: false },
      { pretrainedWeights: null },
    );
    assert.equal(r.cancelled, false, kind);
    assert.ok(!events.some((e) => e.type === "error"), kind);
    assert.equal(r.fidelity, null, `${kind}: no fidelity without the real model`);
    assert.ok(Number.isFinite(r.metrics.ap), kind);
  }
  const events = [];
  await assert.rejects(
    runPipeline({ dataset: { scale: 0.02, seed: 0 }, model: { kind: "pretrained" } }, { stats, pretrainedWeights: null, onEvent: (e) => events.push(e) }),
    /weights\.json\) could not be loaded, so the pretrained model cannot run/,
  );
  assert.equal(events.filter((e) => e.type === "error").length, 1);
});
