// Module Web Worker that runs the Training Lab pipeline off the main thread.
//
// Protocol (main thread -> worker):
//   {type: 'run', runId, config}   start a run (runs are queued and executed one
//                                  at a time, in arrival order)
//   {type: 'cancel', runId}        cancel that run (queued or running); the run
//                                  still ends with a 'result' event whose
//                                  result.cancelled is true
// Worker -> main thread:
//   {type: 'ready'}                once, on load
//   every pipeline event (plan / stage / progress / corpus / split / epoch /
//   fold / result / error), each carrying runId and tMs. 'plan' comes first and
//   carries the model's per-stage weights (stage events carry them too), so the
//   page's "x% of the work" labels match the run. The result's typed-array
//   buffers are transferred, not copied (the memoriser and logistic regression
//   have no attribution array).
//
// Everything is fetched with relative URLs next to this file, so the page works
// from a plain static server or GitHub Pages sub-path.

import { runPipeline } from "./pipeline.js";
import { setSha256 } from "../merkle.js";

// Web Crypto exists only in a secure context (https:// or localhost). Without it
// the pipeline skips the ledger stage instead of failing the run.
const sha256 =
  globalThis.crypto && globalThis.crypto.subtle
    ? async (bytes) => new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", bytes))
    : null;
if (sha256) setSha256(sha256);

const cache = new Map(); // relative path -> Promise<json>

function loadJson(rel) {
  let p = cache.get(rel);
  if (!p) {
    p = fetch(new URL(rel, import.meta.url)).then(async (res) => {
      if (!res.ok) throw new Error(`Could not load ${rel.replace(/^.*\//, "")} (HTTP ${res.status}).`);
      return res.json();
    });
    // a failed load is not cached, so the next run retries it
    p.catch(() => cache.delete(rel));
    cache.set(rel, p);
  }
  return p;
}

const known = new Set(); // runIds queued or running
const cancelled = new Set();
let chain = Promise.resolve();

function transferablesOf(result) {
  const out = [];
  const t = result && result.test;
  if (!t) return out;
  for (const k of ["scores", "y", "replica", "host", "fold", "included", "attribution"]) {
    const a = t[k];
    if (a && ArrayBuffer.isView(a) && !out.includes(a.buffer)) out.push(a.buffer);
  }
  return out;
}

function post(ev) {
  if (ev.type === "result") {
    try {
      self.postMessage(ev, transferablesOf(ev.result));
      return;
    } catch {
      // fall through to a plain structured copy
    }
  }
  self.postMessage(ev);
}

async function execute(runId, config) {
  let reportedError = false;
  const t0 = performance.now();
  try {
    if (cancelled.has(runId)) {
      // cancelled while queued: end it without doing any work
      post({ type: "result", runId, tMs: 0, result: { runId, config, cancelled: true, cancelledAt: null, test: null, metrics: null } });
      return;
    }
    // The pretrained weights are needed only by the 'pretrained' model and for
    // the optional fidelity number: a failed load must not fail the other
    // models' runs (the pipeline reports a clear error for 'pretrained' only).
    const [stats, pretrainedWeights] = await Promise.all([
      loadJson("../../data/replica_stats.json"),
      loadJson("../../data/weights.json").catch(() => null),
    ]);
    await runPipeline(config, {
      stats,
      pretrainedWeights,
      runId,
      ...(sha256 ? { sha256 } : {}),
      shouldCancel: () => cancelled.has(runId),
      onEvent: (ev) => {
        if (ev.type === "error") reportedError = true;
        post(ev);
      },
    });
  } catch (err) {
    if (!reportedError && !err?.pipelineReported) {
      post({ type: "error", runId, tMs: Math.round(performance.now() - t0), message: err && err.message ? err.message : String(err) });
    }
  } finally {
    known.delete(runId);
    cancelled.delete(runId);
  }
}

self.addEventListener("message", (e) => {
  const msg = e.data || {};
  if (msg.type === "run") {
    const runId = msg.runId ?? `run-${Date.now().toString(36)}`;
    known.add(runId);
    chain = chain.then(() => execute(runId, msg.config || {}));
  } else if (msg.type === "cancel") {
    if (msg.runId == null) {
      for (const id of known) cancelled.add(id); // cancel everything
    } else if (known.has(msg.runId)) {
      cancelled.add(msg.runId);
    }
  }
});

self.postMessage({ type: "ready" });
