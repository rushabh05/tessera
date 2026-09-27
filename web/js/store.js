// Shared, in-memory app state. Nothing here leaves the browser.
//
// - cached JSON loads (every tab fetches through loadJSON so a file is fetched once)
// - the model the Live Detector and "score your own data" panel should use:
//   the pretrained real model by default, or a model the viewer just trained in
//   the Training Lab (set via setLabModel).

const jsonCache = new Map();
const listeners = new Map();

function emit(event, payload) {
  for (const fn of listeners.get(event) || []) {
    try {
      fn(payload);
    } catch (err) {
      console.error(`store listener for "${event}" failed`, err);
    }
  }
}

let labModel = null;
let activeModelKind = "pretrained"; // "pretrained" | "lab"

export const store = {
  /** Fetch + parse a JSON file relative to index.html, cached. */
  loadJSON(path) {
    if (!jsonCache.has(path)) {
      const p = fetch(path).then((res) => {
        if (!res.ok) throw new Error(`failed to fetch ${path}: HTTP ${res.status}`);
        return res.json();
      });
      p.catch(() => jsonCache.delete(path));
      jsonCache.set(path, p);
    }
    return jsonCache.get(path);
  },

  loadPretrainedWeights() {
    return this.loadJSON("data/weights.json");
  },

  /**
   * labModel = { weights, label, createdAt, summary: {ap, mcc, split, ...} }
   * weights use exactly the web/data/weights.json format, so js/forward.js runs them.
   */
  setLabModel(model) {
    labModel = model;
    emit("labModel", model);
    if (model) this.setActiveModelKind("lab");
    else if (activeModelKind === "lab") this.setActiveModelKind("pretrained");
  },
  getLabModel() {
    return labModel;
  },

  setActiveModelKind(kind) {
    if (kind === "lab" && !labModel) kind = "pretrained";
    if (kind === activeModelKind) return;
    activeModelKind = kind;
    emit("activeModel", kind);
  },
  getActiveModelKind() {
    return activeModelKind;
  },
  /** Resolves to {kind, label, weights} for whichever model is active. */
  async getActiveModel() {
    if (activeModelKind === "lab" && labModel) {
      return { kind: "lab", label: labModel.label, weights: labModel.weights };
    }
    return {
      kind: "pretrained",
      label: "Pretrained on real AIT data",
      weights: await this.loadPretrainedWeights(),
    };
  },

  /** Subscribe; returns an unsubscribe function. */
  on(event, fn) {
    if (!listeners.has(event)) listeners.set(event, new Set());
    listeners.get(event).add(fn);
    return () => listeners.get(event).delete(fn);
  },
  emit,
};
