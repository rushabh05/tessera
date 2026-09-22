// TESSERA-base forward pass, hand-written in vanilla JS - no framework, no CDN.
//
// This is a from-scratch reimplementation of tessera/models/tessera_base.py's
// _TesseraBaseNet, matching it operation-for-operation so the numbers this page
// shows are the real model's real output, not a simulation. The parity test in
// parity.test.mjs proves this against 40 golden (input, output) pairs computed
// by the real trained PyTorch model.
//
// Two decisions made specifically so this file can match PyTorch exactly rather
// than approximately:
//  - GELU uses the TANH approximation. PyTorch's default GELU uses exact erf,
//    which has no clean dependency-free JS implementation; the Python model was
//    trained with `nn.GELU(approximate='tanh')` instead, so both sides compute
//    the identical closed-form formula - zero approximation error, not a small one.
//  - The softmax masking sentinel is float32's actual minimum representable
//    value (-3.4028234663852886e38), matching `torch.finfo(dtype).min` exactly,
//    not JavaScript's more obvious -Infinity (which PyTorch's own gate does NOT
//    use - matching this detail is what makes a masked logit's post-softmax
//    weight come out bit-for-bit zero rather than merely very small).

const FLOAT32_MIN = -3.4028234663852886e38;

// y = x @ W^T + b. W is (out, in) - the same orientation torch stores it in and
// export.py exports it in, so no transpose is needed on either side.
function linear(x, W, b) {
  const out = new Array(W.length);
  for (let o = 0; o < W.length; o++) {
    let acc = b[o];
    const row = W[o];
    for (let i = 0; i < row.length; i++) acc += x[i] * row[i];
    out[o] = acc;
  }
  return out;
}

// GroupNorm over a flat (C,) vector - PyTorch's GroupNorm applied to a (N, C)
// input (no spatial dims) reduces to exactly this per sample. Uses the BIASED
// variance estimator (divide by group size, no Bessel's correction), which is
// what torch.nn.GroupNorm uses internally - a std-based or corrected-variance
// port would diverge from the real model by a small but real amount.
function groupNorm(x, { num_groups, eps, weight, bias }) {
  const C = x.length;
  const groupSize = C / num_groups;
  const out = new Array(C);
  for (let g = 0; g < num_groups; g++) {
    const start = g * groupSize;
    let mean = 0;
    for (let i = 0; i < groupSize; i++) mean += x[start + i];
    mean /= groupSize;
    let variance = 0;
    for (let i = 0; i < groupSize; i++) {
      const d = x[start + i] - mean;
      variance += d * d;
    }
    variance /= groupSize; // biased, matches torch
    const invStd = 1 / Math.sqrt(variance + eps);
    for (let i = 0; i < groupSize; i++) {
      const c = start + i;
      out[c] = (x[c] - mean) * invStd * weight[c] + bias[c];
    }
  }
  return out;
}

// GELU, tanh approximation - the exact formula `nn.GELU(approximate='tanh')`
// computes: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
function geluTanh(x) {
  return x.map((v) => 0.5 * v * (1 + Math.tanh(SQRT_2_OVER_PI * (v + 0.044715 * v * v * v))));
}

function encode(xSlice, encoderWeights) {
  let h = linear(xSlice, encoderWeights.linear0.weight, encoderWeights.linear0.bias);
  h = groupNorm(h, encoderWeights.groupnorm);
  h = geluTanh(h);
  h = linear(h, encoderWeights.linear1.weight, encoderWeights.linear1.bias);
  return h;
}

function softmaxWithMask(logits, availability) {
  const allAbsent = availability.every((a) => a < 0.5);
  const masked = logits.map((v, i) => (allAbsent ? 0 : availability[i] < 0.5 ? FLOAT32_MIN : v));
  const max = Math.max(...masked);
  const exps = masked.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function tanhVec(x) {
  return x.map(Math.tanh);
}

/**
 * The full forward pass. `input` is a 42-length flat feature vector (M1: 8,
 * M2: 24, M3: 2, M4: 8, in that order - the exact contract
 * tessera/features/pipeline.py produces). `availability` is a 4-length
 * 0/1 array, one entry per modality in the same M1/M2/M3/M4 order.
 *
 * Returns { score, attribution } - `attribution` sums to 1 over the available
 * modalities and IS the model's own gate weights, with the important caveat
 * documented in RESULTS.md: measured to UNDERSTATE a modality's true
 * importance in at least one case, so it is shown as a signal, never as proof.
 */
export function forward(input, availability, weights) {
  const embeds = weights.modality_slices.map(([start, stop], m) =>
    encode(input.slice(start, stop), weights.encoders[m])
  );

  const flat = embeds.flat();
  const gateInput = flat.concat(availability);
  let g = linear(gateInput, weights.fusion.gate_linear0.weight, weights.fusion.gate_linear0.bias);
  g = geluTanh(g);
  g = linear(g, weights.fusion.gate_linear1.weight, weights.fusion.gate_linear1.bias);
  const gates = softmaxWithMask(g, availability);

  const embedDim = weights.fusion.embed_dim;
  const fused = new Array(embedDim).fill(0);
  for (let m = 0; m < embeds.length; m++) {
    const t = tanhVec(embeds[m]);
    for (let d = 0; d < embedDim; d++) fused[d] += gates[m] * t[d];
  }

  let h = linear(fused, weights.head.linear0.weight, weights.head.linear0.bias);
  h = geluTanh(h);
  // Dropout is inference-inert (a no-op at eval time in the Python model too).
  const logit = linear(h, weights.head.linear1.weight, weights.head.linear1.bias)[0];
  const score = 1 / (1 + Math.exp(-logit));

  return { score, attribution: gates };
}
