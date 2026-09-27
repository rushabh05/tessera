// Seeded PRNG shared by the data generator, split builder and trainer, so a
// given seed produces the identical corpus, split and training run in the main
// thread, in the Web Worker, and in the Node tests.
//
// sfc32 (Small Fast Counting, 32-bit), seeded through splitmix32. Deterministic
// across JS engines because it uses only 32-bit integer ops.

function splitmix32(a) {
  return () => {
    a |= 0;
    a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t ^= t >>> 15;
    t = Math.imul(t, 0x735a2d97);
    t ^= t >>> 15;
    return t >>> 0;
  };
}

/** Hash an arbitrary string/number seed (and optional stream label) to 32 bits. */
export function hashSeed(seed, stream = "") {
  const str = `${seed}|${stream}`;
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

/**
 * makeRng(seed, stream?) -> {
 *   next()          uniform float in [0, 1)
 *   uniform(a, b)   uniform float in [a, b)
 *   int(n)          uniform integer in [0, n)
 *   normal()        standard normal (Box-Muller, cached pair)
 *   bernoulli(p)    true with probability p
 *   pick(probs)     index sampled from a probability vector (need not sum to 1)
 *   shuffle(arr)    in-place Fisher-Yates, returns arr
 *   fork(label)     independent child stream, deterministic from this seed + label
 * }
 * Use distinct `stream` labels ("datagen", "split", "train") so changing how many
 * numbers one stage draws never shifts another stage's sequence.
 */
export function makeRng(seed = 0, stream = "") {
  const base = hashSeed(seed, stream);
  const sm = splitmix32(base);
  let a = sm(),
    b = sm(),
    c = sm(),
    d = sm();
  const nextU32 = () => {
    a >>>= 0;
    b >>>= 0;
    c >>>= 0;
    d >>>= 0;
    let t = (a + b) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    d = (d + 1) | 0;
    t = (t + d) | 0;
    c = (c + t) | 0;
    return t >>> 0;
  };
  for (let i = 0; i < 12; i++) nextU32();

  let spare = null;
  const rng = {
    next: () => nextU32() / 4294967296,
    uniform: (lo, hi) => lo + (hi - lo) * (nextU32() / 4294967296),
    int: (n) => Math.floor((nextU32() / 4294967296) * n),
    normal() {
      if (spare !== null) {
        const v = spare;
        spare = null;
        return v;
      }
      let u = 0;
      while (u === 0) u = nextU32() / 4294967296;
      const v = nextU32() / 4294967296;
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    },
    bernoulli: (p) => nextU32() / 4294967296 < p,
    pick(probs) {
      let total = 0;
      for (const p of probs) total += p;
      let r = (nextU32() / 4294967296) * total;
      for (let i = 0; i < probs.length; i++) {
        r -= probs[i];
        if (r < 0) return i;
      }
      return probs.length - 1;
    },
    shuffle(arr) {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor((nextU32() / 4294967296) * (i + 1));
        const tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
      }
      return arr;
    },
    fork: (label) => makeRng(base, `${stream}/${label}`),
  };
  return rng;
}
