// Small DOM + formatting + motion helpers shared by every tab. No framework.

const SVG_NS = "http://www.w3.org/2000/svg";

function applyProps(el, props) {
  for (const [k, v] of Object.entries(props || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class" || k === "className") el.setAttribute("class", v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k === "html") el.innerHTML = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
}

function appendChildren(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

/** h("div", {class: "card", onClick: fn}, "text", child, [more]) */
export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  applyProps(el, props);
  appendChildren(el, children);
  return el;
}

/** SVG-namespace variant of h(). */
export function s(tag, props = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else el.setAttribute(k, String(v));
  }
  appendChildren(el, children);
  return el;
}

export function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

export function reducedMotion() {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export const fmt = {
  int: (n) => (Number.isFinite(n) ? Math.round(n).toLocaleString("en-US") : "—"),
  num: (n, d = 3) => (Number.isFinite(n) ? n.toFixed(d) : "—"),
  pct: (n, d = 1) => (Number.isFinite(n) ? `${(n * 100).toFixed(d)}%` : "—"),
  compact: (n) =>
    Number.isFinite(n)
      ? new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(n)
      : "—",
  ms: (ms) =>
    !Number.isFinite(ms) ? "—" : ms < 1000 ? `${Math.round(ms)} ms` : ms < 60000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`,
  signed: (n, d = 3) => (Number.isFinite(n) ? `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(d)}` : "—"),
};

const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

/**
 * Animate a number into el.textContent. `format` maps the running value to text.
 * Returns a cancel function. Honors prefers-reduced-motion (jumps to the end).
 */
export function countUp(el, to, { from = 0, duration = 900, format = (v) => v.toFixed(3) } = {}) {
  if (!el) return () => {};
  if (!Number.isFinite(to)) {
    el.textContent = "—";
    return () => {};
  }
  if (reducedMotion() || duration <= 0) {
    el.textContent = format(to);
    return () => {};
  }
  let raf = 0;
  const t0 = performance.now();
  const tick = (now) => {
    const t = Math.min(1, (now - t0) / duration);
    el.textContent = format(from + (to - from) * easeOutCubic(t));
    if (t < 1) raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(raf);
}

/** Add staggered .reveal animation to the element's direct children. */
export function stagger(container, { start = 0, selector = ":scope > *" } = {}) {
  if (!container) return;
  container.querySelectorAll(selector).forEach((el, i) => {
    el.classList.add("reveal");
    el.style.setProperty("--i", String(start + i));
  });
}

/** Reveal elements with .reveal-on-scroll when they enter the viewport. */
export function revealOnScroll(root = document) {
  const els = root.querySelectorAll(".reveal-on-scroll:not(.revealed)");
  if (!("IntersectionObserver" in window) || reducedMotion()) {
    els.forEach((el) => el.classList.add("revealed"));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add("revealed");
          io.unobserve(e.target);
        }
      }
    },
    { rootMargin: "0px 0px -8% 0px" }
  );
  els.forEach((el) => io.observe(el));
}

let toastHost = null;
/** toast("Saved", {kind: "good"|"warn"|"bad"|"info", timeout}) */
export function toast(message, { kind = "info", timeout = 3400 } = {}) {
  if (!toastHost) {
    toastHost = h("div", { class: "toast-host", role: "status", "aria-live": "polite" });
    document.body.appendChild(toastHost);
  }
  const el = h("div", { class: `toast ${kind}` }, message);
  toastHost.appendChild(el);
  setTimeout(() => {
    el.classList.add("is-leaving");
    setTimeout(() => el.remove(), 260);
  }, timeout);
  return el;
}

export function downloadFile(filename, content, mime = "application/octet-stream") {
  const blob = content instanceof Blob ? content : new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = h("a", { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

export function debounce(fn, ms = 120) {
  let t = 0;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

/** Keep a range input's filled-track CSS variable in sync with its value. */
export function syncRangeFill(input) {
  const update = () => {
    const min = Number(input.min || 0);
    const max = Number(input.max || 100);
    const pct = ((Number(input.value) - min) / (max - min)) * 100;
    input.style.setProperty("--fill", `${pct}%`);
  };
  input.addEventListener("input", update);
  update();
  return update;
}

/** Safe localStorage wrappers - storage can be absent or throw (private mode, previews). */
export const prefs = {
  get(key, fallback = null) {
    try {
      const v = localStorage.getItem(`tessera:${key}`);
      return v === null ? fallback : JSON.parse(v);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(`tessera:${key}`, JSON.stringify(value));
    } catch {
      /* per-viewer convenience only */
    }
  },
};
