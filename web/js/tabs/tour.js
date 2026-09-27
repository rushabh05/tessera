// Guided tour: a spotlight overlay (dimmed backdrop with a rounded cut-out that
// glides between targets) plus a popover card, walking across every tab via the
// data-tour anchors each module places.
//
//   startTour(ctx, {startAt?}) -> {next(), back(), end()}
//
// Robust to the page it runs on: a step whose anchor does not appear within
// ~2.5 s of navigating (a tab that failed to load, a panel not rendered yet) is
// skipped in the direction of travel; a step may name fallback anchors. Keyboard:
// Right / Left move, Esc ends; focus is trapped in the popover and restored after.
// Motion is calm and disappears entirely under prefers-reduced-motion.

import { h, reducedMotion, prefs } from "../ui/dom.js";
import { icon } from "../ui/icons.js";

const STEPS = [
  {
    anchor: "nav",
    title: "Six tabs, one story",
    body: "Overview, the Live detector, a Training Lab, Real results, Design decisions and About. Move through this tour with the arrow keys; Esc leaves it at any time.",
  },
  {
    tab: "overview",
    anchor: "overview-hero",
    title: "What TESSERA is",
    body: "A multimodal cloud anomaly detector, evaluated honestly: average precision on leakage-checked splits, and a ledger that makes every verdict tamper-evident.",
  },
  {
    tab: "overview",
    anchor: "overview-pipeline",
    title: "How a verdict is made",
    body: "Four feature groups, drawn from two telemetry streams (log files and Suricata network events), are encoded separately, fused by a gate that masks any missing group, scored, then committed to a Merkle ledger. Click a stage for the real numbers behind it.",
  },
  {
    tab: "detector",
    anchor: "detector-list",
    title: "Score a window",
    body: "The real trained model runs in your browser. These example windows are synthetic: by the project's data policy (LICENSE-DATA), no real AIT rows are shipped. Pick one to score it.",
  },
  {
    tab: "detector",
    anchor: "detector-attribution",
    title: "Switch a source off",
    body: "Turn a feature group off and the model re-scores live, using the same mask it applies when a real source goes quiet. The weights are a hint about what it leaned on, not proof.",
  },
  {
    tab: "detector",
    anchor: "ledger",
    title: "Try to rewrite history",
    body: "Every verdict is a leaf in a Merkle log. Edit one and the root changes, so anyone who kept the old root can detect the edit.",
  },
  {
    tab: "lab",
    anchor: "lab-experiment",
    title: "The split experiment",
    body: "One click runs the same model twice on synthetic network-metric windows, once on a random split and once on a chronological one, and compares the two scores against each split's no-skill line. Start with a memoriser (nearest neighbours), then try it with TESSERA-base. It is synthetic, so it shows the mechanism, not a real result.",
  },
  {
    tab: "lab",
    anchor: "lab-split",
    title: "Choose how to split the data",
    body: "Hold out whole replicas, cut by time, or pick the leaky random split on purpose and watch what it does to the score.",
  },
  {
    tab: "lab",
    anchor: "lab-leakage",
    title: "The leakage certificate",
    body: "Before anything trains, the split is checked: replicas and time stretches must not straddle train and test, and duplicated rows are counted.",
  },
  {
    tab: "lab",
    anchor: "lab-run",
    title: "Train it yourself",
    body: "Train the same small model on synthetic data calibrated from real aggregate statistics, entirely in your browser. Nothing is uploaded.",
  },
  {
    tab: "lab",
    anchor: "lab-progress",
    fallback: ["lab-model"],
    title: "Watch every stage",
    body: "Data generation, splitting, the leakage checks, each training epoch, testing and the ledger all report their progress live.",
    fallbackTitle: "Pick a model, then run",
    fallbackBody: "Choose which model to train here. Once you press run, every stage (data, split, leakage checks, training, testing, ledger) reports its progress live.",
  },
  {
    tab: "results",
    anchor: "results-loro",
    title: "Real results, real data",
    body: "Leave-one-replica-out on real AIT data: train on seven replicas, test on the eighth. A fold with too few attacks is shown, but kept out of the average.",
  },
  {
    tab: "design",
    anchor: "design-decisions",
    title: "Design decisions",
    body: "TESSERA's own design-decision record: each choice, next to the evidence behind it and the file that holds it.",
  },
  {
    anchor: "theme",
    title: "Light or dark",
    body: "Switch themes whenever you like. That's the tour: the About tab has the model card, the commands behind the real numbers, and a viva FAQ.",
  },
];

const TAB_LABELS = {
  overview: "Overview",
  detector: "Live detector",
  lab: "Training Lab",
  results: "Real results",
  design: "Design decisions",
  about: "About",
};

const WAIT_MS = 2500;
const FALLBACK_GRACE_MS = 300;
const MOVE_MS = 480;
const DOCK_BELOW = 640;

let current = null;
let tourCount = 0;

/** Start (or restart) the guided tour. */
export function startTour(ctx, opts = {}) {
  if (current) current.end({ silent: true });
  current = createTour(ctx || {}, opts);
  return current;
}

/** Number of stops, for callers that want to mention it. */
export const TOUR_LENGTH = STEPS.length;

/* ------------------------------------------------------------------ helpers */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function isVisible(el) {
  if (!el || !el.isConnected) return false;
  const r = el.getBoundingClientRect();
  if (r.width < 2 || r.height < 2) return false;
  const cs = getComputedStyle(el);
  return cs.visibility !== "hidden" && cs.display !== "none";
}

function findAnchor(name) {
  for (const el of document.querySelectorAll(`[data-tour="${name}"]`)) if (isVisible(el)) return el;
  return null;
}

function inFixedContext(el) {
  for (let n = el; n && n !== document.body && n !== document.documentElement; n = n.parentElement) {
    const p = getComputedStyle(n).position;
    if (p === "fixed" || p === "sticky") return true;
  }
  return false;
}

/* ------------------------------------------------------------------ the tour */

function createTour(ctx, { startAt = 0 } = {}) {
  const steps = STEPS;
  const n = steps.length;
  const RM = reducedMotion();
  const prevFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  const uid = String(++tourCount);

  let index = -1;
  let target = null;
  let targetName = null;
  let token = 0;
  let ended = false;
  let raf = 0;
  let lastKey = "";
  let moveTimer = 0;
  let skippedNote = "";
  let userScrolled = false;
  const markUserScroll = () => (userScrolled = true);

  /* ---------- DOM */
  const root = h("div", { class: "tour-root" });
  const blocker = h("div", { class: "tour-blocker", "aria-hidden": "true" });
  const spot = h("div", { class: "tour-spot is-idle", "aria-hidden": "true" });

  const count = h("span", { class: "tour-count" });
  const closeBtn = h("button", { class: "icon-btn tour-close", type: "button", "aria-label": "End the tour", title: "End the tour (Esc)", html: icon("x") });
  const title = h("h2", { class: "tour-title tour-anim", id: `tour-title-${uid}` });
  const body = h("p", { class: "tour-body tour-anim", id: `tour-body-${uid}` });
  const loadingText = h("span", {});
  const loading = h("div", { class: "tour-loading small", hidden: true }, h("span", { class: "tour-spinner", "aria-hidden": "true" }), loadingText);
  const note = h("div", { class: "tour-note tiny", hidden: true });
  const dots = h("div", { class: "tour-dots", "aria-hidden": "true" });
  const dotEls = steps.map(() => dots.appendChild(h("span", { class: "tour-dot" })));
  const skipBtn = h("button", { class: "btn sm ghost tour-skip", type: "button" }, "Skip tour");
  const backBtn = h("button", { class: "btn sm tour-back", type: "button", html: `${icon("arrowLeft")}<span>Back</span>` });
  const nextBtn = h("button", { class: "btn sm primary tour-next", type: "button" });
  const keys = h("div", { class: "tour-keys tiny", "aria-hidden": "true" }, h("kbd", {}, "←"), h("kbd", {}, "→"), " to move · ", h("kbd", {}, "Esc"), " to close");
  const live = h("div", { class: "sr-only", "aria-live": "polite" });

  const card = h(
    "div",
    { class: "tour-card" },
    h("div", { class: "tour-head" }, count, closeBtn),
    title,
    body,
    loading,
    note,
    h("div", { class: "tour-foot" }, dots, h("div", { class: "tour-actions" }, skipBtn, backBtn, nextBtn)),
    keys,
    live
  );
  const pop = h(
    "div",
    { class: "tour-pop", role: "dialog", "aria-modal": "true", "aria-labelledby": title.id, "aria-describedby": body.id, tabindex: "-1" },
    card
  );
  root.append(blocker, spot, pop);
  document.body.appendChild(root);
  document.documentElement.classList.add("tour-open");
  requestAnimationFrame(() => root.classList.add("is-on"));

  /* ---------- rendering */
  function renderText(i, { waiting = false, fallback = false } = {}) {
    const step = steps[i];
    const nextTitle = fallback && step.fallbackTitle ? step.fallbackTitle : step.title;
    const nextBody = fallback && step.fallbackBody ? step.fallbackBody : step.body;
    const changed = title.textContent !== nextTitle || body.textContent !== nextBody;
    count.textContent = `Step ${i + 1} of ${n}`;
    title.textContent = nextTitle;
    body.textContent = nextBody;
    dotEls.forEach((d, j) => {
      d.classList.toggle("is-done", j < i);
      d.classList.toggle("is-current", j === i);
    });
    backBtn.disabled = i === 0;
    const last = i === n - 1;
    nextBtn.innerHTML = last ? `<span>Finish</span>${icon("check")}` : `<span>Next</span>${icon("arrowRight")}`;
    nextBtn.setAttribute("aria-label", last ? "Finish the tour" : `Next: step ${i + 2} of ${n}`);
    loading.hidden = !waiting;
    loadingText.textContent = waiting ? (step.tab && !ctx.isActive?.(step.tab) ? `Opening ${TAB_LABELS[step.tab] || step.tab}…` : "Finding it on the page…") : "";
    pop.classList.toggle("is-waiting", waiting);
    note.hidden = !skippedNote || waiting;
    note.textContent = skippedNote;
    if (!RM && changed) {
      card.classList.remove("is-swapping");
      void card.offsetWidth;
      card.classList.add("is-swapping");
    }
    if (!waiting) live.textContent = `Step ${i + 1} of ${n}. ${title.textContent}. ${body.textContent}`;
  }

  function docked() {
    return window.innerWidth < DOCK_BELOW;
  }

  function startMove() {
    if (RM) return;
    root.classList.add("is-moving");
    clearTimeout(moveTimer);
    moveTimer = setTimeout(() => root.classList.remove("is-moving"), MOVE_MS + 220);
  }

  function place(animate) {
    if (animate) startMove();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const isDocked = docked();
    pop.classList.toggle("is-docked", isDocked);

    if (!target) {
      spot.classList.add("is-idle");
      spot.style.transform = `translate(${Math.round(vw / 2)}px, ${Math.round(vh / 2)}px)`;
      spot.style.width = "0px";
      spot.style.height = "0px";
      if (!isDocked) {
        const pw = pop.offsetWidth;
        const ph = pop.offsetHeight;
        pop.style.transform = `translate(${Math.round((vw - pw) / 2)}px, ${Math.round(Math.max(12, (vh - ph) / 2))}px)`;
      } else pop.style.transform = "";
      return;
    }

    const r = target.getBoundingClientRect();
    const pad = r.width < 60 || r.height < 60 ? 6 : 10;
    let x1 = r.left - pad;
    let y1 = r.top - pad;
    let w = r.width + pad * 2;
    let hh = r.height + pad * 2;
    // Clip the frame to the viewport so a target larger than the screen (or the
    // sticky header at its very top) still shows its ring on every side.
    const inset = 3;
    const cx1 = Math.max(inset, x1);
    const cy1 = Math.max(inset, y1);
    const cx2 = Math.min(vw - inset, x1 + w);
    const cy2 = Math.min(vh - inset, y1 + hh);
    if (cx2 - cx1 > 16 && cy2 - cy1 > 16) {
      x1 = cx1;
      y1 = cy1;
      w = cx2 - cx1;
      hh = cy2 - cy1;
    }
    spot.classList.remove("is-idle");
    spot.style.transform = `translate(${Math.round(x1)}px, ${Math.round(y1)}px)`;
    spot.style.width = `${Math.round(w)}px`;
    spot.style.height = `${Math.round(hh)}px`;
    const radius = Math.min(18, Math.max(8, Math.min(w, hh) / 5));
    spot.style.borderRadius = `${Math.round(radius)}px`;

    if (isDocked) {
      pop.style.transform = "";
      pop.dataset.side = "dock";
      return;
    }
    const m = 12;
    const gap = 14;
    const pw = pop.offsetWidth;
    const ph = pop.offsetHeight;
    // Decide placement against the part of the target that is on screen.
    const vx1 = Math.max(0, x1);
    const vy1 = Math.max(0, y1);
    const vx2 = Math.min(vw, x1 + w);
    const vy2 = Math.min(vh, y1 + hh);
    const clampX = (x) => Math.min(Math.max(m, x), vw - pw - m);
    const clampY = (y) => Math.min(Math.max(m, y), vh - ph - m);
    const cx = (vx1 + vx2) / 2;
    const cy = (vy1 + vy2) / 2;
    let left;
    let top;
    let side;
    if (vh - vy2 - gap - m >= ph) {
      top = vy2 + gap;
      left = clampX(cx - pw / 2);
      side = "below";
    } else if (vy1 - gap - m >= ph) {
      top = vy1 - gap - ph;
      left = clampX(cx - pw / 2);
      side = "above";
    } else if (vw - vx2 - gap - m >= pw) {
      left = vx2 + gap;
      top = clampY(cy - ph / 2);
      side = "right";
    } else if (vx1 - gap - m >= pw) {
      left = vx1 - gap - pw;
      top = clampY(cy - ph / 2);
      side = "left";
    } else {
      left = clampX(vw - pw - m * 2);
      top = clampY(vh - ph - m * 2);
      side = "over";
    }
    pop.style.transform = `translate(${Math.round(left)}px, ${Math.round(clampY(top))}px)`;
    pop.dataset.side = side;
  }

  function bringIntoView(el) {
    if (inFixedContext(el)) return;
    const r = el.getBoundingClientRect();
    const vh = window.innerHeight;
    const bar = document.querySelector(".topbar");
    const barBottom = bar ? Math.max(0, bar.getBoundingClientRect().bottom) : 0;
    const top = barBottom + 16;
    const reserve = docked() ? pop.offsetHeight + 28 : 0;
    const bottom = vh - 16 - reserve;
    const avail = bottom - top;
    const ph = docked() ? 0 : pop.offsetHeight + 20;
    const fullyVisible = r.top >= top && r.bottom <= bottom;
    const roomForPop = docked() || r.top - top >= ph || bottom - r.bottom >= ph || window.innerWidth - r.right > 400 || r.left > 400;
    if (fullyVisible && roomForPop) return;
    let desired;
    if (!docked() && r.height + ph <= avail) desired = top + (avail - (r.height + ph)) / 2;
    else if (r.height <= avail) desired = top + (avail - r.height) / 2;
    else desired = top;
    const delta = r.top - desired;
    if (Math.abs(delta) < 4) return;
    window.scrollTo({ top: Math.max(0, window.scrollY + delta), behavior: RM ? "auto" : "smooth" });
  }

  async function waitFor(step, cancelled) {
    const t0 = performance.now();
    for (;;) {
      if (cancelled()) return null;
      const el = findAnchor(step.anchor);
      if (el) return { el, name: step.anchor, isFallback: false };
      const elapsed = performance.now() - t0;
      if (elapsed > FALLBACK_GRACE_MS) {
        for (const name of step.fallback || []) {
          const f = findAnchor(name);
          if (f) return { el: f, name, isFallback: true };
        }
      }
      if (step.tab && elapsed > 250) {
        const panel = document.getElementById(`tab-${step.tab}`);
        if (panel?.querySelector(".mount-error")) return null;
      }
      if (elapsed > WAIT_MS) return null;
      await sleep(70);
    }
  }

  async function go(i, direction = 1) {
    if (ended) return;
    if (i >= n) {
      end({ completed: true });
      return;
    }
    if (i < 0) i = 0;
    const my = ++token;
    const step = steps[i];
    index = i;
    target = null;
    targetName = null;
    renderText(i, { waiting: true });
    place(true);
    if (step.tab && !ctx.isActive?.(step.tab)) ctx.navigate?.(step.tab);

    const found = await waitFor(step, () => my !== token || ended);
    if (my !== token || ended) return;
    if (!found) {
      skippedNote = `Skipped “${step.title}”: it isn't on the page right now.`;
      live.textContent = skippedNote;
      const ni = i + direction;
      if (ni < 0) return go(i + 1, 1);
      if (ni >= n) {
        // Nothing further in this direction: finish when moving forward, else stay put.
        if (direction > 0) end({ completed: true });
        return;
      }
      return go(ni, direction);
    }
    target = found.el;
    targetName = found.name;
    renderText(i, { fallback: found.isFallback });
    skippedNote = "";
    // Measure after the text swap, then glide the spotlight and card over.
    await new Promise((r) => requestAnimationFrame(r));
    if (my !== token || ended) return;
    bringIntoView(target);
    place(true);
    nextBtn.focus({ preventScroll: true });
    // Content above the target can still settle (a lazy tab finishing its
    // layout) while the smooth scroll runs; check once more when it has landed.
    setTimeout(() => {
      if (my === token && !ended && target && !userScrolled) bringIntoView(target);
    }, RM ? 80 : 850);
    userScrolled = false;
  }

  function next() {
    if (ended) return;
    if (index >= n - 1) end({ completed: true });
    else go(index + 1, 1);
  }
  function back() {
    if (ended || index <= 0) return;
    go(index - 1, -1);
  }

  /* ---------- follow the target every frame (scroll, resize, layout shifts) */
  function loop() {
    if (ended) return;
    if (target && (!target.isConnected || !isVisible(target)) && targetName) {
      const again = findAnchor(targetName);
      if (again) target = again;
    }
    const r = target ? target.getBoundingClientRect() : null;
    const key = r
      ? `${Math.round(r.left)},${Math.round(r.top)},${Math.round(r.width)},${Math.round(r.height)},${window.innerWidth},${window.innerHeight},${pop.offsetHeight}`
      : `idle,${window.innerWidth},${window.innerHeight},${pop.offsetHeight}`;
    if (key !== lastKey) {
      lastKey = key;
      place(false);
    }
    raf = requestAnimationFrame(loop);
  }

  /* ---------- input */
  function focusables() {
    return [...pop.querySelectorAll("button:not([disabled]), a[href], [tabindex]:not([tabindex='-1'])")].filter((el) => el.getClientRects().length > 0);
  }
  function onKey(e) {
    if (ended) return;
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      end();
    } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      e.stopPropagation();
      if (e.key === "ArrowRight") next();
      else back();
    } else if (e.key === "Tab") {
      const f = focusables();
      if (!f.length) {
        e.preventDefault();
        pop.focus();
        return;
      }
      const first = f[0];
      const lastEl = f[f.length - 1];
      const inside = pop.contains(document.activeElement);
      if (e.shiftKey && (!inside || document.activeElement === first)) {
        e.preventDefault();
        lastEl.focus();
      } else if (!e.shiftKey && (!inside || document.activeElement === lastEl)) {
        e.preventDefault();
        first.focus();
      }
    }
  }
  function onFocusIn(e) {
    if (ended || pop.contains(e.target)) return;
    (nextBtn.disabled ? pop : nextBtn).focus({ preventScroll: true });
  }
  function nudge() {
    if (RM) return nextBtn.focus({ preventScroll: true });
    card.classList.remove("is-nudged");
    void card.offsetWidth;
    card.classList.add("is-nudged");
    nextBtn.focus({ preventScroll: true });
  }

  nextBtn.addEventListener("click", next);
  backBtn.addEventListener("click", back);
  skipBtn.addEventListener("click", () => end());
  closeBtn.addEventListener("click", () => end());
  blocker.addEventListener("click", nudge);
  document.addEventListener("keydown", onKey, true);
  document.addEventListener("focusin", onFocusIn, true);
  window.addEventListener("wheel", markUserScroll, { passive: true });
  window.addEventListener("touchmove", markUserScroll, { passive: true });

  /* ---------- teardown */
  function end({ completed = false, silent = false } = {}) {
    if (ended) return;
    ended = true;
    token++;
    cancelAnimationFrame(raf);
    clearTimeout(moveTimer);
    document.removeEventListener("keydown", onKey, true);
    document.removeEventListener("focusin", onFocusIn, true);
    window.removeEventListener("wheel", markUserScroll);
    window.removeEventListener("touchmove", markUserScroll);
    document.documentElement.classList.remove("tour-open");
    root.classList.remove("is-on");
    root.classList.add("is-leaving");
    setTimeout(() => root.remove(), RM || silent ? 0 : 280);
    if (completed) prefs.set("tourDone", true);
    if (current === api) current = null;
    if (silent) return;
    const restoreTo = prevFocus && prevFocus.isConnected && prevFocus.getClientRects().length ? prevFocus : document.getElementById("tour-btn");
    restoreTo?.focus?.({ preventScroll: true });
    if (completed) ctx.toast?.("Tour complete. The About tab has the viva FAQ and the commands to reproduce every number.", { kind: "good" });
  }

  const api = { next, back, end, get index() { return index; } };
  raf = requestAnimationFrame(loop);
  go(Math.min(Math.max(0, startAt | 0), n - 1), 1);
  return api;
}
