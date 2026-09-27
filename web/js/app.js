// App shell: hash router, animated tab indicator, theme toggle, lazy tab mounts.
//
// Tab module contract: each module exports `mount(el, ctx)` (sync or async). It is
// called once, the first time its tab is shown. It may return an object with
// optional `onShow()` / `onHide()` hooks, called on every later visit/leave.
//   ctx = { store, navigate(tabId), toast, isActive(tabId) }

import { store } from "./store.js";
import { icon } from "./ui/icons.js";
import { h, toast, prefs, reducedMotion } from "./ui/dom.js";

const TABS = [
  { id: "overview", label: "Overview", icon: "overview", module: "./tabs/overview.js", mount: "overview-mount" },
  { id: "detector", label: "Live Detector", short: "Detector", icon: "detector", module: "./main.js", mount: null },
  { id: "lab", label: "Training Lab", short: "Lab", icon: "lab", module: "./lab/lab-ui.js", mount: "lab-mount" },
  { id: "results", label: "Real Results", short: "Results", icon: "results", module: "./tabs/results.js", mount: "results-mount" },
  { id: "design", label: "Design Decisions", short: "Design", icon: "design", module: "./tabs/design.js", mount: "design-mount" },
  { id: "about", label: "About", icon: "about", module: "./tabs/about.js", mount: "about-mount" },
];
const DEFAULT_TAB = "overview";
const isTab = (id) => TABS.some((t) => t.id === id);

const state = { active: null, mounted: new Map() };
const ctx = { store, navigate, toast, isActive: (id) => state.active === id };

function hashId() {
  return (location.hash || "").replace(/^#\/?/, "").split(/[/?]/)[0];
}

/** The tab the URL names; an unknown hash keeps the current tab (or the default on first load). */
function tabFromHash() {
  const id = hashId();
  if (isTab(id)) return id;
  return state.active || DEFAULT_TAB;
}

export function navigate(id, { focusPanel = false } = {}) {
  if (!isTab(id)) id = DEFAULT_TAB;
  if (location.hash !== `#${id}`) history.pushState(null, "", `#${id}`);
  const already = state.active === id;
  const done = already ? Promise.resolve() : show(id);
  if (focusPanel) done.then(() => focusPanelHeading(id));
  return done;
}

function focusPanelHeading(id) {
  const panel = document.getElementById(`tab-${id}`);
  const target = panel?.querySelector("h1") || panel;
  if (!target) return;
  if (!target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
  target.focus({ preventScroll: true });
}

function renderTabs() {
  const nav = document.getElementById("tabs");
  nav.innerHTML = "";
  nav.appendChild(h("span", { class: "tab-indicator", id: "tab-indicator", "aria-hidden": "true" }));
  for (const t of TABS) {
    const btn = h("button", {
      class: "tab-btn",
      type: "button",
      role: "tab",
      id: `tabbtn-${t.id}`,
      "aria-controls": `tab-${t.id}`,
      "aria-selected": "false",
      "aria-label": t.label,
      "data-tab-target": t.id,
      html: `${icon(t.icon)}<span class="tab-label-long">${t.label}</span><span class="tab-label-short" aria-hidden="true">${t.short || t.label}</span>`,
      onClick: () => navigate(t.id),
    });
    nav.appendChild(btn);
  }
  nav.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const i = TABS.findIndex((t) => t.id === state.active);
    const next = TABS[(i + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length];
    navigate(next.id);
    document.getElementById(`tabbtn-${next.id}`)?.focus();
  });
}

function moveIndicator() {
  const btn = document.getElementById(`tabbtn-${state.active}`);
  const ind = document.getElementById("tab-indicator");
  if (!btn || !ind) return;
  ind.style.width = `${btn.offsetWidth}px`;
  ind.style.transform = `translateX(${btn.offsetLeft - 4}px)`;
  const nav = document.getElementById("tabs");
  if (nav.scrollWidth > nav.clientWidth) {
    const left = btn.offsetLeft - nav.clientWidth / 2 + btn.offsetWidth / 2;
    nav.scrollTo({ left, behavior: reducedMotion() ? "auto" : "smooth" });
  }
}

function mountErrorCard(err, what) {
  return h(
    "div",
    { class: "mount-error", role: "alert" },
    `${what} failed to load (${err.message}). The rest of the site still works; reloading the page usually fixes this.`
  );
}

async function mountTab(tab) {
  if (state.mounted.has(tab.id)) return state.mounted.get(tab.id);
  const p = (async () => {
    try {
      const mod = await import(tab.module);
      if (tab.id === "detector") {
        await mountByod();
        return null;
      }
      const el = document.getElementById(tab.mount);
      return (await mod.mount(el, ctx)) || null;
    } catch (err) {
      console.error(`failed to mount tab "${tab.id}"`, err);
      const el = tab.mount ? document.getElementById(tab.mount) : document.getElementById("byod-mount");
      if (el) {
        el.innerHTML = "";
        el.appendChild(mountErrorCard(err, "This section"));
      }
      return null;
    }
  })();
  state.mounted.set(tab.id, p);
  return p;
}

async function mountByod() {
  const el = document.getElementById("byod-mount");
  if (!el || el.dataset.mounted) return;
  el.dataset.mounted = "1";
  try {
    const mod = await import("./tabs/byod.js");
    await mod.mount(el, ctx);
  } catch (err) {
    console.error("failed to mount byod panel", err);
    el.appendChild(mountErrorCard(err, 'The "score your own data" panel'));
  }
}

async function show(id) {
  const prev = state.active;
  if (prev === id) return;
  state.active = id;
  for (const t of TABS) {
    const panel = document.getElementById(`tab-${t.id}`);
    const btn = document.getElementById(`tabbtn-${t.id}`);
    const on = t.id === id;
    panel?.classList.toggle("is-active", on);
    btn?.setAttribute("aria-selected", on ? "true" : "false");
    btn?.setAttribute("tabindex", on ? "0" : "-1");
  }
  moveIndicator();
  const tab = TABS.find((t) => t.id === id);
  document.title = `${tab.label} · TESSERA`;
  window.scrollTo({ top: 0, behavior: prev && !reducedMotion() ? "smooth" : "auto" });

  // Never block the new tab on the previous tab's (possibly still loading) mount.
  if (prev && state.mounted.has(prev)) {
    state.mounted.get(prev).then((hooks) => {
      if (state.active !== prev) hooks?.onHide?.();
    });
  }
  const hooks = await mountTab(tab);
  if (state.active === id) hooks?.onShow?.();
}

function initTheme() {
  const btn = document.getElementById("theme-btn");
  const current = () => {
    const set = document.documentElement.dataset.theme;
    if (set === "light" || set === "dark") return set;
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  };
  const paint = () => {
    const dark = current() === "dark";
    btn.innerHTML = icon(dark ? "sun" : "moon");
    btn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    btn.title = btn.getAttribute("aria-label");
  };
  btn.addEventListener("click", () => {
    const next = current() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    prefs.set("theme", next);
    paint();
    store.emit("theme", next);
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => {
    paint();
    store.emit("theme", current());
  });
  paint();
}

function initTour() {
  const btn = document.getElementById("tour-btn");
  btn.innerHTML = `${icon("compass")}<span class="btn-label">Guided tour</span>`;
  btn.setAttribute("aria-label", "Start the guided tour");
  btn.addEventListener("click", async () => {
    try {
      const mod = await import("./tabs/tour.js");
      mod.startTour(ctx);
    } catch (err) {
      console.error(err);
      toast(`The guided tour failed to load: ${err.message}`, { kind: "bad" });
    }
  });
}

function onHashRoute() {
  // Unknown hashes (e.g. the #main skip link) leave the current tab alone.
  if (isTab(hashId())) show(hashId());
}

function boot() {
  renderTabs();
  initTheme();
  initTour();
  document.getElementById("brand").addEventListener("click", () => navigate("overview"));
  window.addEventListener("popstate", onHashRoute);
  window.addEventListener("hashchange", onHashRoute);
  window.addEventListener("resize", () => moveIndicator());
  document.addEventListener("click", (e) => {
    const a = e.target.closest?.('a[href^="#"]');
    if (!a) return;
    const id = a.getAttribute("href").slice(1);
    if (id === "main") {
      // Skip link: move focus into the visible tab without changing the route.
      e.preventDefault();
      focusPanelHeading(state.active);
      return;
    }
    if (isTab(id)) {
      e.preventDefault();
      // A link that switches tabs moves focus to the new tab's heading, so the
      // next Tab press continues in the content the viewer just opened.
      navigate(id, { focusPanel: !a.closest(".tabs") });
    }
  });
  show(tabFromHash());
  // Mount the detector early (it is the main live demo and cheap to boot) so
  // switching to it is instant. requestIdleCallback is missing in Safari.
  const idle = typeof requestIdleCallback === "function" ? requestIdleCallback : (cb) => setTimeout(cb, 400);
  idle(() => mountTab(TABS.find((t) => t.id === "detector")));
}

boot();
