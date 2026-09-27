#!/usr/bin/env node
// Static integration check for the TESSERA static site (dev-only, not loaded by the page).
//
//   node web/dev/check-imports.mjs        (from anywhere; paths resolve from this file)
//
// For every .js/.mjs file under web/js, web/tests and the web root it:
//   1. resolves every relative import / export-from / dynamic import() specifier and checks
//      that the target file exists;
//   2. checks that every named import (static `import {a as b}`, `export {a} from`,
//      `const {a} = await import()`, `(await import()).a`, `mod.a` after `const mod = await
//      import()`) is really exported by the target module (exports parsed with regexes);
//   3. checks relative asset URLs: `new URL("./x", import.meta.url)` and string literals like
//      "data/*.json" (relative to index.html) or "../data/*.json" (relative to the file);
//   4. runs `node --check` on every file under web/js;
//   5. cross-checks the regex export list against the real module namespace for modules
//      that import cleanly under Node (pure modules), so the regex parser cannot drift.
// Exits 1 on any problem.

import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join, dirname, relative, resolve, extname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const rel = (p) => relative(WEB, p) || ".";
const problems = [];
const notes = [];
const fail = (msg) => problems.push(msg);

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    if (name === "node_modules" || name.startsWith(".")) continue;
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walk(p, out);
    else if ([".js", ".mjs"].includes(extname(p))) out.push(p);
  }
  return out;
}

const files = [
  ...walk(join(WEB, "js")),
  ...walk(join(WEB, "tests")),
  ...walk(join(WEB, "dev")),
  ...readdirSync(WEB)
    .filter((n) => [".js", ".mjs"].includes(extname(n)))
    .map((n) => join(WEB, n)),
];

// Blank out comments (keep newlines + string literals) so commented-out code is ignored.
function stripComments(src) {
  let out = "";
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    const d = src[i + 1];
    if (c === "/" && d === "/") {
      while (i < n && src[i] !== "\n") i++;
    } else if (c === "/" && d === "*") {
      i += 2;
      while (i < n && !(src[i] === "*" && src[i + 1] === "/")) {
        if (src[i] === "\n") out += "\n";
        i++;
      }
      i += 2;
    } else if (c === '"' || c === "'" || c === "`") {
      const q = c;
      out += c;
      i++;
      while (i < n && src[i] !== q) {
        if (src[i] === "\\") {
          out += src[i] + (src[i + 1] ?? "");
          i += 2;
          continue;
        }
        out += src[i++];
      }
      out += q;
      i++;
    } else {
      out += c;
      i++;
    }
  }
  return out;
}

const srcCache = new Map();
function source(file) {
  if (!srcCache.has(file)) srcCache.set(file, stripComments(readFileSync(file, "utf8")));
  return srcCache.get(file);
}

// Names inside `{ a, b as c, default as d }`; side: "local" | "exported" | "imported".
function parseBraceList(body, side) {
  return body
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => {
      const m = s.match(/^(?:type\s+)?([\w$]+|"[^"]+")(?:\s+as\s+([\w$]+|"[^"]+"))?$/);
      if (!m) return null;
      if (side === "exported") return m[2] || m[1];
      return m[1]; // imported name as written in the target module
    })
    .filter(Boolean);
}

const exportCache = new Map();
function exportsOf(file, seen = new Set()) {
  if (exportCache.has(file)) return exportCache.get(file);
  if (seen.has(file)) return { names: new Set(), star: false };
  seen.add(file);
  const src = source(file);
  const names = new Set();
  let re = /\bexport\s+(?:async\s+)?function\s*\*?\s*([\w$]+)/g;
  for (const m of src.matchAll(re)) names.add(m[1]);
  re = /\bexport\s+class\s+([\w$]+)/g;
  for (const m of src.matchAll(re)) names.add(m[1]);
  re = /\bexport\s+(?:const|let|var)\s+([^=;]+?)\s*=/g;
  for (const m of src.matchAll(re)) {
    const lhs = m[1].trim();
    if (lhs.startsWith("{") || lhs.startsWith("[")) {
      for (const part of lhs.replace(/^[{[]|[}\]]$/g, "").split(",")) {
        const nm = part.split(":").pop().split("=")[0].trim();
        if (nm) names.add(nm);
      }
    } else names.add(lhs);
  }
  // `export const a = 1, b = 2` (simple second declarators on the same statement)
  re = /\bexport\s+(?:const|let|var)\s+[\w$]+\s*=\s*[^;{}\[\]()]*?,\s*([\w$]+)\s*=/g;
  for (const m of src.matchAll(re)) names.add(m[1]);
  if (/\bexport\s+default\b/.test(src)) names.add("default");
  re = /\bexport\s*\{([^}]*)\}\s*(?:from\s*["']([^"']+)["'])?/g;
  for (const m of src.matchAll(re)) for (const nm of parseBraceList(m[1], "exported")) names.add(nm);
  let star = false;
  re = /\bexport\s*\*\s*(?:as\s+([\w$]+)\s*)?from\s*["']([^"']+)["']/g;
  for (const m of src.matchAll(re)) {
    if (m[1]) names.add(m[1]);
    else if (m[2].startsWith(".")) {
      const target = resolve(dirname(file), m[2]);
      if (existsSync(target)) for (const nm of exportsOf(target, seen).names) if (nm !== "default") names.add(nm);
    } else star = true;
  }
  const res = { names, star };
  exportCache.set(file, res);
  return res;
}

const lineOf = (src, idx) => src.slice(0, idx).split("\n").length;

let nSpecifiers = 0;
let nNamed = 0;
let nAssets = 0;

function checkTarget(file, spec, idx, named, kind) {
  const src = source(file);
  const where = `${rel(file)}:${lineOf(src, idx)}`;
  if (!spec.startsWith(".")) {
    if (!spec.startsWith("node:")) fail(`${where}: bare specifier "${spec}" (the site must use relative URLs only)`);
    return;
  }
  nSpecifiers++;
  const target = resolve(dirname(file), spec);
  if (!existsSync(target)) {
    fail(`${where}: ${kind} "${spec}" -> ${rel(target)} does not exist`);
    return;
  }
  if (!named.length) return;
  const ex = exportsOf(target);
  for (const nm of named) {
    nNamed++;
    if (!ex.names.has(nm) && !ex.star) fail(`${where}: "${nm}" is not exported by ${rel(target)} (exports: ${[...ex.names].sort().join(", ")})`);
  }
}

const SELF = fileURLToPath(import.meta.url);
for (const file of files) {
  if (file === SELF) continue; // this script's own regex literals would match itself
  const src = source(file);
  // static imports: import X, {a, b as c} from "..."; import * as ns from "..."; import "..."
  let re = /\bimport\s+([\w$*\s{},]*?)\s*from\s*["']([^"']+)["']/g;
  for (const m of src.matchAll(re)) {
    const clause = m[1];
    const named = [];
    const brace = clause.match(/\{([^}]*)\}/);
    if (brace) named.push(...parseBraceList(brace[1], "imported"));
    const def = clause.replace(/\{[^}]*\}/, "").replace(/\*\s*as\s+[\w$]+/, "").replace(/,/g, " ").trim();
    if (def) named.push("default");
    checkTarget(file, m[2], m.index, named, "import");
  }
  re = /\bimport\s*["']([^"']+)["']/g;
  for (const m of src.matchAll(re)) checkTarget(file, m[1], m.index, [], "side-effect import");
  // re-exports
  re = /\bexport\s*\{([^}]*)\}\s*from\s*["']([^"']+)["']/g;
  for (const m of src.matchAll(re)) checkTarget(file, m[2], m.index, parseBraceList(m[1], "imported"), "re-export");
  re = /\bexport\s*\*\s*(?:as\s+[\w$]+\s*)?from\s*["']([^"']+)["']/g;
  for (const m of src.matchAll(re)) checkTarget(file, m[1], m.index, [], "re-export");

  // dynamic imports with a literal specifier
  re = /\bimport\(\s*["']([^"']+)["']\s*\)/g;
  for (const m of src.matchAll(re)) {
    const named = [];
    const before = src.slice(Math.max(0, m.index - 200), m.index);
    const after = src.slice(m.index + m[0].length, m.index + m[0].length + 4000);
    // const { a, b: c } = await import("x")
    let d = before.match(/(?:const|let|var)\s*\{([^}]*)\}\s*=\s*await\s*$/);
    if (d) named.push(...d[1].split(",").map((s) => s.split(":")[0].trim()).filter(Boolean));
    // (await import("x")).a
    if (/\(\s*await\s*$/.test(before)) {
      const a = after.match(/^\s*\)\s*\.\s*([\w$]+)/);
      if (a) named.push(a[1]);
    }
    // const [{ a }, other] = await Promise.all([import("x"), ...])
    d = before.match(/(?:const|let|var)\s*\[\s*\{([^}]*)\}[^=]*=\s*await\s*Promise\.all\(\s*\[\s*$/);
    if (d) named.push(...d[1].split(",").map((s) => s.split(":")[0].trim()).filter(Boolean));
    // const mod = await import("x"); ... mod.a   (until mod is reassigned)
    d = before.match(/(?:const|let|var)\s+([\w$]+)\s*=\s*await\s*$/);
    if (d) {
      const v = d[1];
      const stop = after.search(new RegExp(`\\b(?:const|let|var)\\s+${v}\\b`));
      const scope = stop >= 0 ? after.slice(0, stop) : after;
      for (const a of scope.matchAll(new RegExp(`\\b${v}\\s*\\.\\s*([\\w$]+)`, "g"))) named.push(a[1]);
    }
    checkTarget(file, m[1], m.index, [...new Set(named)], "dynamic import");
  }

  // app.js-style module tables: module: "./tabs/x.js"
  re = /\bmodule\s*:\s*["'](\.{1,2}\/[^"']+\.m?js)["']/g;
  for (const m of src.matchAll(re)) checkTarget(file, m[1], m.index, ["mount"].filter(() => !m[1].endsWith("main.js")), "tab module");

  // new URL("./x", import.meta.url)
  re = /new\s+URL\(\s*["']([^"']+)["']\s*,\s*import\.meta\.url\s*\)/g;
  for (const m of src.matchAll(re)) {
    nAssets++;
    const t = resolve(dirname(file), m[1]);
    if (!existsSync(t)) fail(`${rel(file)}:${lineOf(src, m.index)}: new URL("${m[1]}") -> ${rel(t)} does not exist`);
  }
  // relative asset literals ("data/x.json" is page-relative; "../x.json" is file-relative)
  re = /["'`]((?:\.{1,2}\/)*(?:data\/)?[\w./-]+\.(?:json|csv|css|html))["'`]/g;
  for (const m of src.matchAll(re)) {
    const lit = m[1];
    if (!lit.includes("/") && !/^[\w-]+\.json$/.test(lit)) continue;
    if (/^[\w-]+\.(?:csv|html)$/.test(lit)) continue; // download filenames etc.
    const pageRel = !lit.startsWith(".");
    const t = pageRel ? join(WEB, lit) : resolve(dirname(file), lit);
    // A worker's fetch base is itself; tests read relative to themselves.
    const alt = pageRel ? resolve(dirname(file), lit) : join(WEB, lit.replace(/^(\.\.?\/)+/, ""));
    nAssets++;
    if (!existsSync(t) && !existsSync(alt)) {
      if (/^[\w-]+\.json$/.test(lit)) notes.push(`${rel(file)}:${lineOf(src, m.index)}: bare "${lit}" not found at web root (probably a filename, not a URL)`);
      else fail(`${rel(file)}:${lineOf(src, m.index)}: asset "${lit}" -> ${rel(t)} does not exist`);
    }
  }
}

// node --check on everything under web/js (and the test/dev scripts, which are cheap).
let nChecked = 0;
for (const file of files) {
  const r = spawnSync(process.execPath, ["--check", file], { encoding: "utf8" });
  nChecked++;
  if (r.status !== 0) fail(`node --check ${rel(file)} failed:\n${(r.stderr || r.stdout).trim()}`);
}

// Cross-check the regex export parser against real namespaces of Node-importable modules.
let nCross = 0;
for (const file of files.filter((f) => f.startsWith(join(WEB, "js")))) {
  // Import in a child process so a module's top-level side effects (DOM access, fetches,
  // unhandled rejections) cannot disturb this script.
  const code = `import(${JSON.stringify(pathToFileURL(file).href)}).then((ns) => { process.stdout.write("KEYS:" + JSON.stringify(Object.keys(ns))); process.exit(0); }, () => process.exit(3));`;
  const r = spawnSync(process.execPath, ["--input-type=module", "-e", code], { encoding: "utf8", timeout: 20000 });
  const m = r.status === 0 && (r.stdout || "").match(/KEYS:(\[.*\])/);
  if (!m) continue; // touches the DOM / window at top level: the regex result stands alone
  nCross++;
  const real = new Set(JSON.parse(m[1]));
  const parsed = exportsOf(file).names;
  for (const nm of real) if (!parsed.has(nm)) fail(`${rel(file)}: export "${nm}" exists but the regex parser missed it`);
  for (const nm of parsed) if (!real.has(nm)) fail(`${rel(file)}: regex parser found "${nm}" but the module does not export it`);
}

console.log(
  `check-imports: ${files.length} files, ${nSpecifiers} relative specifiers, ${nNamed} named bindings, ` +
    `${nAssets} asset URLs, ${nChecked} node --check runs, ${nCross} modules cross-checked by real import`
);
for (const n of notes) console.log(`note: ${n}`);
if (problems.length) {
  console.error(`\n${problems.length} problem(s):`);
  for (const p of problems) console.error(`  - ${p}`);
  process.exit(1);
}
console.log("check-imports: OK");
