// Shared bits for the browser tests of the served web page (serve/web): a headless Chromium,
// a page against the mock server, small helpers. puppeteer-core is not a dependency of the repo: install it anywhere and
// point WEBTEST_MODULES at that folder (the one that holds node_modules); CHROME_BIN is the Chrome or Edge to drive.
import { createRequire } from "node:module";
import path from "node:path";
import fs from "node:fs";

const req = createRequire(path.join(process.env.WEBTEST_MODULES || process.cwd(), "noop.js"));
export const puppeteer = req("puppeteer-core");

const CANDIDATES = [process.env.CHROME_BIN, "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].filter(Boolean);
export const chrome = () => {
  const found = CANDIDATES.find((p) => fs.existsSync(p));
  if (!found) throw new Error("no Chrome or Edge found; set CHROME_BIN");
  return found;
};
export const launch = () => puppeteer.launch({ executablePath: chrome(), headless: true, args: ["--no-sandbox", "--disable-gpu"] });

// a page with a seeded browser store (localStorage values are JSON, as the page's own store writes them), console errors collected
export async function openPage(browser, mock, { width = 1280, height = 800, seed = {}, query = "", dark = false, hash = "", dialogs = "accept" } = {}) {
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  page.errors = [];
  page.on("pageerror", (e) => page.errors.push("pageerror: " + e.message));
  // (with the address it is about: "Failed to load resource" alone does not say which one)
  page.on("console", (m) => { if (m.type() === "error") page.errors.push("console: " + m.text() + (m.location().url ? ` (${m.location().url})` : "")); });
  page.on("dialog", async (d) => { page.lastDialog = d.message(); if (dialogs === "accept") await d.accept(); else await d.dismiss(); });
  await page.setViewport({ width, height });
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: dark ? "dark" : "light" }]);
  await page.evaluateOnNewDocument((s) => {
    if (window !== window.top) return;                  // not in the preview frames: they have no storage
    if (sessionStorage.getItem("seeded")) return;
    sessionStorage.setItem("seeded", "1");
    for (const [k, v] of Object.entries(s)) localStorage.setItem("strata." + k, JSON.stringify(v));
  }, seed);
  await page.goto(mock.url + "/" + query + hash, { waitUntil: "networkidle0" });
  return page;
}

// The page's errors less the one every page load logs on a model with no expert profile: the Experts probe's GET api/expert-heat
// is answered 404 (serve/expert_heat.py), the page reads that as "no Experts tab", and the browser still logs the 404.
export const pageErrors = (page) => page.errors.filter((e) => !/^console: Failed to load resource: .* 404 .*\/api\/expert-heat\)$/.test(e));

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// tiny test runner: node file.mjs [name-substring]
export function runner(title) {
  const tests = [];
  const only = process.argv[2];
  return {
    test: (name, fn) => tests.push([name, fn]),
    async run(setup) {
      const ctx = await setup();
      let failed = 0;
      for (const [name, fn] of tests) {
        if (only && !name.includes(only)) continue;
        const t = Date.now();
        try { await fn(ctx); console.log(`ok   ${name} (${Date.now() - t} ms)`); }
        catch (e) { failed++; console.log(`FAIL ${name}\n     ${String(e.stack || e).split("\n").slice(0, 6).join("\n     ")}`); }
      }
      await ctx.close();
      console.log(failed ? `${title}: ${failed} failed` : `${title}: all passed`);
      // exit now: a test that failed before stopping a mock server of its own leaves it listening, and node would wait on it
      process.exit(failed ? 1 : 0);
    },
  };
}
