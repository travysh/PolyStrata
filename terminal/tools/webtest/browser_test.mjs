// Browser tests of the served web page (serve/web) in headless Chrome or Edge, against the mock server in mock.mjs.
// No model, no GPU. Optional: the normal test run (serve/test_web_*.py, node serve/test_web_*.js) does not need a browser.
//
//   WEBTEST_MODULES=<folder with node_modules/puppeteer-core> [CHROME_BIN=<chrome>] node tools/webtest/browser_test.mjs [name]
import assert from "node:assert/strict";
import { startMock } from "./mock.mjs";
import { launch, openPage, sleep, runner, pageErrors } from "./lib.mjs";

const { test, run } = runner("web page browser tests");

// ------------------------------------------------------------------------------------------------ load / unload (#3550)
const btnText = (page) => page.$eval("#unload-btn", (b) => b.textContent);
const waitBtn = (page, text) => page.waitForFunction((t) => document.getElementById("unload-btn").textContent === t, { timeout: 8000 }, text);
const posts = (mock, what) => mock.state.calls.filter((c) => c === "POST " + what).length;

test("load/unload: the header button unloads after a confirmation and loads again", async ({ browser, mock }) => {
  mock.state.loaded = true; mock.state.calls.length = 0;
  const page = await openPage(browser, mock);
  await waitBtn(page, "Unload");
  await page.click("#unload-btn");
  assert.match(page.lastDialog, /Unload the model and free the GPU/);
  await waitBtn(page, "Load");
  assert.equal(posts(mock, "/unload"), 1);
  assert.equal(await page.$eval("#pill-text", (e) => e.textContent), "Unloaded");
  await sleep(2500);                                       // the poll must agree, not flip the button back
  assert.equal(await btnText(page), "Load");
  await page.click("#unload-btn");
  await waitBtn(page, "Unload");
  assert.equal(posts(mock, "/load"), 1);
  assert.equal(await page.$eval("#pill-text", (e) => e.textContent), "Idle");
  await page.close();
});

test("load/unload: declining the question changes nothing", async ({ browser, mock }) => {
  mock.state.loaded = true; mock.state.calls.length = 0;
  const page = await openPage(browser, mock, { dialogs: "dismiss" });
  await waitBtn(page, "Unload");
  await page.click("#unload-btn");
  await sleep(500);
  assert.equal(posts(mock, "/unload"), 0);
  assert.equal(await btnText(page), "Unload");
  await page.close();
});

test("load/unload: the header fits a phone", async ({ browser, mock }) => {
  mock.state.loaded = true;
  for (const width of [320, 360, 390, 640]) {
    const page = await openPage(browser, mock, { width, height: 700 });
    await waitBtn(page, "Unload");
    const r = await page.evaluate(() => {
      const box = (id) => { const b = document.getElementById(id).getBoundingClientRect(); return [b.left, b.right]; };
      return { unload: box("unload-btn"), theme: box("theme-btn"), scroll: document.documentElement.scrollWidth, width: innerWidth };
    });
    for (const [l, rt] of [r.unload, r.theme]) assert.ok(l >= 0 && rt <= r.width, `${width}px: ${JSON.stringify(r)}`);
    assert.ok(r.scroll <= r.width, `${width}px: the page scrolls sideways: ${JSON.stringify(r)}`);
    await page.close();
  }
});

// ------------------------------------------------------------------------------------------------ the model list and Requests (#142)
const withMock = async (opts, fn) => { const m = await startMock(opts); try { await fn(m); } finally { await m.stop(); } };
const toastTitles = (page) => page.$$eval(".st-toast__title", (e) => e.map((x) => x.textContent));

test("models: one installed model, no list; several, a list; choosing one loads it", async ({ browser, mock }) => {
  const one = await openPage(browser, mock);
  await one.waitForFunction(() => document.querySelector("#model-select").dataset.current === "Mock-Model");
  assert.equal(await one.$eval("#model-select", (e) => e.hidden), true, "nothing to choose from");
  await one.close();
  await withMock({ models: true }, async (m) => {
    const page = await openPage(browser, m);
    await page.waitForFunction(() => !document.querySelector("#model-select").hidden);
    assert.deepEqual(await page.$$eval("#model-select option", (o) => o.map((x) => [x.value, x.disabled, x.textContent])),
                     [["Mock-Model", false, "Mock-Model"], ["Other-Model", false, "Other Model (Other-Model)"], ["Gone-Model", true, "Gone Model (Gone-Model) - unavailable"]]);
    await page.select("#model-select", "Other-Model");
    await page.waitForFunction(() => document.querySelector("#model-select").dataset.current === "Other-Model");
    assert.ok(m.state.calls.includes("POST /api/models/load"));
    assert.ok((await toastTitles(page)).includes("Other-Model is loaded"));
    // a switch the server refuses leaves the list as it was
    m.state.failLoad = true;
    await page.select("#model-select", "Mock-Model");
    await page.waitForFunction(() => [...document.querySelectorAll(".st-toast__title")].some((t) => t.textContent === "The model was not switched"));
    await page.waitForFunction(() => !document.querySelector("#model-select").disabled);
    assert.equal(await page.$eval("#model-select", (e) => e.value), "Other-Model");
    assert.deepEqual(pageErrors(page).filter((e) => !/ 409 .*\/api\/models\/load\)$/.test(e)), [], "(the refused switch logs its 409)");
    await page.close();
  });
});

test("models: the header with the model list fits a phone", async ({ browser }) => {
  await withMock({ models: true }, async (m) => {
    for (const width of [320, 360, 390, 420, 458, 500, 560, 640, 768]) {
      const page = await openPage(browser, m, { width, height: 700 });
      await page.waitForFunction(() => !document.querySelector("#model-select").hidden);
      await waitBtn(page, "Unload");
      const r = await page.evaluate(() => {
        const box = (id) => { const b = document.getElementById(id).getBoundingClientRect(); return [b.left, b.right]; };
        return { select: box("model-select"), unload: box("unload-btn"), theme: box("theme-btn"), scroll: document.documentElement.scrollWidth, width: innerWidth };
      });
      for (const [l, rt] of [r.select, r.unload, r.theme]) assert.ok(l >= 0 && rt <= r.width, `${width}px: ${JSON.stringify(r)}`);
      assert.ok(r.select[1] - r.select[0] >= 60, `${width}px: the list is too narrow to read: ${JSON.stringify(r)}`);
      assert.ok(r.scroll <= r.width, `${width}px: the page scrolls sideways: ${JSON.stringify(r)}`);
      await page.close();
    }
  });
});

test("requests: the tab says how to turn the API monitor on; on, it opens the monitor in a tab of its own, with the saved key", async ({ browser, mock }) => {
  const page = await openPage(browser, mock);
  assert.equal(await page.$eval("#tab-btn-requests", (e) => e.hidden), false, "the tab is always there");
  await page.click("#tab-btn-requests");
  assert.equal(await page.$eval("#requests-off", (e) => e.hidden), false, "off: how to turn it on");
  assert.equal(await page.$eval("#requests-on", (e) => e.hidden), true);
  await page.close();
  await withMock({ apiMonitor: true }, async (m) => {
    const page2 = await openPage(browser, m, { seed: { apikey: "saved-key" }, hash: "#requests" });
    assert.equal(await page2.$eval("#requests-off", (e) => e.hidden), true);
    assert.equal(await page2.$eval("#requests-on", (e) => e.hidden), false);
    assert.equal(await page2.$eval("#requests-open", (e) => [e.getAttribute("href"), e.target].join(" ")), "api-monitor _blank");
    assert.equal(await page2.$("iframe"), null, "no frame: the page's policy allows none");
    await page2.click("#requests-open");
    const monitor = await (await page2.browserContext().waitForTarget((t) => t.url().endsWith("/api-monitor"))).page();
    await monitor.waitForFunction(() => document.getElementById("updated").textContent.startsWith("Live"));
    assert.equal(m.state.monitorAuth, "Bearer saved-key", "the key saved on this page is not asked for again");
    assert.equal(await monitor.$eval("#auth", (e) => e.hidden), true);
    assert.deepEqual(pageErrors(page2), []);
    await monitor.close();
    await page2.close();
  });
});

const mock = await startMock();
const browser = await launch();
await run(async () => ({ browser, mock, close: async () => { await browser.close(); await mock.stop(); } }));
