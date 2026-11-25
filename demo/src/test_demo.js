// usage: node test_demo.js <html> <shots_dir>
const { chromium } = require('playwright');
(async () => {
  const [file, out] = process.argv.slice(2);
  const browser = await chromium.launch();
  const errors = [];
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1 });
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errors.push(m.type() + ': ' + m.text()); });
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  await page.goto('file://' + file);
  await page.waitForTimeout(600);
  const t0 = await page.textContent('#total');
  await page.screenshot({ path: out + '/1_layout.png' });
  // drag buoy 10 by 120 px left
  const pos = await page.evaluate(() => { const g = window.__wfm.geom(), f = window.__wfm.farm, r = document.getElementById('farm').getBoundingClientRect(); return { x: r.left + g.X(f.x[10]), y: r.top + g.Y(f.y[10]) }; });
  await page.mouse.move(pos.x, pos.y); await page.mouse.down(); await page.mouse.move(pos.x - 60, pos.y + 30, { steps: 8 }); await page.mouse.move(pos.x - 120, pos.y + 60, { steps: 8 }); await page.mouse.up();
  await page.waitForTimeout(200);
  const t1 = await page.textContent('#total'); const truth = await page.textContent('#truth');
  console.log('total before', t0, 'after drag', t1, '|', truth.slice(0, 60));
  await page.screenshot({ path: out + '/1b_dragged.png' });
  // optimise from best layout
  await page.click('#presets button:nth-child(1)');
  const t2 = Date.now();
  await page.click('#optimise');
  await page.waitForFunction(() => document.getElementById('opt-status').textContent.startsWith('Done'), null, { timeout: 30000 });
  console.log('optimise took', Date.now() - t2, 'ms; total', await page.textContent('#total'));
  console.log('claim:', (await page.textContent('#claim')).replace(/\s+/g, ' ').slice(0, 400));
  await page.screenshot({ path: out + '/1c_optimised.png', fullPage: true });
  for (const v of ['testing', 'loop', 'sites']) {
    await page.evaluate(h => { location.hash = h; }, v);
    await page.waitForTimeout(500);
    await page.screenshot({ path: out + `/2_${v}.png`, fullPage: true });
  }
  await page.evaluate(() => { location.hash = 'testing'; }); await page.waitForTimeout(700);
  await page.click('[data-twin="b"]'); await page.waitForTimeout(800);
  await page.screenshot({ path: out + '/2_testing_b.png', fullPage: true });
  await page.click('[data-site="sydney"]'); await page.waitForTimeout(200);
  await page.screenshot({ path: out + '/2_testing_sydney.png', clip: { x: 800, y: 80, width: 800, height: 820 } });
  // dark theme
  await page.click('#theme'); await page.waitForTimeout(400);
  await page.evaluate(() => { location.hash = 'sites'; }); await page.waitForTimeout(500);
  await page.screenshot({ path: out + '/3_dark_sites.png', fullPage: true });
  await page.evaluate(() => { location.hash = 'layout'; }); await page.waitForTimeout(500);
  await page.screenshot({ path: out + '/3_dark_layout.png' });
  // phone
  const ph = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
  ph.on('pageerror', e => errors.push('phone pageerror: ' + e.message));
  await ph.goto('file://' + file); await ph.waitForTimeout(500);
  for (const v of ['layout', 'testing', 'loop', 'sites']) {
    await ph.evaluate(h => { location.hash = h; }, v); await ph.waitForTimeout(400);
    const sw = await ph.evaluate(() => [document.documentElement.scrollWidth, innerWidth]);
    console.log('phone', v, 'scrollWidth', sw[0], 'innerWidth', sw[1]);
    await ph.screenshot({ path: out + `/4_phone_${v}.png`, fullPage: true });
  }
  console.log('errors:', errors.length ? errors : 'none');
  await browser.close();
})();
