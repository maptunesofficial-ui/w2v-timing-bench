// Launch the installed W2VBench app under Appium and poll its on-screen rows until DONE; write every row to $DEVICEFARM_LOG_DIR/rows.txt.
const { remote } = require('webdriverio');
const fs = require('fs');
const path = require('path');
const LOGS = process.env.DEVICEFARM_LOG_DIR || '/tmp/logs';
const SHOTS = process.env.DEVICEFARM_SCREENSHOT_PATH || '/tmp/shots';
const deadline = Date.now() + Number(process.env.BUDGET_S || 660) * 1000;
const dec = (s) => s.replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&#10;/g, '\n').replace(/&amp;/g, '&');
(async () => {
  fs.mkdirSync(LOGS, { recursive: true }); fs.mkdirSync(SHOTS, { recursive: true });
  const driver = await remote({
    hostname: '127.0.0.1', port: 4723, path: '/', logLevel: 'warn', connectionRetryTimeout: 240000,
    capabilities: {
      platformName: 'iOS', 'appium:automationName': 'XCUITest', 'appium:udid': process.env.DEVICEFARM_DEVICE_UDID,
      'appium:app': process.env.DEVICEFARM_APP_PATH, 'appium:newCommandTimeout': 900,
      'appium:usePrebuiltWDA': true, 'appium:derivedDataPath': process.env.DEVICEFARM_WDA_DERIVED_DATA_PATH, 'appium:wdaLaunchTimeout': 240000,
    },
  });
  fs.appendFileSync(path.join(LOGS, 'rows.txt'), 'SESSION ok ' + new Date().toISOString() + '\n');
  let last = '', n = 0;
  while (Date.now() < deadline) {
    let src = '';
    try { src = await driver.getPageSource(); } catch (e) { fs.appendFileSync(path.join(LOGS, 'rows.txt'), 'SRC-ERR ' + String(e).slice(0, 120) + '\n'); }
    const rows = [...src.matchAll(/label="([^"]*)"/g)].map((m) => dec(m[1])).filter((l) => /^(INFO|LOAD|ERR|R\||DP\||ACC\||DONE|starting)/.test(l));
    const text = rows.join('\n');
    if (text !== last) { fs.writeFileSync(path.join(LOGS, 'rows.txt'), text + '\n'); last = text; if (n++ % 5 === 0) await driver.saveScreenshot(path.join(SHOTS, `s${n}.png`)).catch(() => {}); }
    if (/^DONE/m.test(text)) break;
    await driver.pause(8000);
  }
  await driver.saveScreenshot(path.join(SHOTS, 'final.png')).catch(() => {});
  await driver.deleteSession().catch(() => {});
})().catch((e) => { console.log('FAIL', e); try { fs.appendFileSync(path.join(LOGS, 'rows.txt'), 'FAIL ' + String(e).slice(0, 300) + '\n'); } catch {} process.exit(1); });
