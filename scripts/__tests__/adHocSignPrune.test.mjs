import { test } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { pruneForeignOnnxBinaries } = require('../ad-hoc-sign.js');

const COMBOS = ['darwin/x64', 'darwin/arm64', 'linux/x64', 'linux/arm64', 'win32/x64', 'win32/arm64'];

function makeFixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'onnxprune-'));
  const bin = path.join(root, 'Contents/Resources/app.asar.unpacked/node_modules/onnxruntime-node/bin');
  for (const combo of COMBOS) {
    const d = path.join(bin, 'napi-v6', combo);
    fs.mkdirSync(d, { recursive: true });
    fs.writeFileSync(path.join(d, 'onnxruntime_binding.node'), Buffer.alloc(1024));
  }
  return { root, bin };
}

function survivors(bin) {
  const out = [];
  (function walk(d) {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) walk(p);
      else out.push(path.relative(bin, p));
    }
  })(bin);
  return out;
}

test('keeps only darwin/<target> and removes every foreign platform/arch, idempotently', () => {
  const { root, bin } = makeFixture();
  try {
    pruneForeignOnnxBinaries(root, 'arm64');
    assert.deepStrictEqual(survivors(bin), ['napi-v6/darwin/arm64/onnxruntime_binding.node']);
    pruneForeignOnnxBinaries(root, 'arm64'); // re-run must not throw
    assert.deepStrictEqual(survivors(bin), ['napi-v6/darwin/arm64/onnxruntime_binding.node']);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('x64 target keeps darwin/x64 instead', () => {
  const { root, bin } = makeFixture();
  try {
    pruneForeignOnnxBinaries(root, 'x64');
    assert.deepStrictEqual(survivors(bin), ['napi-v6/darwin/x64/onnxruntime_binding.node']);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('unknown target arch is a no-op (safety guard)', () => {
  const { root, bin } = makeFixture();
  try {
    pruneForeignOnnxBinaries(root, 'universal');
    assert.strictEqual(survivors(bin).length, COMBOS.length);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
