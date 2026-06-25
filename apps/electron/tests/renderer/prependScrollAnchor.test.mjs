import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

const electronRoot = findElectronRoot(process.cwd());
const sourcePath = path.join(electronRoot, "src", "renderer", "features", "chat", "prependScrollAnchor.ts");
const source = fs.readFileSync(sourcePath, "utf8");
const transpiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ES2022,
    target: ts.ScriptTarget.ES2022,
    strict: true,
  },
});
const tempPath = path.join(os.tmpdir(), `prependScrollAnchor.${process.pid}.${Date.now()}.mjs`);
fs.writeFileSync(tempPath, transpiled.outputText);

try {
  const { capturePrependScrollSnapshot, getRestoredScrollTopAfterPrepend } = await import(pathToFileURL(tempPath).href);

  test("captures scroll height and scroll top before prepend", () => {
    assert.deepEqual(capturePrependScrollSnapshot({ scrollHeight: 1000, scrollTop: 80 }), {
      scrollHeight: 1000,
      scrollTop: 80,
    });
  });

  test("restores scroll top by adding prepended height", () => {
    const snapshot = { scrollHeight: 1000, scrollTop: 80 };
    assert.equal(getRestoredScrollTopAfterPrepend(snapshot, 1450), 530);
  });

  test("does not move upward if scroll height shrinks during reflow", () => {
    const snapshot = { scrollHeight: 1000, scrollTop: 80 };
    assert.equal(getRestoredScrollTopAfterPrepend(snapshot, 900), 80);
  });

  console.log("prepend scroll anchor tests passed");
} finally {
  fs.rmSync(tempPath, { force: true });
}

function test(name, run) {
  try {
    run();
  } catch (error) {
    error.message = `${name}: ${error.message}`;
    throw error;
  }
}

function findElectronRoot(start) {
  let current = path.resolve(start);
  while (true) {
    if (fs.existsSync(path.join(current, "package.json")) && fs.existsSync(path.join(current, "src", "renderer"))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      throw new Error("Could not find apps/electron root");
    }
    current = parent;
  }
}
