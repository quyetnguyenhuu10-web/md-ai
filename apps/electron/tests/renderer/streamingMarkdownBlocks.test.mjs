import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

const electronRoot = findElectronRoot(process.cwd());
const sourcePath = path.join(electronRoot, "src", "renderer", "features", "chat", "streaming", "streamingMarkdownBlocks.ts");
const source = fs.readFileSync(sourcePath, "utf8");
const transpiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ES2022,
    target: ts.ScriptTarget.ES2022,
    strict: true,
  },
});
const tempPath = path.join(os.tmpdir(), `streamingMarkdownBlocks.${process.pid}.${Date.now()}.mjs`);
fs.writeFileSync(tempPath, transpiled.outputText);

try {
  const { splitStreamingMarkdownBlocks } = await import(pathToFileURL(tempPath).href);

  test("keeps empty markdown as a single block", () => {
    assert.deepEqual(splitStreamingMarkdownBlocks(""), [""]);
  });

  test("keeps unsplit markdown without blank-line boundaries", () => {
    assert.deepEqual(splitStreamingMarkdownBlocks("A\nB\nC"), ["A\nB\nC"]);
  });

  test("splits markdown after blank lines so completed blocks stay stable", () => {
    assert.deepEqual(splitStreamingMarkdownBlocks("A\n\nB\n\nC"), ["A\n\n", "B\n\n", "C"]);
  });

  test("treats whitespace-only blank lines as boundaries", () => {
    assert.deepEqual(splitStreamingMarkdownBlocks("A\n  \nB"), ["A\n  \n", "B"]);
  });

  test("preserves CRLF boundaries", () => {
    assert.deepEqual(splitStreamingMarkdownBlocks("A\r\n\r\nB"), ["A\r\n\r\n", "B"]);
  });

  console.log("streaming markdown block tests passed");
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
