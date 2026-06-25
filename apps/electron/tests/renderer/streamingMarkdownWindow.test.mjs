import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

const electronRoot = findElectronRoot(process.cwd());
const sourceRoot = path.join(electronRoot, "src", "renderer", "features", "chat", "streaming");
const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "streamingMarkdownWindow."));
const tempStreamSegments = path.join(tempDir, "streamSegments.mjs");
const tempStreamingMarkdownBlocks = path.join(tempDir, "streamingMarkdownBlocks.mjs");
const tempStreamingMarkdownWindow = path.join(tempDir, "streamingMarkdownWindow.mjs");

try {
  fs.writeFileSync(tempStreamSegments, transpile(path.join(sourceRoot, "streamSegments.ts")));
  fs.writeFileSync(tempStreamingMarkdownBlocks, transpile(path.join(sourceRoot, "streamingMarkdownBlocks.ts")));
  fs.writeFileSync(
    tempStreamingMarkdownWindow,
    transpile(path.join(sourceRoot, "streamingMarkdownWindow.ts"))
      .replace(`from "./streamSegments"`, `from ${JSON.stringify(pathToFileURL(tempStreamSegments).href)}`)
      .replace(
        `from "./streamingMarkdownBlocks"`,
        `from ${JSON.stringify(pathToFileURL(tempStreamingMarkdownBlocks).href)}`,
      ),
  );

  const { createStreamingMarkdownWindowState, updateStreamingMarkdownWindow } = await import(
    pathToFileURL(tempStreamingMarkdownWindow).href
  );

  test("commits completed text blocks and keeps the current tail live", () => {
    const state = updateStreamingMarkdownWindow(createStreamingMarkdownWindowState(), "A\n\nB");

    assert.deepEqual(
      state.committedSegments.map((segment) => segment.kind === "text" && segment.value),
      ["A\n\n"],
    );
    assert.equal(state.liveSegments.length, 1);
    assert.equal(state.liveSegments[0].kind, "text");
    assert.equal(state.liveSegments[0].value, "B");
  });

  test("reuses committed segment array when appending inside the live tail", () => {
    const first = updateStreamingMarkdownWindow(createStreamingMarkdownWindowState(), "A\n\nB");
    const second = updateStreamingMarkdownWindow(first, "A\n\nB more");

    assert.equal(second.committedSegments, first.committedSegments);
    assert.equal(second.liveSegments[0].kind, "text");
    assert.equal(second.liveSegments[0].value, "B more");
  });

  test("promotes previous live text when a blank-line boundary arrives", () => {
    const first = updateStreamingMarkdownWindow(createStreamingMarkdownWindowState(), "A\n\nB more");
    const second = updateStreamingMarkdownWindow(first, "A\n\nB more\n\nC");

    assert.deepEqual(
      second.committedSegments.map((segment) => segment.kind === "text" && segment.value),
      ["A\n\n", "B more\n\n"],
    );
    assert.equal(second.liveSegments[0].kind, "text");
    assert.equal(second.liveSegments[0].value, "C");
  });

  test("keeps incomplete fenced code in the live tail", () => {
    const state = updateStreamingMarkdownWindow(createStreamingMarkdownWindowState(), "before\n\n```ts\nconst x = 1");

    assert.equal(state.committedSegments.length, 1);
    assert.equal(state.committedSegments[0].kind, "text");
    assert.equal(state.liveSegments.length, 1);
    assert.equal(state.liveSegments[0].kind, "fencedCode");
    assert.equal(state.liveSegments[0].complete, false);
  });

  test("commits fenced code after the closing fence arrives", () => {
    const first = updateStreamingMarkdownWindow(createStreamingMarkdownWindowState(), "before\n\n```ts\nconst x = 1");
    const second = updateStreamingMarkdownWindow(first, "before\n\n```ts\nconst x = 1\n```\n\nafter");

    assert.equal(second.committedSegments.length, 2);
    assert.equal(second.committedSegments[0].kind, "text");
    assert.equal(second.committedSegments[1].kind, "fencedCode");
    assert.equal(second.committedSegments[1].complete, true);
    assert.equal(second.liveSegments[0].kind, "text");
    assert.equal(second.liveSegments[0].value, "\nafter");
  });

  console.log("streaming markdown window tests passed");
} finally {
  fs.rmSync(tempDir, { recursive: true, force: true });
}

function transpile(sourcePath) {
  const source = fs.readFileSync(sourcePath, "utf8");
  return ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.ES2022,
      target: ts.ScriptTarget.ES2022,
      strict: true,
    },
  }).outputText;
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
