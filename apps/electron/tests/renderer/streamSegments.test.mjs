import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

const electronRoot = findElectronRoot(process.cwd());
const sourcePath = path.join(electronRoot, "src", "renderer", "features", "chat", "streaming", "streamSegments.ts");
const source = fs.readFileSync(sourcePath, "utf8");
const transpiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ES2022,
    target: ts.ScriptTarget.ES2022,
    strict: true,
  },
});
const tempPath = path.join(os.tmpdir(), `streamSegments.${process.pid}.${Date.now()}.mjs`);
fs.writeFileSync(tempPath, transpiled.outputText);

try {
  const { DISPLAY_MATH_CLOSE, DISPLAY_MATH_DOLLAR, DISPLAY_MATH_OPEN, splitProtectedStreamSegments } = await import(pathToFileURL(tempPath).href);

  test("display math is split with exact backslash-square markers", () => {
    assert.equal(DISPLAY_MATH_OPEN, "\\[");
    assert.equal(DISPLAY_MATH_CLOSE, "\\]");
    assert.deepEqual(compact(splitProtectedStreamSegments("A \\[x+1\\] B")), [
      { kind: "text", value: "A " },
      { kind: "displayMath", value: "x+1", raw: "\\[x+1\\]", complete: true },
      { kind: "text", value: " B" },
    ]);
  });

  test("plain square brackets are not math", () => {
    assert.deepEqual(compact(splitProtectedStreamSegments("A [x+1] B")), [{ kind: "text", value: "A [x+1] B" }]);
  });

  test("invented slash markers are not math", () => {
    assert.deepEqual(compact(splitProtectedStreamSegments("A /[\\] B")), [{ kind: "text", value: "A /[\\] B" }]);
  });

  test("incomplete display math stays protected", () => {
    assert.deepEqual(compact(splitProtectedStreamSegments("A \\[x+1")), [
      { kind: "text", value: "A " },
      { kind: "displayMath", value: "x+1", raw: "\\[x+1", complete: false },
    ]);
  });

  test("dollar display math block is split", () => {
    assert.equal(DISPLAY_MATH_DOLLAR, "$$");
    const input = "A\n$$\nx+1\n$$\nB";
    assert.deepEqual(compact(splitProtectedStreamSegments(input)), [
      { kind: "text", value: "A\n" },
      { kind: "displayMath", value: "x+1\n", raw: "$$\nx+1\n$$", complete: true },
      { kind: "text", value: "\nB" },
    ]);
  });

  test("inline dollars are not display math blocks", () => {
    assert.deepEqual(compact(splitProtectedStreamSegments("A $x+1$ B")), [{ kind: "text", value: "A $x+1$ B" }]);
  });

  test("incomplete dollar display math block stays protected", () => {
    assert.deepEqual(compact(splitProtectedStreamSegments("A\n$$\nx+1")), [
      { kind: "text", value: "A\n" },
      { kind: "displayMath", value: "x+1", raw: "$$\nx+1", complete: false },
    ]);
  });

  test("backtick fenced code block with language is split", () => {
    const input = "A\n```ts\nconst x = 1;\n```\nB";
    assert.deepEqual(compact(splitProtectedStreamSegments(input)), [
      { kind: "text", value: "A\n" },
      { kind: "fencedCode", language: "ts", code: "const x = 1;\n", raw: "```ts\nconst x = 1;\n```\n", fence: "```", complete: true },
      { kind: "text", value: "B" },
    ]);
  });

  test("tilde fenced code block is split", () => {
    const input = "A\n~~~python\nprint(1)\n~~~\nB";
    assert.deepEqual(compact(splitProtectedStreamSegments(input)), [
      { kind: "text", value: "A\n" },
      { kind: "fencedCode", language: "python", code: "print(1)\n", raw: "~~~python\nprint(1)\n~~~\n", fence: "~~~", complete: true },
      { kind: "text", value: "B" },
    ]);
  });

  test("display math markers inside code are literal", () => {
    const input = 'A\n```ts\nconst x = "\\\\[not math\\\\]";\n```\nB';
    assert.deepEqual(compact(splitProtectedStreamSegments(input)), [
      { kind: "text", value: "A\n" },
      {
        kind: "fencedCode",
        language: "ts",
        code: 'const x = "\\\\[not math\\\\]";\n',
        raw: '```ts\nconst x = "\\\\[not math\\\\]";\n```\n',
        fence: "```",
        complete: true,
      },
      { kind: "text", value: "B" },
    ]);
  });

  test("dollar display math markers inside code are literal", () => {
    const input = 'A\n```ts\nconst x = "$$not math$$";\n```\nB';
    assert.deepEqual(compact(splitProtectedStreamSegments(input)), [
      { kind: "text", value: "A\n" },
      {
        kind: "fencedCode",
        language: "ts",
        code: 'const x = "$$not math$$";\n',
        raw: '```ts\nconst x = "$$not math$$";\n```\n',
        fence: "```",
        complete: true,
      },
      { kind: "text", value: "B" },
    ]);
  });

  test("incomplete code fence stays protected", () => {
    const input = "A\n```ts\nconst x = 1;";
    assert.deepEqual(compact(splitProtectedStreamSegments(input)), [
      { kind: "text", value: "A\n" },
      { kind: "fencedCode", language: "ts", code: "const x = 1;", raw: "```ts\nconst x = 1;", fence: "```", complete: false },
    ]);
  });

  test("code fence split by chunks is correct once source is complete", () => {
    const sourceFromChunks = ["A\n`", "``t", "s\nconst x = 1;\n", "``", "`\nB"].join("");
    assert.deepEqual(compact(splitProtectedStreamSegments(sourceFromChunks)), [
      { kind: "text", value: "A\n" },
      { kind: "fencedCode", language: "ts", code: "const x = 1;\n", raw: "```ts\nconst x = 1;\n```\n", fence: "```", complete: true },
      { kind: "text", value: "B" },
    ]);
  });

  test("multiple mixed blocks stay ordered", () => {
    const input = "A\n\\[\nx\n\\]\nB\n```ts\nconst y = 1;\n```\nC";
    assert.deepEqual(compact(splitProtectedStreamSegments(input)), [
      { kind: "text", value: "A\n" },
      { kind: "displayMath", value: "\nx\n", raw: "\\[\nx\n\\]", complete: true },
      { kind: "text", value: "\nB\n" },
      { kind: "fencedCode", language: "ts", code: "const y = 1;\n", raw: "```ts\nconst y = 1;\n```\n", fence: "```", complete: true },
      { kind: "text", value: "C" },
    ]);
  });

  test("empty code block does not crash", () => {
    assert.deepEqual(compact(splitProtectedStreamSegments("```\n```\n")), [
      { kind: "fencedCode", language: "", code: "", raw: "```\n```\n", fence: "```", complete: true },
    ]);
  });

  test("empty display math does not crash", () => {
    assert.deepEqual(compact(splitProtectedStreamSegments("\\[\\]")), [
      { kind: "displayMath", value: "", raw: "\\[\\]", complete: true },
    ]);
  });

  test("empty dollar display math does not crash", () => {
    assert.deepEqual(compact(splitProtectedStreamSegments("$$\n$$")), [
      { kind: "displayMath", value: "", raw: "$$\n$$", complete: true },
    ]);
  });

  console.log("stream segment parser tests passed");
} finally {
  fs.rmSync(tempPath, { force: true });
}

function compact(segments) {
  return segments.map((segment) => ({ ...segment }));
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
