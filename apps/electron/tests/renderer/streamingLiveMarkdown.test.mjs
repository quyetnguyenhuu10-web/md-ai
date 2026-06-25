import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

const electronRoot = findElectronRoot(process.cwd());
const sourcePath = path.join(electronRoot, "src", "renderer", "features", "chat", "streaming", "streamingLiveMarkdown.ts");
const source = fs.readFileSync(sourcePath, "utf8");
const transpiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ES2022,
    target: ts.ScriptTarget.ES2022,
    strict: true,
  },
});
const tempPath = path.join(os.tmpdir(), `streamingLiveMarkdown.${process.pid}.${Date.now()}.mjs`);
fs.writeFileSync(tempPath, transpiled.outputText);

try {
  const live = await import(pathToFileURL(tempPath).href);

  test("classifies active heading block", () => {
    assert.deepEqual(live.classifyLiveMarkdownBlock("## Hello world"), {
      kind: "heading",
      level: 2,
      text: "Hello world",
    });
  });

  test("classifies active unordered list block", () => {
    assert.deepEqual(live.classifyLiveMarkdownBlock("- One\n- Two"), {
      kind: "unorderedList",
      items: ["One", "Two"],
    });
  });

  test("classifies active ordered list block with start number", () => {
    assert.deepEqual(live.classifyLiveMarkdownBlock("3. Three\n4. Four"), {
      kind: "orderedList",
      start: 3,
      items: ["Three", "Four"],
    });
  });

  test("classifies active blockquote block", () => {
    assert.deepEqual(live.classifyLiveMarkdownBlock("> A\n> B"), {
      kind: "blockquote",
      text: "A\nB",
    });
  });

  test("streaming inline window commits safe prefix and keeps suffix live", () => {
    const longText = `${"word ".repeat(60)}tail`;
    const state = live.updateStreamingInlineWindow(live.createStreamingInlineWindowState(), longText);

    assert.equal(state.chunks.length, 1);
    assert.ok(state.stableLength > 0);
    assert.equal(live.getStreamingInlineLiveText(state), longText.slice(state.stableLength));
    assert.ok(live.getStreamingInlineLiveText(state).length <= 170);
  });

  test("streaming inline window reuses chunks when appending inside live suffix", () => {
    const firstText = `${"word ".repeat(60)}tail`;
    const first = live.updateStreamingInlineWindow(live.createStreamingInlineWindowState(), firstText);
    const second = live.updateStreamingInlineWindow(first, `${firstText} more`);

    assert.equal(second.chunks[0], first.chunks[0]);
    assert.ok(live.getStreamingInlineLiveText(second).endsWith("tail more"));
  });

  test("streaming inline window avoids committing unclosed strong markdown", () => {
    const text = `**${"bold ".repeat(80)}`;
    const state = live.updateStreamingInlineWindow(live.createStreamingInlineWindowState(), text);

    assert.equal(state.chunks.length, 0);
    assert.equal(live.getStreamingInlineLiveText(state), text);
  });

  console.log("streaming live markdown tests passed");
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
