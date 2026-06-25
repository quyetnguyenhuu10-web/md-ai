import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

const electronRoot = findElectronRoot(process.cwd());
const sourcePath = path.join(electronRoot, "src", "main", "modules", "utilities", "RenderCacheDiskService.ts");
const source = fs
  .readFileSync(sourcePath, "utf8")
  .replace('import { getDataDir } from "../../platform/electronPaths";', 'const getDataDir = () => process.cwd();');
const transpiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ES2022,
    target: ts.ScriptTarget.ES2022,
    strict: true,
  },
});
const tempModulePath = path.join(os.tmpdir(), `RenderCacheDiskService.${process.pid}.${Date.now()}.mjs`);
fs.writeFileSync(tempModulePath, transpiled.outputText);

const tempDataDir = fs.mkdtempSync(path.join(os.tmpdir(), "render-cache-disk-test-"));

try {
  const { RenderCacheDiskService } = await import(pathToFileURL(tempModulePath).href);

  test("putArtifact writes readable artifacts and stats", () => {
    const service = new RenderCacheDiskService(tempDataDir);
    const heightArtifact = makeHeightArtifact("height-key", 120);
    const codeArtifact = makeCodeArtifact("code-key", "<pre><code>ok</code></pre>");

    service.putArtifact(heightArtifact);
    service.putArtifact(codeArtifact);

    const artifacts = service.getArtifacts(["height-key", "code-key"]);
    assert.deepEqual(
      artifacts.map((artifact) => artifact.key).sort(),
      ["code-key", "height-key"],
    );
    assert.equal(service.getStats().artifactCount, 2);
    assert.equal(service.getStats().warningLevel, "ok");
  });

  test("invalid JSON artifacts are ignored without crashing", () => {
    const service = new RenderCacheDiskService(tempDataDir);
    const key = "invalid-json-key";
    const artifactDir = path.join(tempDataDir, "render-cache", "artifacts");
    fs.mkdirSync(artifactDir, { recursive: true });
    fs.writeFileSync(path.join(artifactDir, `${sha256(key)}.json`), "{", "utf8");

    assert.deepEqual(service.getArtifacts([key]), []);
  });

  test("clear removes only render cache artifacts and refreshes stats", () => {
    const service = new RenderCacheDiskService(tempDataDir);
    const cacheRoot = path.join(tempDataDir, "render-cache");
    const siblingPath = path.join(cacheRoot, "keep.txt");

    service.putArtifact(makeHeightArtifact("clear-height-key", 180));
    fs.mkdirSync(cacheRoot, { recursive: true });
    fs.writeFileSync(siblingPath, "keep", "utf8");

    const before = service.getStats();
    assert.equal(before.artifactCount > 0, true);

    const after = service.clear();

    assert.equal(after.artifactCount, 0);
    assert.equal(after.diskEstimatedBytes, 0);
    assert.equal(after.warningLevel, "ok");
    assert.equal(fs.existsSync(siblingPath), true);
    assert.deepEqual(service.getArtifacts(["clear-height-key"]), []);
  });

  console.log("render cache disk service tests passed");
} finally {
  fs.rmSync(tempDataDir, { recursive: true, force: true });
  fs.rmSync(tempModulePath, { force: true });
}

function makeHeightArtifact(key, height) {
  return {
    key,
    kind: "messageHeight",
    rendererVersion: "message-height-v1",
    contentHash: "hash",
    conversationId: "conv",
    messageId: "msg",
    payload: { height },
    estimatedBytes: 96,
    updatedAt: Date.now(),
  };
}

function makeCodeArtifact(key, html) {
  return {
    key,
    kind: "codeHighlightHtml",
    rendererVersion: "code-highlight-v1",
    contentHash: "hash",
    payload: { html },
    estimatedBytes: html.length,
    updatedAt: Date.now(),
  };
}

function sha256(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
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
    if (fs.existsSync(path.join(current, "package.json")) && fs.existsSync(path.join(current, "src", "main"))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      throw new Error("Could not find apps/electron root");
    }
    current = parent;
  }
}
