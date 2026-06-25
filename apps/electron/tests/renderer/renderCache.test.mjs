import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

const electronRoot = findElectronRoot(process.cwd());
const sourcePath = path.join(electronRoot, "src", "renderer", "features", "chat", "renderCache.ts");
const source = fs.readFileSync(sourcePath, "utf8");
const transpiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ES2022,
    target: ts.ScriptTarget.ES2022,
    strict: true,
  },
});
const tempPath = path.join(os.tmpdir(), `renderCache.${process.pid}.${Date.now()}.mjs`);
fs.writeFileSync(tempPath, transpiled.outputText);

try {
  const cache = await import(pathToFileURL(tempPath).href);

  await test("code highlight cache evicts entries and metadata together", async () => {
    cache.clearHotRenderCaches();
    installChatApi();

    for (let index = 0; index < 260; index += 1) {
      const key = cache.makeCodeHighlightCacheKey({
        code: `const value${index} = ${index};`,
        language: "ts",
      });
      cache.setCachedCodeHighlightHtml(key, `<pre><code>${index}</code></pre>`);
    }

    const stats = cache.getHotRenderCacheStats();
    assert.equal(stats.codeHighlightEntries, 240);
    assert.equal(stats.codeHighlightMetadataEntries, 240);
  });

  await test("persisted artifact signatures dedupe repeated writes", async () => {
    cache.clearHotRenderCaches();
    const writes = [];
    installChatApi({ writes });

    const key = cache.makeCodeHighlightCacheKey({
      code: "const answer = 42;",
      language: "ts",
    });

    cache.setCachedCodeHighlightHtml(key, "<pre><code>one</code></pre>");
    cache.setCachedCodeHighlightHtml(key, "<pre><code>one</code></pre>");
    assert.equal(writes.length, 1);

    cache.setCachedCodeHighlightHtml(key, "<pre><code>two</code></pre>");
    assert.equal(writes.length, 2);
    assert.equal(cache.getHotRenderCacheStats().persistedArtifactSignatureEntries, 1);
  });

  await test("persisted artifact signatures are bounded", async () => {
    cache.clearHotRenderCaches();
    installChatApi();

    for (let index = 0; index < 4105; index += 1) {
      const key = cache.makeCodeHighlightCacheKey({
        code: `const bounded${index} = ${index};`,
        language: "ts",
      });
      cache.setCachedCodeHighlightHtml(key, `<pre><code>${index}</code></pre>`);
    }

    assert.equal(cache.getHotRenderCacheStats().persistedArtifactSignatureEntries, 4000);
  });

  await test("disk code highlight hydrate rejects invalid payloads", async () => {
    cache.clearHotRenderCaches();
    const key = cache.makeCodeHighlightCacheKey({ code: "console.log(1);", language: "js" });
    installChatApi({
      artifacts: [
        {
          key,
          kind: "codeHighlightHtml",
          rendererVersion: cache.CODE_HIGHLIGHT_CACHE_VERSION,
          contentHash: "hash",
          payload: { html: 123 },
          estimatedBytes: 123,
        },
      ],
    });

    assert.equal(await cache.getDiskCachedCodeHighlightHtml(key), null);
    assert.equal(cache.getHotRenderCacheStats().codeHighlightEntries, 0);
  });

  await test("disk code highlight hydrate accepts valid HTML strings", async () => {
    cache.clearHotRenderCaches();
    const key = cache.makeCodeHighlightCacheKey({ code: "console.log(2);", language: "js" });
    const html = "<pre><code>console.log(2);</code></pre>";
    installChatApi({
      artifacts: [
        {
          key,
          kind: "codeHighlightHtml",
          rendererVersion: cache.CODE_HIGHLIGHT_CACHE_VERSION,
          contentHash: "hash",
          payload: { html },
          estimatedBytes: html.length,
        },
      ],
    });

    assert.equal(await cache.getDiskCachedCodeHighlightHtml(key), html);
    assert.equal(cache.getCachedCodeHighlightHtml(key), html);
  });

  await test("message height hydrate only accepts stable messages with valid heights", async () => {
    cache.clearHotRenderCaches();
    const complete = makeMessage({ id: "m1", status: "complete", content: "complete" });
    const streaming = makeMessage({ id: "m2", status: "streaming", content: "streaming" });
    const completeKey = cache.makeMessageHeightCacheKey(complete);
    const requestedKeys = [];

    installChatApi({
      getArtifacts: async (keys) => {
        requestedKeys.push(...keys);
        return [
          {
            key: completeKey,
            kind: "messageHeight",
            rendererVersion: cache.MESSAGE_HEIGHT_CACHE_VERSION,
            contentHash: "hash",
            conversationId: complete.conversationId,
            messageId: complete.id,
            payload: { height: 120.1 },
            estimatedBytes: 96,
          },
          {
            key: "unknown",
            kind: "messageHeight",
            rendererVersion: cache.MESSAGE_HEIGHT_CACHE_VERSION,
            contentHash: "hash",
            payload: { height: -1 },
            estimatedBytes: 96,
          },
        ];
      },
    });

    const hydrated = await cache.hydrateCachedMessageHeightsFromDisk([complete, streaming]);
    assert.deepEqual(requestedKeys, [completeKey]);
    assert.equal(hydrated.get(complete.id), 121);
    assert.equal(hydrated.has(streaming.id), false);
    assert.equal(cache.getCachedMessageHeight(complete), 121);
  });

  await test("clearHotRenderCaches clears entries, metadata, and signatures", async () => {
    cache.clearHotRenderCaches();
    installChatApi();
    const key = cache.makeCodeHighlightCacheKey({ code: "const clearMe = true;", language: "ts" });
    cache.setCachedCodeHighlightHtml(key, "<pre><code>clear</code></pre>");
    cache.setCachedMessageHeight(makeMessage({ id: "m3", status: "complete" }), 99);

    cache.clearHotRenderCaches();

    assert.deepEqual(cache.getHotRenderCacheStats(), {
      renderCacheVersion: cache.RENDER_CACHE_VERSION,
      codeHighlightEntries: 0,
      codeHighlightMetadataEntries: 0,
      codeHighlightEstimatedBytes: 0,
      messageHeightEntries: 0,
      messageHeightEstimatedBytes: 0,
      persistedArtifactSignatureEntries: 0,
      estimatedBytes: 0,
    });
  });

  console.log("render cache tests passed");
} finally {
  delete globalThis.window;
  fs.rmSync(tempPath, { force: true });
}

function installChatApi({ artifacts = [], writes = [], getArtifacts = null } = {}) {
  globalThis.window = {
    chatAPI: {
      getRenderCacheArtifacts: getArtifacts ?? (async (keys) => artifacts.filter((artifact) => keys.includes(artifact.key))),
      putRenderCacheArtifact: async (artifact) => {
        writes.push(artifact);
      },
    },
  };
}

function makeMessage({ id, status, content = "hello" }) {
  return {
    id,
    conversationId: "conv",
    role: "assistant",
    content,
    status,
    createdAt: 1,
    updatedAt: 1,
    tokenEstimate: 1,
    layout: {
      estimatedHeight: 72,
      hasCodeBlock: false,
      hasMarkdown: false,
      hasMath: false,
      hasImage: false,
      isLongMessage: false,
    },
  };
}

async function test(name, run) {
  try {
    await run();
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
