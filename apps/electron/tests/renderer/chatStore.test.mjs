import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

const electronRoot = findElectronRoot(process.cwd());
const sourcePath = path.join(electronRoot, "src", "renderer", "features", "chat", "store.ts");
const source = fs.readFileSync(sourcePath, "utf8");
const transpiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ES2022,
    target: ts.ScriptTarget.ES2022,
    strict: true,
  },
});
const zustandPath = path.join(electronRoot, "node_modules", "zustand", "esm", "index.mjs");
const output = transpiled.outputText.replace(
  `from "zustand"`,
  `from ${JSON.stringify(pathToFileURL(zustandPath).href)}`,
);
const tempPath = path.join(os.tmpdir(), `chatStore.${process.pid}.${Date.now()}.mjs`);
fs.writeFileSync(tempPath, output);

try {
  const { useChatStore } = await import(pathToFileURL(tempPath).href);

  test("setActiveConversation resets render window state", () => {
    resetStore(useChatStore);
    useChatStore.getState().replaceVisibleMessages([makeMessage("m1"), makeMessage("m2")], {
      hasOlder: true,
      hasNewer: true,
    });
    useChatStore.getState().setLoadingOlder(true);
    useChatStore.getState().setLoadingNewer(true);
    useChatStore.getState().setScrollTop(120);

    useChatStore.getState().setActiveConversation("next");
    const state = useChatStore.getState();

    assert.equal(state.activeConversationId, "next");
    assert.deepEqual(state.visibleMessages, []);
    assert.equal(state.oldestMessageId, null);
    assert.equal(state.newestMessageId, null);
    assert.equal(state.hasOlder, false);
    assert.equal(state.hasNewer, false);
    assert.equal(state.isLoadingOlder, false);
    assert.equal(state.isLoadingNewer, false);
    assert.equal(state.scrollTop, 0);
  });

  test("replaceVisibleMessages tracks window bounds and flags", () => {
    resetStore(useChatStore);
    useChatStore.getState().replaceVisibleMessages([makeMessage("m1"), makeMessage("m2")], {
      hasOlder: true,
      hasNewer: false,
    });

    const state = useChatStore.getState();
    assert.equal(state.oldestMessageId, "m1");
    assert.equal(state.newestMessageId, "m2");
    assert.equal(state.hasOlder, true);
    assert.equal(state.hasNewer, false);
  });

  test("prependMessages dedupes and updates oldest cursor", () => {
    resetStore(useChatStore);
    useChatStore.getState().replaceVisibleMessages([makeMessage("m2"), makeMessage("m3")], { hasOlder: true });
    useChatStore.getState().prependMessages([makeMessage("m1"), makeMessage("m2")], { hasOlder: true });

    const state = useChatStore.getState();
    assert.deepEqual(
      state.visibleMessages.map((message) => message.id),
      ["m1", "m2", "m3"],
    );
    assert.equal(state.oldestMessageId, "m1");
    assert.equal(state.newestMessageId, "m3");
    assert.equal(state.hasOlder, true);
  });

  test("empty older chunk closes older pagination", () => {
    resetStore(useChatStore);
    useChatStore.getState().replaceVisibleMessages([makeMessage("m2")], { hasOlder: true });
    useChatStore.getState().prependMessages([], { hasOlder: false });

    const state = useChatStore.getState();
    assert.deepEqual(
      state.visibleMessages.map((message) => message.id),
      ["m2"],
    );
    assert.equal(state.oldestMessageId, "m2");
    assert.equal(state.hasOlder, false);
  });

  test("appendMessages dedupes and updates newest cursor", () => {
    resetStore(useChatStore);
    useChatStore.getState().replaceVisibleMessages([makeMessage("m1")]);
    useChatStore.getState().appendMessages([makeMessage("m1"), makeMessage("m2")]);

    const state = useChatStore.getState();
    assert.deepEqual(
      state.visibleMessages.map((message) => message.id),
      ["m1", "m2"],
    );
    assert.equal(state.oldestMessageId, "m1");
    assert.equal(state.newestMessageId, "m2");
  });

  test("appendToMessage writes streaming overlay without replacing visible window", () => {
    resetStore(useChatStore);
    useChatStore.getState().replaceVisibleMessages([makeMessage("m1", { content: "hello", status: "streaming" })]);
    const previousVisibleMessages = useChatStore.getState().visibleMessages;

    useChatStore.getState().appendToMessage("m1", " world");

    const state = useChatStore.getState();
    assert.equal(state.visibleMessages, previousVisibleMessages);
    assert.equal(state.visibleMessages[0].content, "hello");
    assert.equal(state.streamingMessageContentById.m1, "hello world");
  });

  test("appendToMessage buffers early stream deltas before the placeholder is visible", () => {
    resetStore(useChatStore);

    useChatStore.getState().appendToMessage("m1", "early");

    assert.equal(useChatStore.getState().streamingMessageContentById.m1, "early");
  });

  test("replaceMessage commits completed message and clears streaming overlay", () => {
    resetStore(useChatStore);
    useChatStore.getState().replaceVisibleMessages([makeMessage("m1", { content: "", status: "streaming" })]);
    useChatStore.getState().appendToMessage("m1", "draft");

    useChatStore.getState().replaceMessage(makeMessage("m1", { content: "final", status: "complete" }));

    const state = useChatStore.getState();
    assert.equal(state.visibleMessages[0].content, "final");
    assert.equal(state.visibleMessages[0].status, "complete");
    assert.equal(state.streamingMessageContentById.m1, undefined);
  });

  console.log("chat store tests passed");
} finally {
  fs.rmSync(tempPath, { force: true });
}

function makeMessage(id, overrides = {}) {
  return {
    id,
    conversationId: "conv",
    role: "assistant",
    content: overrides.content ?? id,
    status: overrides.status ?? "complete",
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

function resetStore(useChatStore) {
  useChatStore.setState({
    activeConversationId: null,
    visibleMessages: [],
    streamingMessageContentById: {},
    oldestMessageId: null,
    newestMessageId: null,
    hasOlder: false,
    hasNewer: false,
    isSending: false,
    isLoadingOlder: false,
    isLoadingNewer: false,
    scrollTop: 0,
    streamUpdateIntervalMs: 45,
  });
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
