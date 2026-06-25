import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

const electronRoot = findElectronRoot(process.cwd());
const sourcePath = path.join(electronRoot, "src", "main", "modules", "chat", "ChatService.ts");
const source = fs
  .readFileSync(sourcePath, "utf8")
  .replace('import type { WebContents } from "electron";\n', "")
  .replace(
    'import { CHAT_CHANNELS } from "../../../contracts/ipc-contracts/chat.contract";',
    'const CHAT_CHANNELS = { assistantChunk: "assistant:chunk", assistantDone: "assistant:done", assistantError: "assistant:error" };',
  )
  .replace('import type { ChatMessage, SendMessageResult } from "../../../contracts/types/message";\n', "")
  .replace('import type { NativeCoreService } from "../native_core/NativeCoreService";\n', "")
  .replace('import { ModelService } from "./ModelService";\n', "");
const transpiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ES2022,
    target: ts.ScriptTarget.ES2022,
    strict: true,
  },
});
const tempPath = path.join(os.tmpdir(), `ChatService.${process.pid}.${Date.now()}.mjs`);
fs.writeFileSync(tempPath, transpiled.outputText);

try {
  const { ChatService } = await import(pathToFileURL(tempPath).href);

  await test("streams chunks immediately while batching native persistence", async () => {
    const streamedText = "x".repeat(600);
    const updates = [];
    const events = [];
    let doneResolve;
    const done = new Promise((resolve) => {
      doneResolve = resolve;
    });

    const nativeCore = {
      async request(method, params = {}) {
        if (method === "message.append") {
          return {
            id: params.role === "assistant" ? "assistant-message" : "user-message",
            conversationId: params.conversationId,
            role: params.role,
            content: params.content,
            status: params.status,
            createdAt: 1,
            updatedAt: 1,
            tokenEstimate: 0,
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

        if (method === "message.update") {
          updates.push(params.content);
          await delay(1);
          return { ok: true };
        }

        if (method === "message.finalize") {
          return {
            id: params.messageId,
            conversationId: "conv",
            role: "assistant",
            content: updates.at(-1) ?? "",
            status: "complete",
            createdAt: 1,
            updatedAt: 2,
            tokenEstimate: 0,
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

        throw new Error(`Unexpected native method: ${method}`);
      },
    };

    const modelService = {
      async *streamResponse() {
        for (const char of streamedText) {
          yield char;
        }
      },
    };

    const webContents = {
      send(channel, payload) {
        events.push({ channel, payload });
        if (channel === "assistant:done") {
          doneResolve();
        }
      },
    };

    const service = new ChatService(nativeCore, modelService);
    await service.sendMessage("conv", "hello", webContents);
    await done;

    const chunks = events.filter((event) => event.channel === "assistant:chunk");
    assert.equal(chunks.length, streamedText.length);
    assert.equal(updates.length < streamedText.length / 10, true);
    assert.equal(updates.at(-1), streamedText);
    assert.equal(events.at(-1).channel, "assistant:done");
  });

  console.log("chat service streaming tests passed");
} finally {
  fs.rmSync(tempPath, { force: true });
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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
