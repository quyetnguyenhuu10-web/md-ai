import type { WebContents } from "electron";
import { CHAT_CHANNELS } from "../../../contracts/ipc-contracts/chat.contract";
import type { ChatMessage, SendMessageResult } from "../../../contracts/types/message";
import type { NativeCoreService } from "../native_core/NativeCoreService";
import { ModelService } from "./ModelService";

const STREAM_PERSIST_INTERVAL_MS = 250;
const STREAM_PERSIST_CHAR_DELTA = 512;

export class ChatService {
  constructor(
    private readonly nativeCore: NativeCoreService,
    private readonly modelService: ModelService,
  ) {}

  async sendMessage(conversationId: string, content: string, webContents: WebContents): Promise<SendMessageResult> {
    const userMessage = await this.nativeCore.request<ChatMessage>("message.append", {
      conversationId,
      role: "user",
      content,
      status: "complete",
    });
    const assistantMessage = await this.nativeCore.request<ChatMessage>("message.append", {
      conversationId,
      role: "assistant",
      content: "",
      status: "streaming",
    });

    void this.streamAssistant(conversationId, assistantMessage.id, content, webContents);

    return { userMessage, assistantMessage };
  }

  private async streamAssistant(
    conversationId: string,
    messageId: string,
    userContent: string,
    webContents: WebContents,
  ): Promise<void> {
    let fullContent = "";
    let lastQueuedPersistAt = 0;
    let lastQueuedPersistLength = 0;
    let persistFailure: unknown = null;
    let persistChain = Promise.resolve();

    const queuePersist = (force = false) => {
      const now = Date.now();
      const charsSincePersist = fullContent.length - lastQueuedPersistLength;
      const msSincePersist = now - lastQueuedPersistAt;

      if (!force && charsSincePersist < STREAM_PERSIST_CHAR_DELTA && msSincePersist < STREAM_PERSIST_INTERVAL_MS) {
        return;
      }

      if (fullContent.length === lastQueuedPersistLength) {
        return;
      }

      const content = fullContent;
      lastQueuedPersistAt = now;
      lastQueuedPersistLength = content.length;
      persistChain = persistChain
        .then(() => this.nativeCore.request("message.update", { messageId, content }))
        .then(() => undefined)
        .catch((error: unknown) => {
          persistFailure ??= error;
        });
    };

    const flushPersist = async (force = false) => {
      queuePersist(force);
      await persistChain;
      if (persistFailure) {
        throw persistFailure;
      }
    };

    try {
      for await (const delta of this.modelService.streamResponse(userContent)) {
        fullContent += delta;
        webContents.send(CHAT_CHANNELS.assistantChunk, { conversationId, messageId, delta });
        queuePersist();
      }

      await flushPersist(true);
      const message = await this.nativeCore.request<ChatMessage>("message.finalize", { messageId });
      webContents.send(CHAT_CHANNELS.assistantDone, { conversationId, messageId, message });
    } catch (error) {
      await flushPersist(true).catch(() => undefined);
      const message = error instanceof Error ? error.message : "Unknown streaming error";
      webContents.send(CHAT_CHANNELS.assistantError, message);
    }
  }
}
