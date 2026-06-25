import type { AssistantChunkEvent, AssistantDoneEvent, SendMessageResult } from "../types/message";

export const CHAT_CHANNELS = {
  sendMessage: "chat:sendMessage",
  assistantChunk: "chat:assistantChunk",
  assistantDone: "chat:assistantDone",
  assistantError: "chat:assistantError",
} as const;

export interface ChatApiContract {
  sendMessage(conversationId: string, content: string): Promise<SendMessageResult>;
  onAssistantChunk(callback: (event: AssistantChunkEvent) => void): () => void;
  onAssistantDone(callback: (event: AssistantDoneEvent) => void): () => void;
  onAssistantError(callback: (message: string) => void): () => void;
}
