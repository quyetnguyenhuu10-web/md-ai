import type { ChatApiContract } from "./chat.contract";
import type { ConversationApiContract } from "./conversation.contract";
import type { NativeApiContract } from "./native.contract";
import type { UtilitiesApiContract } from "./utilities.contract";
import type { WindowApiContract } from "./window.contract";

export type ChatApi = ChatApiContract & ConversationApiContract & UtilitiesApiContract & NativeApiContract;

declare global {
  interface Window {
    chatAPI: ChatApi;
    windowAPI: WindowApiContract;
  }
}
