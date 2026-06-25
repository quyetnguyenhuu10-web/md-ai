import { ipcMain } from "electron";
import { CONVERSATION_CHANNELS } from "../../../contracts/ipc-contracts/conversation.contract";
import type { ConversationService } from "./ConversationService";

export function registerConversationIpc(conversationService: ConversationService): void {
  ipcMain.handle(CONVERSATION_CHANNELS.createConversation, (_event, title?: string) =>
    conversationService.createConversation(title),
  );
  ipcMain.handle(CONVERSATION_CHANNELS.listConversations, () => conversationService.listConversations());
  ipcMain.handle(CONVERSATION_CHANNELS.renameConversation, (_event, conversationId: string, title: string) =>
    conversationService.renameConversation(conversationId, title),
  );
  ipcMain.handle(CONVERSATION_CHANNELS.deleteConversation, (_event, conversationId: string) =>
    conversationService.deleteConversation(conversationId),
  );
  ipcMain.handle(CONVERSATION_CHANNELS.loadRecentMessages, (_event, conversationId: string, limit?: number) =>
    conversationService.loadRecentMessages(conversationId, limit),
  );
  ipcMain.handle(
    CONVERSATION_CHANNELS.loadMessagesBefore,
    (_event, conversationId: string, beforeMessageId: string, limit?: number) =>
      conversationService.loadMessagesBefore(conversationId, beforeMessageId, limit),
  );
}
