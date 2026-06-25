import { ipcMain } from "electron";
import { CHAT_CHANNELS } from "../../../contracts/ipc-contracts/chat.contract";
import type { ChatService } from "./ChatService";

export function registerChatIpc(chatService: ChatService): void {
  ipcMain.handle(CHAT_CHANNELS.sendMessage, (event, conversationId: string, content: string) =>
    chatService.sendMessage(conversationId, content, event.sender),
  );
}
