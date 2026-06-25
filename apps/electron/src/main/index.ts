import { app, BrowserWindow, Menu } from "electron";
import { createMainWindow } from "./app/createMainWindow";
import { registerChatIpc } from "./modules/chat/chat.ipc";
import { ChatService } from "./modules/chat/ChatService";
import { ModelService } from "./modules/chat/ModelService";
import { registerConversationIpc } from "./modules/conversations/conversation.ipc";
import { ConversationService } from "./modules/conversations/ConversationService";
import { registerNativeIpc } from "./modules/native_core/native.ipc";
import { NativeCoreService } from "./modules/native_core/NativeCoreService";
import { StorageService } from "./modules/storage/StorageService";
import { RenderCacheDiskService } from "./modules/utilities/RenderCacheDiskService";
import { registerUtilitiesIpc } from "./modules/utilities/utilities.ipc";
import { UtilitiesService } from "./modules/utilities/UtilitiesService";
import { registerWindowIpc } from "./modules/window/window.ipc";

const nativeCore = new NativeCoreService();
const storageService = new StorageService();
const renderCacheDiskService = new RenderCacheDiskService();
const modelService = new ModelService();
const chatService = new ChatService(nativeCore, modelService);
const conversationService = new ConversationService(nativeCore);
const utilitiesService = new UtilitiesService(nativeCore, renderCacheDiskService);

registerChatIpc(chatService);
registerConversationIpc(conversationService);
registerUtilitiesIpc(utilitiesService);
registerNativeIpc(nativeCore);
registerWindowIpc();

app.whenReady().then(() => {
  app.setName("MintDim");
  Menu.setApplicationMenu(null);
  storageService.ensureDataDir();
  nativeCore.start();
  void conversationService.bootstrap().catch((error) => {
    console.error("[startup] native bootstrap failed", error);
  });
  createMainWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  nativeCore.stop();
});
