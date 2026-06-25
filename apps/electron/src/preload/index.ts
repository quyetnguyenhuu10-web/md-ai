import { contextBridge, ipcRenderer } from "electron";
import { CHAT_CHANNELS } from "../contracts/ipc-contracts/chat.contract";
import { CONVERSATION_CHANNELS } from "../contracts/ipc-contracts/conversation.contract";
import type { ChatApi } from "../contracts/ipc-contracts";
import { NATIVE_CHANNELS } from "../contracts/ipc-contracts/native.contract";
import { UTILITIES_CHANNELS } from "../contracts/ipc-contracts/utilities.contract";
import { WINDOW_CHANNELS, type WindowApiContract } from "../contracts/ipc-contracts/window.contract";
import type { AssistantChunkEvent, AssistantDoneEvent } from "../contracts/types/message";
import type { RenderCacheArtifact } from "../contracts/types/tool";

const api: ChatApi = {
  createConversation: (title?: string) => ipcRenderer.invoke(CONVERSATION_CHANNELS.createConversation, title),
  listConversations: () => ipcRenderer.invoke(CONVERSATION_CHANNELS.listConversations),
  renameConversation: (conversationId: string, title: string) =>
    ipcRenderer.invoke(CONVERSATION_CHANNELS.renameConversation, conversationId, title),
  deleteConversation: (conversationId: string) => ipcRenderer.invoke(CONVERSATION_CHANNELS.deleteConversation, conversationId),
  loadRecentMessages: (conversationId: string, limit?: number) =>
    ipcRenderer.invoke(CONVERSATION_CHANNELS.loadRecentMessages, conversationId, limit),
  loadMessagesBefore: (conversationId: string, beforeMessageId: string, limit?: number) =>
    ipcRenderer.invoke(CONVERSATION_CHANNELS.loadMessagesBefore, conversationId, beforeMessageId, limit),
  sendMessage: (conversationId: string, content: string) =>
    ipcRenderer.invoke(CHAT_CHANNELS.sendMessage, conversationId, content),
  searchMessages: (query: string) => ipcRenderer.invoke(UTILITIES_CHANNELS.searchMessages, query),
  getContextPreview: (conversationId: string, currentInput: string) =>
    ipcRenderer.invoke(UTILITIES_CHANNELS.getContextPreview, conversationId, currentInput),
  getTokenBudget: (conversationId: string, currentInput: string) =>
    ipcRenderer.invoke(UTILITIES_CHANNELS.getTokenBudget, conversationId, currentInput),
  getRenderStats: () => ipcRenderer.invoke(UTILITIES_CHANNELS.getRenderStats),
  getRenderCacheArtifacts: (keys: string[]) => ipcRenderer.invoke(UTILITIES_CHANNELS.getRenderCacheArtifacts, keys),
  putRenderCacheArtifact: (artifact: RenderCacheArtifact) => ipcRenderer.invoke(UTILITIES_CHANNELS.putRenderCacheArtifact, artifact),
  getRenderCacheDiskStats: () => ipcRenderer.invoke(UTILITIES_CHANNELS.getRenderCacheDiskStats),
  clearRenderCache: () => ipcRenderer.invoke(UTILITIES_CHANNELS.clearRenderCache),
  ping: () => ipcRenderer.invoke(NATIVE_CHANNELS.ping),
  stats: () => ipcRenderer.invoke(NATIVE_CHANNELS.stats),
  onAssistantChunk: (callback: (event: AssistantChunkEvent) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, payload: AssistantChunkEvent) => callback(payload);
    ipcRenderer.on(CHAT_CHANNELS.assistantChunk, listener);
    return () => ipcRenderer.off(CHAT_CHANNELS.assistantChunk, listener);
  },
  onAssistantDone: (callback: (event: AssistantDoneEvent) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, payload: AssistantDoneEvent) => callback(payload);
    ipcRenderer.on(CHAT_CHANNELS.assistantDone, listener);
    return () => ipcRenderer.off(CHAT_CHANNELS.assistantDone, listener);
  },
  onAssistantError: (callback: (message: string) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, payload: string) => callback(payload);
    ipcRenderer.on(CHAT_CHANNELS.assistantError, listener);
    return () => ipcRenderer.off(CHAT_CHANNELS.assistantError, listener);
  },
};

contextBridge.exposeInMainWorld("chatAPI", api);

const windowApi: WindowApiContract = {
  minimizeWindow: () => ipcRenderer.invoke(WINDOW_CHANNELS.minimize),
  toggleMaximizeWindow: () => ipcRenderer.invoke(WINDOW_CHANNELS.toggleMaximize),
  closeWindow: () => ipcRenderer.invoke(WINDOW_CHANNELS.close),
};

contextBridge.exposeInMainWorld("windowAPI", windowApi);
