import { contextBridge, ipcRenderer } from "electron";
const CHAT_CHANNELS = {
  sendMessage: "chat:sendMessage",
  assistantChunk: "chat:assistantChunk",
  assistantDone: "chat:assistantDone",
  assistantError: "chat:assistantError"
};
const CONVERSATION_CHANNELS = {
  createConversation: "conversation:create",
  listConversations: "conversation:list",
  renameConversation: "conversation:rename",
  deleteConversation: "conversation:delete",
  loadRecentMessages: "conversation:loadRecentMessages",
  loadMessagesBefore: "conversation:loadMessagesBefore"
};
const NATIVE_CHANNELS = {
  ping: "native:ping",
  stats: "native:stats"
};
const UTILITIES_CHANNELS = {
  searchMessages: "utilities:searchMessages",
  getContextPreview: "utilities:getContextPreview",
  getTokenBudget: "utilities:getTokenBudget",
  getRenderStats: "utilities:getRenderStats",
  getRenderCacheArtifacts: "utilities:getRenderCacheArtifacts",
  putRenderCacheArtifact: "utilities:putRenderCacheArtifact",
  getRenderCacheDiskStats: "utilities:getRenderCacheDiskStats",
  clearRenderCache: "utilities:clearRenderCache"
};
const WINDOW_CHANNELS = {
  minimize: "window:minimize",
  toggleMaximize: "window:toggleMaximize",
  close: "window:close"
};
const api = {
  createConversation: (title) => ipcRenderer.invoke(CONVERSATION_CHANNELS.createConversation, title),
  listConversations: () => ipcRenderer.invoke(CONVERSATION_CHANNELS.listConversations),
  renameConversation: (conversationId, title) => ipcRenderer.invoke(CONVERSATION_CHANNELS.renameConversation, conversationId, title),
  deleteConversation: (conversationId) => ipcRenderer.invoke(CONVERSATION_CHANNELS.deleteConversation, conversationId),
  loadRecentMessages: (conversationId, limit) => ipcRenderer.invoke(CONVERSATION_CHANNELS.loadRecentMessages, conversationId, limit),
  loadMessagesBefore: (conversationId, beforeMessageId, limit) => ipcRenderer.invoke(CONVERSATION_CHANNELS.loadMessagesBefore, conversationId, beforeMessageId, limit),
  sendMessage: (conversationId, content) => ipcRenderer.invoke(CHAT_CHANNELS.sendMessage, conversationId, content),
  searchMessages: (query) => ipcRenderer.invoke(UTILITIES_CHANNELS.searchMessages, query),
  getContextPreview: (conversationId, currentInput) => ipcRenderer.invoke(UTILITIES_CHANNELS.getContextPreview, conversationId, currentInput),
  getTokenBudget: (conversationId, currentInput) => ipcRenderer.invoke(UTILITIES_CHANNELS.getTokenBudget, conversationId, currentInput),
  getRenderStats: () => ipcRenderer.invoke(UTILITIES_CHANNELS.getRenderStats),
  getRenderCacheArtifacts: (keys) => ipcRenderer.invoke(UTILITIES_CHANNELS.getRenderCacheArtifacts, keys),
  putRenderCacheArtifact: (artifact) => ipcRenderer.invoke(UTILITIES_CHANNELS.putRenderCacheArtifact, artifact),
  getRenderCacheDiskStats: () => ipcRenderer.invoke(UTILITIES_CHANNELS.getRenderCacheDiskStats),
  clearRenderCache: () => ipcRenderer.invoke(UTILITIES_CHANNELS.clearRenderCache),
  ping: () => ipcRenderer.invoke(NATIVE_CHANNELS.ping),
  stats: () => ipcRenderer.invoke(NATIVE_CHANNELS.stats),
  onAssistantChunk: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on(CHAT_CHANNELS.assistantChunk, listener);
    return () => ipcRenderer.off(CHAT_CHANNELS.assistantChunk, listener);
  },
  onAssistantDone: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on(CHAT_CHANNELS.assistantDone, listener);
    return () => ipcRenderer.off(CHAT_CHANNELS.assistantDone, listener);
  },
  onAssistantError: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on(CHAT_CHANNELS.assistantError, listener);
    return () => ipcRenderer.off(CHAT_CHANNELS.assistantError, listener);
  }
};
contextBridge.exposeInMainWorld("chatAPI", api);
const windowApi = {
  minimizeWindow: () => ipcRenderer.invoke(WINDOW_CHANNELS.minimize),
  toggleMaximizeWindow: () => ipcRenderer.invoke(WINDOW_CHANNELS.toggleMaximize),
  closeWindow: () => ipcRenderer.invoke(WINDOW_CHANNELS.close)
};
contextBridge.exposeInMainWorld("windowAPI", windowApi);
