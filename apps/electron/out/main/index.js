import { BrowserWindow, ipcMain, app, Menu } from "electron";
import path from "node:path";
import { spawn } from "node:child_process";
import fs from "node:fs";
import crypto from "node:crypto";
import __cjs_mod__ from "node:module";
const __filename = import.meta.filename;
const __dirname = import.meta.dirname;
const require2 = __cjs_mod__.createRequire(import.meta.url);
function createMainWindow() {
  const win = new BrowserWindow({
    width: 1360,
    height: 860,
    minWidth: 980,
    minHeight: 640,
    title: "HighPerf Chat UI",
    backgroundColor: "#101214",
    frame: false,
    titleBarStyle: "hidden",
    webPreferences: {
      preload: path.join(__dirname, "../preload/index.mjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });
  if (process.env.ELECTRON_RENDERER_URL) {
    void win.loadURL(process.env.ELECTRON_RENDERER_URL);
  } else {
    void win.loadFile(path.join(__dirname, "../renderer/index.html"));
  }
  return win;
}
const CHAT_CHANNELS = {
  sendMessage: "chat:sendMessage",
  assistantChunk: "chat:assistantChunk",
  assistantDone: "chat:assistantDone",
  assistantError: "chat:assistantError"
};
function registerChatIpc(chatService2) {
  ipcMain.handle(
    CHAT_CHANNELS.sendMessage,
    (event, conversationId, content) => chatService2.sendMessage(conversationId, content, event.sender)
  );
}
const STREAM_PERSIST_INTERVAL_MS = 250;
const STREAM_PERSIST_CHAR_DELTA = 512;
class ChatService {
  constructor(nativeCore2, modelService2) {
    this.nativeCore = nativeCore2;
    this.modelService = modelService2;
  }
  async sendMessage(conversationId, content, webContents) {
    const userMessage = await this.nativeCore.request("message.append", {
      conversationId,
      role: "user",
      content,
      status: "complete"
    });
    const assistantMessage = await this.nativeCore.request("message.append", {
      conversationId,
      role: "assistant",
      content: "",
      status: "streaming"
    });
    void this.streamAssistant(conversationId, assistantMessage.id, content, webContents);
    return { userMessage, assistantMessage };
  }
  async streamAssistant(conversationId, messageId, userContent, webContents) {
    let fullContent = "";
    let lastQueuedPersistAt = 0;
    let lastQueuedPersistLength = 0;
    let persistFailure = null;
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
      persistChain = persistChain.then(() => this.nativeCore.request("message.update", { messageId, content })).then(() => void 0).catch((error) => {
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
      const message = await this.nativeCore.request("message.finalize", { messageId });
      webContents.send(CHAT_CHANNELS.assistantDone, { conversationId, messageId, message });
    } catch (error) {
      await flushPersist(true).catch(() => void 0);
      const message = error instanceof Error ? error.message : "Unknown streaming error";
      webContents.send(CHAT_CHANNELS.assistantError, message);
    }
  }
}
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
class ModelService {
  async *streamResponse(userMessage) {
    const text = [
      `# Mock Markdown + LaTeX stress response`,
      ``,
      `Ban vua gui: **${userMessage}**`,
      ``,
      `> Muc tieu cua doan mock nay la tao nhieu dang Markdown, KaTeX, bang, list va code block de kiem tra renderer.`,
      ``,
      `## 1. Checklist Markdown`,
      ``,
      `- **Bold text** va *italic text* trong cung mot cau.`,
      `- Inline code: \`const score = alpha * beta;\`.`,
      `- Inline math bang dollar: $E = mc^2$ va $\\nabla \\cdot \\vec{E} = \\frac{\\rho}{\\varepsilon_0}$.`,
      `- Inline math bang marker: \\( a^2 + b^2 = c^2 \\).`,
      `- Link mau: [GitHub](https://github.com).`,
      ``,
      `## 2. Bang so lieu`,
      ``,
      `| Metric | Value | Formula | Note |`,
      `| --- | ---: | --- | --- |`,
      `| perplexity | 12.42 | $e^{H(p,q)}$ | lower is better |`,
      `| accuracy | 98.1% | $\\frac{TP+TN}{N}$ | synthetic |`,
      `| latency | 34ms | $p95 < 50ms$ | renderer budget |`,
      `| tokens | 4096 | $O(n^2)$ attention | stress case |`,
      ``,
      `## 3. Display math bang $$`,
      ``,
      `$$`,
      `\\mathcal{L}(\\theta) = -\\sum_{t=1}^{T} \\log p_\\theta(y_t \\mid y_{<t}, x) + \\lambda \\lVert \\theta \\rVert_2^2`,
      `$$`,
      ``,
      `Mot he phuong trinh co canh hang:`,
      ``,
      `\\[`,
      `\\begin{aligned}`,
      `x_{t+1} &= A x_t + B u_t \\\\`,
      `y_t &= C x_t + \\epsilon_t \\\\`,
      `J &= \\sum_{t=0}^{T} \\left(x_t^T Q x_t + u_t^T R u_t\\right)`,
      `\\end{aligned}`,
      `\\]`,
      ``,
      `Ma tran block:`,
      ``,
      `$$`,
      `K = \\begin{bmatrix}`,
      `K_{xx} & K_{xz} \\\\`,
      `K_{zx} & K_{zz}`,
      `\\end{bmatrix}, \\qquad`,
      `K^{-1} = \\frac{1}{\\det(K)} \\operatorname{adj}(K)`,
      `$$`,
      ``,
      `## 4. Code block TypeScript`,
      ``,
      "```ts",
      `type RenderSample = {`,
      `  id: string;`,
      `  markdown: boolean;`,
      `  latex: "inline" | "display";`,
      `  metrics: Record<string, number>;`,
      `};`,
      ``,
      `export function summarize(sample: RenderSample): string {`,
      `  const total = Object.values(sample.metrics).reduce((sum, value) => sum + value, 0);`,
      `  return \`\${sample.id}: \${total.toFixed(2)}\`;`,
      `}`,
      "```",
      ``,
      `## 5. Code block Python`,
      ``,
      "```python",
      `import math`,
      ``,
      `def softmax(xs):`,
      `    scale = max(xs)`,
      `    exps = [math.exp(x - scale) for x in xs]`,
      `    total = sum(exps)`,
      `    return [x / total for x in exps]`,
      ``,
      `print(softmax([1.0, 2.0, 3.0]))`,
      "```",
      ``,
      `## 6. JSON config`,
      ``,
      "```json",
      `{`,
      `  "renderer": "react-markdown",`,
      `  "math": ["$", "\\\\(", "\\\\[", "$$"],`,
      `  "codeBlock": {`,
      `    "engine": "shiki+monaco",`,
      `    "copyButton": true`,
      `  }`,
      `}`,
      "```",
      ``,
      `## 7. Nested list`,
      ``,
      `1. Parser`,
      `   - tach fenced code truoc`,
      `   - sau do xu ly math display`,
      `2. Renderer`,
      `   - code block dung Shiki khi complete`,
      `   - Monaco readonly khi streaming`,
      `3. Scroll`,
      `   - chi auto khi dang o day`,
      `   - user scroll len thi dung auto`,
      ``,
      `Ket luan: neu UI on dinh, bang, math display, inline math va nhieu code block se render khong bi vo layout.`
    ].join("\n");
    for (const char of Array.from(text)) {
      await delay(2);
      yield char;
    }
  }
}
const CONVERSATION_CHANNELS = {
  createConversation: "conversation:create",
  listConversations: "conversation:list",
  renameConversation: "conversation:rename",
  deleteConversation: "conversation:delete",
  loadRecentMessages: "conversation:loadRecentMessages",
  loadMessagesBefore: "conversation:loadMessagesBefore"
};
function registerConversationIpc(conversationService2) {
  ipcMain.handle(
    CONVERSATION_CHANNELS.createConversation,
    (_event, title) => conversationService2.createConversation(title)
  );
  ipcMain.handle(CONVERSATION_CHANNELS.listConversations, () => conversationService2.listConversations());
  ipcMain.handle(
    CONVERSATION_CHANNELS.renameConversation,
    (_event, conversationId, title) => conversationService2.renameConversation(conversationId, title)
  );
  ipcMain.handle(
    CONVERSATION_CHANNELS.deleteConversation,
    (_event, conversationId) => conversationService2.deleteConversation(conversationId)
  );
  ipcMain.handle(
    CONVERSATION_CHANNELS.loadRecentMessages,
    (_event, conversationId, limit) => conversationService2.loadRecentMessages(conversationId, limit)
  );
  ipcMain.handle(
    CONVERSATION_CHANNELS.loadMessagesBefore,
    (_event, conversationId, beforeMessageId, limit) => conversationService2.loadMessagesBefore(conversationId, beforeMessageId, limit)
  );
}
class ConversationService {
  constructor(nativeCore2) {
    this.nativeCore = nativeCore2;
  }
  cachedConversations = null;
  recentMessageCache = /* @__PURE__ */ new Map();
  servedConversationCache = false;
  servedRecentCache = /* @__PURE__ */ new Set();
  bootstrapPromise = null;
  bootstrap() {
    if (!this.bootstrapPromise) {
      this.bootstrapPromise = this.bootstrapNow();
    }
    return this.bootstrapPromise;
  }
  async createConversation(title) {
    const conversation = await this.nativeCore.request("conversation.create", { title });
    this.cachedConversations = [conversation, ...this.cachedConversations ?? []];
    this.recentMessageCache.set(conversation.id, []);
    return conversation;
  }
  async renameConversation(conversationId, title) {
    const conversation = await this.nativeCore.request("conversation.rename", { conversationId, title });
    this.cachedConversations = this.cachedConversations?.map((item) => item.id === conversation.id ? conversation : item) ?? null;
    return conversation;
  }
  async deleteConversation(conversationId) {
    await this.nativeCore.request("conversation.delete", { conversationId });
    this.cachedConversations = this.cachedConversations?.filter((conversation) => conversation.id !== conversationId) ?? null;
    this.recentMessageCache.delete(conversationId);
    this.servedRecentCache.delete(conversationId);
  }
  async listConversations() {
    if (this.cachedConversations && !this.servedConversationCache) {
      this.servedConversationCache = true;
      void this.refreshConversationCache();
      return this.cachedConversations;
    }
    if (this.bootstrapPromise) {
      await this.bootstrapPromise;
      return this.cachedConversations ?? [];
    }
    return this.refreshConversationCache();
  }
  async loadRecentMessages(conversationId, limit = 50) {
    const cached = this.recentMessageCache.get(conversationId);
    if (cached && limit === 50 && !this.servedRecentCache.has(conversationId)) {
      this.servedRecentCache.add(conversationId);
      void this.refreshRecentCache(conversationId);
      return cached;
    }
    const messages = await this.nativeCore.request("message.loadRecent", { conversationId, limit });
    if (limit === 50) {
      this.recentMessageCache.set(conversationId, messages);
    }
    return messages;
  }
  loadMessagesBefore(conversationId, beforeMessageId, limit = 50) {
    return this.nativeCore.request("message.loadBefore", { conversationId, beforeMessageId, limit });
  }
  async bootstrapNow() {
    await this.nativeCore.warmup();
    const conversations = await this.refreshConversationCache();
    const first = conversations[0];
    if (first) {
      await this.refreshRecentCache(first.id);
    }
  }
  async refreshConversationCache() {
    const conversations = await this.nativeCore.request("conversation.list");
    this.cachedConversations = conversations;
    return conversations;
  }
  async refreshRecentCache(conversationId) {
    const messages = await this.nativeCore.request("message.loadRecent", { conversationId, limit: 50 });
    this.recentMessageCache.set(conversationId, messages);
    return messages;
  }
}
const NATIVE_CHANNELS = {
  ping: "native:ping",
  stats: "native:stats"
};
function registerNativeIpc(nativeCore2) {
  ipcMain.handle(NATIVE_CHANNELS.ping, () => nativeCore2.request("ping"));
  ipcMain.handle(NATIVE_CHANNELS.stats, () => nativeCore2.request("stats.get"));
}
function getProjectRoot() {
  if (app.isPackaged) {
    return path.dirname(process.resourcesPath);
  }
  return process.cwd();
}
function getDataDir() {
  return app.isPackaged ? path.join(app.getPath("userData"), "data") : path.join(getProjectRoot(), "data", "dev");
}
function getNativeSidecarPath() {
  const executable = process.platform === "win32" ? "chat_core_sidecar.exe" : "chat_core_sidecar";
  const root = getProjectRoot();
  const candidates = [
    path.join(root, "native", "build", "sidecar", "Release", executable),
    path.join(root, "native", "build", "sidecar", executable),
    path.join(root, "native", "build", "Release", executable),
    path.join(root, "native", "build", executable)
  ];
  return candidates[0];
}
class NativeCoreService {
  child = null;
  pending = /* @__PURE__ */ new Map();
  buffer = "";
  requestSeq = 0;
  timeoutMs = 1e4;
  start() {
    this.ensureStarted();
  }
  async warmup() {
    await this.request("ping");
  }
  async request(method, params = {}) {
    this.ensureStarted();
    const child = this.child;
    if (!child) {
      throw new Error("Native sidecar is not available");
    }
    const id = `req_${Date.now()}_${++this.requestSeq}`;
    const payload = JSON.stringify({ id, method, params });
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Native request timed out: ${method}`));
      }, this.timeoutMs);
      this.pending.set(id, {
        resolve,
        reject,
        timeout
      });
      child.stdin.write(`${payload}
`, (error) => {
        if (error) {
          clearTimeout(timeout);
          this.pending.delete(id);
          reject(error);
        }
      });
    });
  }
  stop() {
    if (this.child) {
      this.child.kill();
      this.child = null;
    }
  }
  ensureStarted() {
    if (this.child) {
      return;
    }
    const sidecarPath = getNativeSidecarPath();
    const dataDir = getDataDir();
    const conversationStoreDir = path.join(dataDir, "conversations");
    const legacyDatabasePath = path.join(dataDir, "chat.db");
    fs.mkdirSync(dataDir, { recursive: true });
    fs.mkdirSync(conversationStoreDir, { recursive: true });
    if (!fs.existsSync(sidecarPath)) {
      throw new Error(`Native sidecar not found at ${sidecarPath}. Run npm run build:native first.`);
    }
    this.child = spawn(sidecarPath, [conversationStoreDir, legacyDatabasePath], {
      cwd: path.dirname(sidecarPath),
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true
    });
    this.child.stdout.on("data", (chunk) => {
      this.handleStdout(chunk.toString("utf8"));
    });
    this.child.stderr.on("data", (chunk) => {
      console.error(`[native-core] ${chunk.toString("utf8").trim()}`);
    });
    this.child.on("exit", (code, signal) => {
      const pending = [...this.pending.values()];
      this.pending.clear();
      for (const request of pending) {
        clearTimeout(request.timeout);
        request.reject(new Error(`Native sidecar exited code=${code ?? "null"} signal=${signal ?? "null"}`));
      }
      this.child = null;
      this.buffer = "";
    });
  }
  handleStdout(text) {
    this.buffer += text;
    let newline = this.buffer.indexOf("\n");
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (line.length > 0) {
        this.handleLine(line);
      }
      newline = this.buffer.indexOf("\n");
    }
  }
  handleLine(line) {
    let response;
    try {
      response = JSON.parse(line);
    } catch {
      console.error(`[native-core] invalid JSON line: ${line}`);
      return;
    }
    const pending = this.pending.get(response.id);
    if (!pending) {
      return;
    }
    clearTimeout(pending.timeout);
    this.pending.delete(response.id);
    if ("error" in response) {
      pending.reject(new Error(`${response.error.code}: ${response.error.message}`));
      return;
    }
    pending.resolve(response.result);
  }
}
class StorageService {
  ensureDataDir() {
    const dir = getDataDir();
    fs.mkdirSync(dir, { recursive: true });
    return dir;
  }
}
const DISK_CACHE_VERSION = "disk-render-cache-v1";
const SOFT_LIMIT_BYTES = 256 * 1024 * 1024;
const STRONG_LIMIT_BYTES = 512 * 1024 * 1024;
const DANGER_LIMIT_BYTES = 1024 * 1024 * 1024;
const MAX_KEYS_PER_READ = 500;
const MAX_KEY_LENGTH = 4096;
const MAX_STRING_FIELD_LENGTH = 512;
const MAX_ESTIMATED_BYTES = 64 * 1024 * 1024;
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function clampString(value, fallback, maxLength = MAX_STRING_FIELD_LENGTH) {
  if (typeof value !== "string") {
    return fallback;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return fallback;
  }
  return trimmed.slice(0, maxLength);
}
function normalizeEstimatedBytes(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return 0;
  }
  return Math.min(Math.ceil(value), MAX_ESTIMATED_BYTES);
}
function warningLevel(bytes) {
  if (bytes >= DANGER_LIMIT_BYTES) {
    return "danger";
  }
  if (bytes >= STRONG_LIMIT_BYTES) {
    return "strong";
  }
  if (bytes >= SOFT_LIMIT_BYTES) {
    return "soft";
  }
  return "ok";
}
class RenderCacheDiskService {
  constructor(dataDir = getDataDir()) {
    this.dataDir = dataDir;
  }
  cachedStats = null;
  cachedStatsUntil = 0;
  getArtifacts(keys) {
    const cleanKeys = [...new Set(Array.isArray(keys) ? keys : [])].filter((key) => typeof key === "string" && key.length > 0 && key.length <= MAX_KEY_LENGTH).slice(0, MAX_KEYS_PER_READ);
    if (cleanKeys.length === 0) {
      return [];
    }
    this.ensureCacheDir();
    const artifacts = [];
    for (const key of cleanKeys) {
      const artifact = this.readArtifact(key);
      if (artifact) {
        artifacts.push(artifact);
      }
    }
    return artifacts;
  }
  putArtifact(artifact) {
    const normalized = this.normalizeArtifact(artifact);
    this.ensureCacheDir();
    const artifactPath = this.artifactPath(normalized.key);
    const tempPath = `${artifactPath}.tmp-${process.pid}-${Date.now()}`;
    fs.writeFileSync(tempPath, JSON.stringify({ version: DISK_CACHE_VERSION, artifact: normalized }), "utf8");
    fs.renameSync(tempPath, artifactPath);
    this.cachedStats = null;
    this.cachedStatsUntil = 0;
  }
  getStats() {
    const now = Date.now();
    if (this.cachedStats && now < this.cachedStatsUntil) {
      return this.cachedStats;
    }
    this.ensureCacheDir();
    let artifactCount = 0;
    let diskEstimatedBytes = 0;
    for (const entry of fs.readdirSync(this.artifactDir(), { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.endsWith(".json")) {
        continue;
      }
      artifactCount += 1;
      diskEstimatedBytes += fs.statSync(path.join(this.artifactDir(), entry.name)).size;
    }
    const stats = {
      artifactCount,
      diskEstimatedBytes,
      softLimitBytes: SOFT_LIMIT_BYTES,
      strongLimitBytes: STRONG_LIMIT_BYTES,
      dangerLimitBytes: DANGER_LIMIT_BYTES,
      warningLevel: warningLevel(diskEstimatedBytes)
    };
    this.cachedStats = stats;
    this.cachedStatsUntil = now + 5e3;
    return stats;
  }
  clear() {
    const artifactDir = this.artifactDir();
    this.assertSafeArtifactDir(artifactDir);
    if (fs.existsSync(artifactDir)) {
      const artifactDirStat = fs.lstatSync(artifactDir);
      if (artifactDirStat.isSymbolicLink()) {
        throw new Error("Refusing to clear render cache artifact symlink");
      }
      fs.rmSync(artifactDir, { recursive: true, force: true });
    }
    this.cachedStats = null;
    this.cachedStatsUntil = 0;
    return this.getStats();
  }
  ensureCacheDir() {
    fs.mkdirSync(this.artifactDir(), { recursive: true });
  }
  cacheRootDir() {
    return path.join(this.dataDir, "render-cache");
  }
  artifactDir() {
    return path.join(this.cacheRootDir(), "artifacts");
  }
  assertSafeArtifactDir(artifactDir) {
    const cacheRoot = path.resolve(this.cacheRootDir());
    const target = path.resolve(artifactDir);
    if (path.basename(cacheRoot) !== "render-cache" || path.basename(target) !== "artifacts" || path.dirname(target) !== cacheRoot) {
      throw new Error(`Refusing to clear unsafe render cache path: ${target}`);
    }
  }
  artifactPath(key) {
    const digest = crypto.createHash("sha256").update(key).digest("hex");
    return path.join(this.artifactDir(), `${digest}.json`);
  }
  readArtifact(key) {
    const artifactPath = this.artifactPath(key);
    if (!fs.existsSync(artifactPath)) {
      return null;
    }
    try {
      const parsed = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
      if (!isRecord(parsed) || parsed.version !== DISK_CACHE_VERSION || !isRecord(parsed.artifact)) {
        return null;
      }
      const artifact = this.normalizeArtifact(parsed.artifact);
      return artifact.key === key ? artifact : null;
    } catch {
      return null;
    }
  }
  normalizeArtifact(artifact) {
    if (!isRecord(artifact)) {
      throw new Error("Render cache artifact must be an object");
    }
    const key = clampString(artifact.key, "", MAX_KEY_LENGTH);
    if (!key) {
      throw new Error("Render cache artifact key is required");
    }
    if (artifact.kind !== "codeHighlightHtml" && artifact.kind !== "messageHeight") {
      throw new Error(`Unsupported render cache artifact kind: ${String(artifact.kind)}`);
    }
    if (!isRecord(artifact.payload)) {
      throw new Error("Render cache artifact payload must be an object");
    }
    const normalized = {
      key,
      kind: artifact.kind,
      rendererVersion: clampString(artifact.rendererVersion, "unknown"),
      contentHash: clampString(artifact.contentHash, "unknown", 1024),
      payload: artifact.payload,
      estimatedBytes: normalizeEstimatedBytes(artifact.estimatedBytes),
      updatedAt: typeof artifact.updatedAt === "number" && Number.isFinite(artifact.updatedAt) ? artifact.updatedAt : Date.now()
    };
    if (typeof artifact.conversationId === "string" && artifact.conversationId.trim()) {
      normalized.conversationId = artifact.conversationId.trim().slice(0, MAX_STRING_FIELD_LENGTH);
    }
    if (typeof artifact.messageId === "string" && artifact.messageId.trim()) {
      normalized.messageId = artifact.messageId.trim().slice(0, MAX_STRING_FIELD_LENGTH);
    }
    return normalized;
  }
}
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
function registerUtilitiesIpc(utilitiesService2) {
  ipcMain.handle(UTILITIES_CHANNELS.searchMessages, (_event, query) => utilitiesService2.searchMessages(query));
  ipcMain.handle(
    UTILITIES_CHANNELS.getContextPreview,
    (_event, conversationId, currentInput) => utilitiesService2.getContextPreview(conversationId, currentInput)
  );
  ipcMain.handle(
    UTILITIES_CHANNELS.getTokenBudget,
    (_event, conversationId, currentInput) => utilitiesService2.getTokenBudget(conversationId, currentInput)
  );
  ipcMain.handle(UTILITIES_CHANNELS.getRenderStats, () => utilitiesService2.getRenderStats());
  ipcMain.handle(
    UTILITIES_CHANNELS.getRenderCacheArtifacts,
    (_event, keys) => utilitiesService2.getRenderCacheArtifacts(keys)
  );
  ipcMain.handle(
    UTILITIES_CHANNELS.putRenderCacheArtifact,
    (_event, artifact) => utilitiesService2.putRenderCacheArtifact(artifact)
  );
  ipcMain.handle(UTILITIES_CHANNELS.getRenderCacheDiskStats, () => utilitiesService2.getRenderCacheDiskStats());
  ipcMain.handle(UTILITIES_CHANNELS.clearRenderCache, () => utilitiesService2.clearRenderCache());
}
class UtilitiesService {
  constructor(nativeCore2, renderCacheDisk) {
    this.nativeCore = nativeCore2;
    this.renderCacheDisk = renderCacheDisk;
  }
  searchMessages(query) {
    return this.nativeCore.request("message.search", { query });
  }
  getContextPreview(conversationId, currentInput) {
    return this.nativeCore.request("context.build", { conversationId, currentInput, maxTokens: 4096 });
  }
  getTokenBudget(conversationId, currentInput) {
    return this.nativeCore.request("token.estimate", { conversationId, currentInput, maxTokens: 4096 });
  }
  getRenderStats() {
    return this.nativeCore.request("stats.get");
  }
  getRenderCacheArtifacts(keys) {
    return Promise.resolve(this.renderCacheDisk.getArtifacts(keys));
  }
  putRenderCacheArtifact(artifact) {
    this.renderCacheDisk.putArtifact(artifact);
    return Promise.resolve();
  }
  getRenderCacheDiskStats() {
    return Promise.resolve(this.renderCacheDisk.getStats());
  }
  clearRenderCache() {
    return Promise.resolve(this.renderCacheDisk.clear());
  }
}
const WINDOW_CHANNELS = {
  minimize: "window:minimize",
  toggleMaximize: "window:toggleMaximize",
  close: "window:close"
};
function windowFromEvent(event) {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (!win) {
    throw new Error("Window is not available");
  }
  return win;
}
function registerWindowIpc() {
  ipcMain.handle(WINDOW_CHANNELS.minimize, (event) => {
    windowFromEvent(event).minimize();
  });
  ipcMain.handle(WINDOW_CHANNELS.toggleMaximize, (event) => {
    const win = windowFromEvent(event);
    if (win.isMaximized()) {
      win.unmaximize();
      return;
    }
    win.maximize();
  });
  ipcMain.handle(WINDOW_CHANNELS.close, (event) => {
    windowFromEvent(event).close();
  });
}
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
