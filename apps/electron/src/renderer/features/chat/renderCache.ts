import type { ChatMessage } from "../../../contracts/types/message";
import type { RenderCacheArtifact } from "../../../contracts/types/tool";

// Bump these when render semantics, Shiki theme, or layout measurement behavior changes.
export const RENDER_CACHE_VERSION = "render-cache-v1";
export const CODE_HIGHLIGHT_CACHE_VERSION = "code-highlight-v1";
export const MESSAGE_HEIGHT_CACHE_VERSION = "message-height-v1";

const DEFAULT_SHIKI_THEME = "slack-dark";
const CACHE_KEY_SEPARATOR = "::";

const MAX_CODE_HIGHLIGHT_ENTRIES = 240;
const MAX_CODE_HIGHLIGHT_BYTES = 16 * 1024 * 1024;
const MAX_CODE_HIGHLIGHT_METADATA_ENTRIES = MAX_CODE_HIGHLIGHT_ENTRIES * 4;
const CODE_HIGHLIGHT_ENTRY_OVERHEAD_BYTES = 160;

const MAX_MESSAGE_HEIGHT_ENTRIES = 5000;
const MAX_MESSAGE_HEIGHT_BYTES = 512 * 1024;
const MAX_MESSAGE_HEIGHT_PX = 1_000_000;

const MAX_PERSISTED_ARTIFACT_SIGNATURES = 4000;

interface SizedCacheEntry {
  estimatedBytes: number;
  updatedAt: number;
}

interface CodeHighlightCacheEntry extends SizedCacheEntry {
  html: string;
}

interface MessageHeightCacheEntry extends SizedCacheEntry {
  height: number;
}

interface CodeHighlightCacheMetadata {
  contentHash: string;
  language: string;
  theme: string;
}

export interface CodeHighlightCacheKeyParts {
  code: string;
  language: string;
  theme?: string;
}

export interface HotRenderCacheStats {
  renderCacheVersion: string;
  codeHighlightEntries: number;
  codeHighlightMetadataEntries: number;
  codeHighlightEstimatedBytes: number;
  messageHeightEntries: number;
  messageHeightEstimatedBytes: number;
  persistedArtifactSignatureEntries: number;
  estimatedBytes: number;
}

const codeHighlightCache = new Map<string, CodeHighlightCacheEntry>();
const codeHighlightKeyMetadata = new Map<string, CodeHighlightCacheMetadata>();
const messageHeightCache = new Map<string, MessageHeightCacheEntry>();
const persistedArtifactSignatures = new Map<string, string>();

let codeHighlightEstimatedBytes = 0;
let messageHeightEstimatedBytes = 0;

function estimateStringBytes(value: string): number {
  return value.length * 2;
}

function estimateMessageHeightEntryBytes(): number {
  return 96;
}

function estimateCodeHighlightEntryBytes(html: string): number {
  return estimateStringBytes(html) + CODE_HIGHLIGHT_ENTRY_OVERHEAD_BYTES;
}

function isCacheableCodeHighlightHtml(html: string): boolean {
  return estimateCodeHighlightEntryBytes(html) <= MAX_CODE_HIGHLIGHT_BYTES;
}

function normalizeMeasuredHeight(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0 || value > MAX_MESSAGE_HEIGHT_PX) {
    return null;
  }

  return Math.ceil(value);
}

function getChatApi() {
  if (typeof window === "undefined" || !window.chatAPI) {
    return null;
  }

  return window.chatAPI;
}

function evictOldestCodeHighlightEntries(): void {
  while (codeHighlightCache.size > MAX_CODE_HIGHLIGHT_ENTRIES || codeHighlightEstimatedBytes > MAX_CODE_HIGHLIGHT_BYTES) {
    const oldestKey = codeHighlightCache.keys().next().value as string | undefined;
    if (oldestKey === undefined) {
      break;
    }

    const oldestEntry = codeHighlightCache.get(oldestKey);
    if (oldestEntry) {
      codeHighlightEstimatedBytes -= oldestEntry.estimatedBytes;
    }

    codeHighlightCache.delete(oldestKey);
    codeHighlightKeyMetadata.delete(oldestKey);
  }
}

function evictOldestCodeHighlightMetadataEntries(): void {
  while (codeHighlightKeyMetadata.size > MAX_CODE_HIGHLIGHT_METADATA_ENTRIES) {
    const oldestKey = codeHighlightKeyMetadata.keys().next().value as string | undefined;
    if (oldestKey === undefined) {
      break;
    }

    codeHighlightKeyMetadata.delete(oldestKey);
  }
}

function evictOldestMessageHeightEntries(): void {
  while (messageHeightCache.size > MAX_MESSAGE_HEIGHT_ENTRIES || messageHeightEstimatedBytes > MAX_MESSAGE_HEIGHT_BYTES) {
    const oldestKey = messageHeightCache.keys().next().value as string | undefined;
    if (oldestKey === undefined) {
      break;
    }

    const oldestEntry = messageHeightCache.get(oldestKey);
    if (oldestEntry) {
      messageHeightEstimatedBytes -= oldestEntry.estimatedBytes;
    }

    messageHeightCache.delete(oldestKey);
  }
}

function setCodeHighlightEntry(key: string, entry: CodeHighlightCacheEntry): void {
  const existing = codeHighlightCache.get(key);
  if (existing) {
    codeHighlightEstimatedBytes -= existing.estimatedBytes;
    codeHighlightCache.delete(key);
  }

  codeHighlightCache.set(key, entry);
  codeHighlightEstimatedBytes += entry.estimatedBytes;
  evictOldestCodeHighlightEntries();
}

function setMessageHeightEntry(key: string, entry: MessageHeightCacheEntry): void {
  const existing = messageHeightCache.get(key);
  if (existing) {
    messageHeightEstimatedBytes -= existing.estimatedBytes;
    messageHeightCache.delete(key);
  }

  messageHeightCache.set(key, entry);
  messageHeightEstimatedBytes += entry.estimatedBytes;
  evictOldestMessageHeightEntries();
}

function rememberPersistedArtifactSignature(key: string, signature: string): boolean {
  if (persistedArtifactSignatures.get(key) === signature) {
    return false;
  }

  persistedArtifactSignatures.delete(key);
  persistedArtifactSignatures.set(key, signature);

  while (persistedArtifactSignatures.size > MAX_PERSISTED_ARTIFACT_SIGNATURES) {
    const oldestKey = persistedArtifactSignatures.keys().next().value as string | undefined;
    if (oldestKey === undefined) {
      break;
    }
    persistedArtifactSignatures.delete(oldestKey);
  }

  return true;
}

function persistRenderCacheArtifact(artifact: RenderCacheArtifact, signature: string): void {
  const api = getChatApi();
  if (!api || typeof api.putRenderCacheArtifact !== "function") {
    return;
  }

  if (!rememberPersistedArtifactSignature(artifact.key, signature)) {
    return;
  }

  void api.putRenderCacheArtifact(artifact).catch((error) => {
    if (persistedArtifactSignatures.get(artifact.key) === signature) {
      persistedArtifactSignatures.delete(artifact.key);
    }
    console.warn("[render-cache] failed to persist disk artifact", error);
  });
}

function readCodeHighlightHtmlPayload(artifact: RenderCacheArtifact | undefined): string | null {
  if (!artifact || artifact.kind !== "codeHighlightHtml") {
    return null;
  }

  const html = artifact.payload.html;
  if (typeof html !== "string" || !isCacheableCodeHighlightHtml(html)) {
    return null;
  }

  return html;
}

async function readDiskRenderCacheArtifacts(keys: string[]): Promise<RenderCacheArtifact[]> {
  const api = getChatApi();
  if (!api || typeof api.getRenderCacheArtifacts !== "function" || keys.length === 0) {
    return [];
  }

  try {
    return await api.getRenderCacheArtifacts(keys);
  } catch (error) {
    console.warn("[render-cache] failed to read disk artifacts", error);
    return [];
  }
}

export function makeContentHash(value: string): string {
  let hashA = 0x811c9dc5;
  let hashB = 0x9e3779b9;

  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    hashA ^= code;
    hashA = Math.imul(hashA, 16777619) >>> 0;
    hashB = Math.imul(hashB ^ code, 2246822519) >>> 0;
  }

  return `${value.length.toString(36)}:${hashA.toString(36)}:${hashB.toString(36)}`;
}

export function makeCodeHighlightCacheKey({
  code,
  language,
  theme = DEFAULT_SHIKI_THEME,
}: CodeHighlightCacheKeyParts): string {
  const contentHash = makeContentHash(code);
  const key = [RENDER_CACHE_VERSION, CODE_HIGHLIGHT_CACHE_VERSION, theme, language, contentHash].join(CACHE_KEY_SEPARATOR);

  codeHighlightKeyMetadata.set(key, {
    contentHash,
    language,
    theme,
  });
  evictOldestCodeHighlightMetadataEntries();

  return key;
}

export function getCachedCodeHighlightHtml(key: string): string | null {
  const entry = codeHighlightCache.get(key);
  if (!entry) {
    return null;
  }

  setCodeHighlightEntry(key, {
    ...entry,
    updatedAt: Date.now(),
  });

  return entry.html;
}

export async function getDiskCachedCodeHighlightHtml(key: string): Promise<string | null> {
  const artifacts = await readDiskRenderCacheArtifacts([key]);
  const artifact = artifacts.find((item) => item.key === key && item.kind === "codeHighlightHtml");
  const html = readCodeHighlightHtmlPayload(artifact);

  if (html === null) {
    return null;
  }

  setCodeHighlightEntry(key, {
    html,
    estimatedBytes: estimateCodeHighlightEntryBytes(html),
    updatedAt: Date.now(),
  });

  return html;
}

export function setCachedCodeHighlightHtml(key: string, html: string): void {
  if (!isCacheableCodeHighlightHtml(html)) {
    codeHighlightKeyMetadata.delete(key);
    return;
  }

  const estimatedBytes = estimateCodeHighlightEntryBytes(html);
  setCodeHighlightEntry(key, {
    html,
    estimatedBytes,
    updatedAt: Date.now(),
  });

  const metadata = codeHighlightKeyMetadata.get(key);
  persistRenderCacheArtifact(
    {
      key,
      kind: "codeHighlightHtml",
      rendererVersion: CODE_HIGHLIGHT_CACHE_VERSION,
      contentHash: metadata?.contentHash ?? makeContentHash(key),
      payload: {
        html,
        language: metadata?.language ?? "unknown",
        theme: metadata?.theme ?? DEFAULT_SHIKI_THEME,
      },
      estimatedBytes,
      updatedAt: Date.now(),
    },
    makeContentHash(html),
  );
}

export function isMessageRenderStable(message: ChatMessage): boolean {
  return message.status === "complete" || message.status === "error";
}

export function makeMessageHeightCacheKey(message: ChatMessage): string {
  return [
    RENDER_CACHE_VERSION,
    MESSAGE_HEIGHT_CACHE_VERSION,
    message.conversationId,
    message.id,
    makeContentHash(message.content),
  ].join(CACHE_KEY_SEPARATOR);
}

export function getCachedMessageHeight(message: ChatMessage): number | null {
  const key = makeMessageHeightCacheKey(message);
  const entry = messageHeightCache.get(key);
  if (!entry) {
    return null;
  }

  setMessageHeightEntry(key, {
    ...entry,
    updatedAt: Date.now(),
  });

  return entry.height;
}

export async function hydrateCachedMessageHeightsFromDisk(messages: ChatMessage[]): Promise<Map<string, number>> {
  const stableMessages = messages.filter(isMessageRenderStable);
  if (stableMessages.length === 0) {
    return new Map();
  }

  const keyToMessage = new Map<string, ChatMessage>();
  for (const message of stableMessages) {
    keyToMessage.set(makeMessageHeightCacheKey(message), message);
  }

  const artifacts = await readDiskRenderCacheArtifacts([...keyToMessage.keys()]);
  const hydrated = new Map<string, number>();

  for (const artifact of artifacts) {
    if (artifact.kind !== "messageHeight") {
      continue;
    }

    const message = keyToMessage.get(artifact.key);
    const measuredHeight = normalizeMeasuredHeight(artifact.payload.height);

    if (!message || measuredHeight === null) {
      continue;
    }

    setMessageHeightEntry(artifact.key, {
      height: measuredHeight,
      estimatedBytes: estimateMessageHeightEntryBytes(),
      updatedAt: Date.now(),
    });
    hydrated.set(message.id, measuredHeight);
  }

  return hydrated;
}

export function setCachedMessageHeight(message: ChatMessage, height: number): void {
  const measuredHeight = normalizeMeasuredHeight(height);
  if (!isMessageRenderStable(message) || measuredHeight === null) {
    return;
  }

  const key = makeMessageHeightCacheKey(message);
  const contentHash = makeContentHash(message.content);

  setMessageHeightEntry(key, {
    height: measuredHeight,
    estimatedBytes: estimateMessageHeightEntryBytes(),
    updatedAt: Date.now(),
  });

  persistRenderCacheArtifact(
    {
      key,
      kind: "messageHeight",
      rendererVersion: MESSAGE_HEIGHT_CACHE_VERSION,
      contentHash,
      conversationId: message.conversationId,
      messageId: message.id,
      payload: {
        height: measuredHeight,
      },
      estimatedBytes: estimateMessageHeightEntryBytes(),
      updatedAt: Date.now(),
    },
    `${measuredHeight}`,
  );
}

export function clearHotRenderCaches(): void {
  codeHighlightCache.clear();
  codeHighlightKeyMetadata.clear();
  messageHeightCache.clear();
  persistedArtifactSignatures.clear();
  codeHighlightEstimatedBytes = 0;
  messageHeightEstimatedBytes = 0;
}

export function getHotRenderCacheStats(): HotRenderCacheStats {
  return {
    renderCacheVersion: RENDER_CACHE_VERSION,
    codeHighlightEntries: codeHighlightCache.size,
    codeHighlightMetadataEntries: codeHighlightKeyMetadata.size,
    codeHighlightEstimatedBytes,
    messageHeightEntries: messageHeightCache.size,
    messageHeightEstimatedBytes,
    persistedArtifactSignatureEntries: persistedArtifactSignatures.size,
    estimatedBytes: codeHighlightEstimatedBytes + messageHeightEstimatedBytes,
  };
}
