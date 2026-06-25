import type { ChatMessage } from "./message";

export interface ContextPreview {
  systemPrompt: string;
  summary: string;
  memoryFacts: string[];
  recentMessages: ChatMessage[];
  relevantMessages: ChatMessage[];
  currentUserMessage: string;
  tokenEstimate: number;
  maxTokens: number;
}

export interface TokenBudget {
  inputTokens: number;
  historyTokens: number;
  totalTokens: number;
  maxTokens: number;
  remainingTokens: number;
}

export interface SearchResult {
  message: ChatMessage;
  rank: number;
}

export type RenderCacheArtifactKind = "codeHighlightHtml" | "messageHeight";

export interface RenderCacheArtifact {
  key: string;
  kind: RenderCacheArtifactKind;
  rendererVersion: string;
  contentHash: string;
  conversationId?: string;
  messageId?: string;
  payload: Record<string, unknown>;
  estimatedBytes: number;
  updatedAt?: number;
}

export interface RenderCacheDiskStats {
  artifactCount: number;
  diskEstimatedBytes: number;
  softLimitBytes: number;
  strongLimitBytes: number;
  dangerLimitBytes: number;
  warningLevel: "ok" | "soft" | "strong" | "danger";
}

export interface RenderStats {
  visibleMessageCount: number;
  cachedMessageCount: number;
  estimatedLayoutCount: number;
  streamingFps: number;
  streamUpdateIntervalMs: number;
}
