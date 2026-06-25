import type {
  ContextPreview,
  RenderCacheArtifact,
  RenderCacheDiskStats,
  RenderStats,
  SearchResult,
  TokenBudget,
} from "../types/tool";

export const UTILITIES_CHANNELS = {
  searchMessages: "utilities:searchMessages",
  getContextPreview: "utilities:getContextPreview",
  getTokenBudget: "utilities:getTokenBudget",
  getRenderStats: "utilities:getRenderStats",
  getRenderCacheArtifacts: "utilities:getRenderCacheArtifacts",
  putRenderCacheArtifact: "utilities:putRenderCacheArtifact",
  getRenderCacheDiskStats: "utilities:getRenderCacheDiskStats",
  clearRenderCache: "utilities:clearRenderCache",
} as const;

export interface UtilitiesApiContract {
  searchMessages(query: string): Promise<SearchResult[]>;
  getContextPreview(conversationId: string, currentInput: string): Promise<ContextPreview>;
  getTokenBudget(conversationId: string, currentInput: string): Promise<TokenBudget>;
  getRenderStats(): Promise<RenderStats>;
  getRenderCacheArtifacts(keys: string[]): Promise<RenderCacheArtifact[]>;
  putRenderCacheArtifact(artifact: RenderCacheArtifact): Promise<void>;
  getRenderCacheDiskStats(): Promise<RenderCacheDiskStats>;
  clearRenderCache(): Promise<RenderCacheDiskStats>;
}
