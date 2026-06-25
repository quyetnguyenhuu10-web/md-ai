import type {
  ContextPreview,
  RenderCacheArtifact,
  RenderCacheDiskStats,
  RenderStats,
  SearchResult,
  TokenBudget,
} from "../../../contracts/types/tool";
import type { NativeCoreService } from "../native_core/NativeCoreService";
import type { RenderCacheDiskService } from "./RenderCacheDiskService";

export class UtilitiesService {
  constructor(
    private readonly nativeCore: NativeCoreService,
    private readonly renderCacheDisk: RenderCacheDiskService,
  ) {}

  searchMessages(query: string): Promise<SearchResult[]> {
    return this.nativeCore.request<SearchResult[]>("message.search", { query });
  }

  getContextPreview(conversationId: string, currentInput: string): Promise<ContextPreview> {
    return this.nativeCore.request<ContextPreview>("context.build", { conversationId, currentInput, maxTokens: 4096 });
  }

  getTokenBudget(conversationId: string, currentInput: string): Promise<TokenBudget> {
    return this.nativeCore.request<TokenBudget>("token.estimate", { conversationId, currentInput, maxTokens: 4096 });
  }

  getRenderStats(): Promise<RenderStats> {
    return this.nativeCore.request<RenderStats>("stats.get");
  }

  getRenderCacheArtifacts(keys: string[]): Promise<RenderCacheArtifact[]> {
    return Promise.resolve(this.renderCacheDisk.getArtifacts(keys));
  }

  putRenderCacheArtifact(artifact: RenderCacheArtifact): Promise<void> {
    this.renderCacheDisk.putArtifact(artifact);
    return Promise.resolve();
  }

  getRenderCacheDiskStats(): Promise<RenderCacheDiskStats> {
    return Promise.resolve(this.renderCacheDisk.getStats());
  }

  clearRenderCache(): Promise<RenderCacheDiskStats> {
    return Promise.resolve(this.renderCacheDisk.clear());
  }
}
