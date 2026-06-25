import type { ConversationSummary } from "../../../contracts/types/conversation";
import type { ChatMessage } from "../../../contracts/types/message";
import type { NativeCoreService } from "../native_core/NativeCoreService";

export class ConversationService {
  private cachedConversations: ConversationSummary[] | null = null;
  private recentMessageCache = new Map<string, ChatMessage[]>();
  private servedConversationCache = false;
  private servedRecentCache = new Set<string>();
  private bootstrapPromise: Promise<void> | null = null;

  constructor(private readonly nativeCore: NativeCoreService) {}

  bootstrap(): Promise<void> {
    if (!this.bootstrapPromise) {
      this.bootstrapPromise = this.bootstrapNow();
    }
    return this.bootstrapPromise;
  }

  async createConversation(title?: string): Promise<ConversationSummary> {
    const conversation = await this.nativeCore.request<ConversationSummary>("conversation.create", { title });
    this.cachedConversations = [conversation, ...(this.cachedConversations ?? [])];
    this.recentMessageCache.set(conversation.id, []);
    return conversation;
  }

  async renameConversation(conversationId: string, title: string): Promise<ConversationSummary> {
    const conversation = await this.nativeCore.request<ConversationSummary>("conversation.rename", { conversationId, title });
    this.cachedConversations =
      this.cachedConversations?.map((item) => (item.id === conversation.id ? conversation : item)) ?? null;
    return conversation;
  }

  async deleteConversation(conversationId: string): Promise<void> {
    await this.nativeCore.request("conversation.delete", { conversationId });
    this.cachedConversations =
      this.cachedConversations?.filter((conversation) => conversation.id !== conversationId) ?? null;
    this.recentMessageCache.delete(conversationId);
    this.servedRecentCache.delete(conversationId);
  }

  async listConversations(): Promise<ConversationSummary[]> {
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

  async loadRecentMessages(conversationId: string, limit = 50): Promise<ChatMessage[]> {
    const cached = this.recentMessageCache.get(conversationId);
    if (cached && limit === 50 && !this.servedRecentCache.has(conversationId)) {
      this.servedRecentCache.add(conversationId);
      void this.refreshRecentCache(conversationId);
      return cached;
    }

    const messages = await this.nativeCore.request<ChatMessage[]>("message.loadRecent", { conversationId, limit });
    if (limit === 50) {
      this.recentMessageCache.set(conversationId, messages);
    }
    return messages;
  }

  loadMessagesBefore(conversationId: string, beforeMessageId: string, limit = 50): Promise<ChatMessage[]> {
    return this.nativeCore.request<ChatMessage[]>("message.loadBefore", { conversationId, beforeMessageId, limit });
  }

  private async bootstrapNow(): Promise<void> {
    await this.nativeCore.warmup();
    const conversations = await this.refreshConversationCache();
    const first = conversations[0];
    if (first) {
      await this.refreshRecentCache(first.id);
    }
  }

  private async refreshConversationCache(): Promise<ConversationSummary[]> {
    const conversations = await this.nativeCore.request<ConversationSummary[]>("conversation.list");
    this.cachedConversations = conversations;
    return conversations;
  }

  private async refreshRecentCache(conversationId: string): Promise<ChatMessage[]> {
    const messages = await this.nativeCore.request<ChatMessage[]>("message.loadRecent", { conversationId, limit: 50 });
    this.recentMessageCache.set(conversationId, messages);
    return messages;
  }
}
