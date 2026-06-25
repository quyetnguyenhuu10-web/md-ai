import type { ConversationSummary } from "../types/conversation";
import type { ChatMessage } from "../types/message";

export const CONVERSATION_CHANNELS = {
  createConversation: "conversation:create",
  listConversations: "conversation:list",
  renameConversation: "conversation:rename",
  deleteConversation: "conversation:delete",
  loadRecentMessages: "conversation:loadRecentMessages",
  loadMessagesBefore: "conversation:loadMessagesBefore",
} as const;

export interface ConversationApiContract {
  createConversation(title?: string): Promise<ConversationSummary>;
  listConversations(): Promise<ConversationSummary[]>;
  renameConversation(conversationId: string, title: string): Promise<ConversationSummary>;
  deleteConversation(conversationId: string): Promise<void>;
  loadRecentMessages(conversationId: string, limit?: number): Promise<ChatMessage[]>;
  loadMessagesBefore(conversationId: string, beforeMessageId: string, limit?: number): Promise<ChatMessage[]>;
}
