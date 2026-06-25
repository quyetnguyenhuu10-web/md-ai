export type MessageRole = "system" | "user" | "assistant";
export type MessageStatus = "draft" | "streaming" | "complete" | "error";

export interface LayoutMetadata {
  estimatedHeight: number;
  hasCodeBlock: boolean;
  hasMarkdown: boolean;
  hasMath: boolean;
  hasImage: boolean;
  isLongMessage: boolean;
}

export interface ChatMessage {
  id: string;
  conversationId: string;
  role: MessageRole;
  content: string;
  status: MessageStatus;
  createdAt: number;
  updatedAt: number;
  tokenEstimate: number;
  layout: LayoutMetadata;
}

export interface SendMessageResult {
  userMessage: ChatMessage;
  assistantMessage: ChatMessage;
}

export interface AssistantChunkEvent {
  conversationId: string;
  messageId: string;
  delta: string;
}

export interface AssistantDoneEvent {
  conversationId: string;
  messageId: string;
  message: ChatMessage;
}
