import { memo } from "react";
import type { ChatMessage } from "../../../../contracts/types/message";
import { useChatStore } from "../store";
import { MessageContent } from "./MessageContent";

interface MessageItemProps {
  message: ChatMessage;
  isStreaming: boolean;
}

function MessageArticle({ message, content, isStreaming }: MessageItemProps & { content: string }) {
  return (
    <article className={`message-item ${message.role}`}>
      <div className="message-meta">
        <strong>{message.role}</strong>
        <span>{message.tokenEstimate} tok</span>
      </div>
      <div className="message-surface">
        <MessageContent content={content} isStreaming={isStreaming} />
      </div>
    </article>
  );
}

/**
 * STREAMING MESSAGE ITEM
 * - Subscribes to streamingMessageContentById[message.id] via granular selector
 * - Re-renders ONLY when this specific message's overlay content changes
 * - Does NOT re-render when other messages stream or when VirtualMessageList re-renders
 */
function StreamingMessageItem({ message }: MessageItemProps) {
  // Granular selector: returns string | undefined for this messageId only
  const streamingContent = useChatStore((state) => state.streamingMessageContentById[message.id]);
  const content = streamingContent ?? message.content;
  return <MessageArticle message={message} content={content} isStreaming />;
}

/**
 * STATIC MESSAGE ITEM (history: complete/error)
 * - Does NOT subscribe to streamingMessageContentById store
 * - Receives content purely from message.content prop
 * - Memo comparison on message.id + content + status + tokenEstimate
 */
function StaticMessageItem({ message }: MessageItemProps) {
  return <MessageArticle message={message} content={message.content} isStreaming={false} />;
}

export const StreamingMessageItemMemo = memo(StreamingMessageItem);
export const StaticMessageItemMemo = memo(StaticMessageItem);
