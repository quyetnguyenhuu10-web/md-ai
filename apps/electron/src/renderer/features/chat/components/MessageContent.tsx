import { memo } from "react";
import { MarkdownRenderer } from "./MarkdownRenderer";

interface MessageContentProps {
  content: string;
  isStreaming?: boolean;
}

function MessageContentComponent({ content, isStreaming = false }: MessageContentProps) {
  return <MarkdownRenderer content={content} isStreaming={isStreaming} />;
}

export const MessageContent = memo(MessageContentComponent);
