import { useEffect, useState } from "react";
import type { ContextPreview as ContextPreviewData } from "../../../../contracts/types/tool";
import { useChatStore } from "../../chat/store";
import { ToolPanel } from "./ToolPanel";

export function ContextPreview() {
  const activeConversationId = useChatStore((state) => state.activeConversationId);
  const lastUserContent = useChatStore(
    (state) => [...state.visibleMessages].reverse().find((message) => message.role === "user")?.content ?? "",
  );
  const [preview, setPreview] = useState<ContextPreviewData | null>(null);

  useEffect(() => {
    if (!activeConversationId) {
      setPreview(null);
      return;
    }
    const timeout = window.setTimeout(() => {
      void window.chatAPI.getContextPreview(activeConversationId, lastUserContent).then(setPreview);
    }, 120);
    return () => window.clearTimeout(timeout);
  }, [activeConversationId, lastUserContent]);

  return (
    <ToolPanel title="Context Preview">
      {!preview ? (
        <p className="muted">No active context</p>
      ) : (
        <div className="tool-stack">
          <div className="metric-row">
            <span>Tokens</span>
            <strong>
              {preview.tokenEstimate}/{preview.maxTokens}
            </strong>
          </div>
          <h3>System</h3>
          <p>{preview.systemPrompt}</p>
          <h3>Recent</h3>
          {preview.recentMessages.map((message) => (
            <p key={message.id} className="context-line">
              <strong>{message.role}</strong> {message.content}
            </p>
          ))}
          <h3>Relevant</h3>
          {preview.relevantMessages.length === 0 ? (
            <p className="muted">No older matches selected</p>
          ) : (
            preview.relevantMessages.map((message) => (
              <p key={message.id} className="context-line">
                <strong>{message.role}</strong> {message.content}
              </p>
            ))
          )}
        </div>
      )}
    </ToolPanel>
  );
}
