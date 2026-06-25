import { useEffect, useState } from "react";
import type { TokenBudget } from "../../../../contracts/types/tool";
import { useChatStore } from "../../chat/store";
import { ToolPanel } from "./ToolPanel";

export function TokenBudgetPanel() {
  const activeConversationId = useChatStore((state) => state.activeConversationId);
  const lastUserContent = useChatStore(
    (state) => [...state.visibleMessages].reverse().find((message) => message.role === "user")?.content ?? "",
  );
  const [budget, setBudget] = useState<TokenBudget | null>(null);

  useEffect(() => {
    if (!activeConversationId) {
      setBudget(null);
      return;
    }
    const timeout = window.setTimeout(() => {
      void window.chatAPI.getTokenBudget(activeConversationId, lastUserContent).then(setBudget);
    }, 120);
    return () => window.clearTimeout(timeout);
  }, [activeConversationId, lastUserContent]);

  return (
    <ToolPanel title="Token Budget">
      {budget ? (
        <div className="tool-stack">
          <div className="budget-bar">
            <span style={{ width: `${Math.min(100, (budget.totalTokens / budget.maxTokens) * 100)}%` }} />
          </div>
          <div className="metric-row">
            <span>Input</span>
            <strong>{budget.inputTokens}</strong>
          </div>
          <div className="metric-row">
            <span>History</span>
            <strong>{budget.historyTokens}</strong>
          </div>
          <div className="metric-row">
            <span>Remaining</span>
            <strong>{budget.remainingTokens}</strong>
          </div>
        </div>
      ) : (
        <p className="muted">No budget available</p>
      )}
    </ToolPanel>
  );
}
