import { SendHorizontal } from "lucide-react";
import { useState } from "react";
import { useChatStore } from "../store";
import { useConversationsStore } from "../../conversations/store";

export function Composer() {
  const [content, setContent] = useState("");
  const activeConversationId = useChatStore((state) => state.activeConversationId);
  const isSending = useChatStore((state) => state.isSending);
  const setSending = useChatStore((state) => state.setSending);
  const appendMessages = useChatStore((state) => state.appendMessages);
  const refreshConversations = useConversationsStore((state) => state.refresh);

  const send = async () => {
    const trimmed = content.trim();
    if (!activeConversationId || !trimmed || isSending) {
      return;
    }

    setContent("");
    setSending(true);

    try {
      const result = await window.chatAPI.sendMessage(activeConversationId, trimmed);
      appendMessages([result.userMessage, result.assistantMessage]);
      void refreshConversations();
    } catch (error) {
      setSending(false);
      console.error(error);
    }
  };

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault();
        void send();
      }}
    >
      <textarea
        value={content}
        placeholder={activeConversationId ? "Ask for follow-up changes" : "Create a chat first"}
        rows={1}
        disabled={!activeConversationId}
        onChange={(event) => setContent(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            void send();
          }
        }}
      />
      <button type="submit" className="send-button" disabled={!content.trim() || !activeConversationId || isSending} title="Send">
        <SendHorizontal size={18} strokeWidth={2.4} aria-hidden="true" />
      </button>
    </form>
  );
}
