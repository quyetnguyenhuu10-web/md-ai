import { useChatStore } from "../../chat/store";
import { useConversations } from "../hooks/useConversations";
import { useConversationsStore } from "../store";
import { ConversationItem } from "./ConversationItem";

export function ConversationList() {
  const conversations = useConversations();
  const isLoading = useConversationsStore((state) => state.isLoading);
  const activeConversationId = useChatStore((state) => state.activeConversationId);
  const selectConversation = useConversationsStore((state) => state.selectConversation);
  const renameConversation = useConversationsStore((state) => state.renameConversation);
  const deleteConversation = useConversationsStore((state) => state.deleteConversation);

  if (isLoading && conversations.length === 0) {
    return <div className="empty-state">Loading conversations</div>;
  }

  return (
    <div className="conversation-list">
      {conversations.map((conversation) => (
        <ConversationItem
          key={conversation.id}
          conversation={conversation}
          isActive={conversation.id === activeConversationId}
          onSelect={() => void selectConversation(conversation.id)}
          onRename={(title) => renameConversation(conversation.id, title)}
          onDelete={() => void deleteConversation(conversation.id)}
        />
      ))}
    </div>
  );
}
