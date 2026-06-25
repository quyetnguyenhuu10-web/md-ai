import { useMemo } from "react";
import { useConversationsStore } from "../store";

export function useConversations() {
  const conversations = useConversationsStore((state) => state.conversations);
  const query = useConversationsStore((state) => state.searchQuery.trim().toLowerCase());

  return useMemo(() => {
    if (!query) {
      return conversations;
    }
    return conversations.filter((conversation) => conversation.title.toLowerCase().includes(query));
  }, [conversations, query]);
}
