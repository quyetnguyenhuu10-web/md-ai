import { Plus } from "lucide-react";
import { useConversationsStore } from "../store";

export function NewConversationButton() {
  const createConversation = useConversationsStore((state) => state.createConversation);

  return (
    <button type="button" className="icon-button" onClick={() => void createConversation()} title="New Chat">
      <Plus size={18} aria-hidden="true" />
    </button>
  );
}
