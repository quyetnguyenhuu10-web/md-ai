import { Plus, Search } from "lucide-react";
import { ConversationList } from "../../features/conversations/components/ConversationList";
import { useConversationsStore } from "../../features/conversations/store";

export function LeftPanel() {
  const searchQuery = useConversationsStore((state) => state.searchQuery);
  const setSearchQuery = useConversationsStore((state) => state.setSearchQuery);
  const createConversation = useConversationsStore((state) => state.createConversation);

  return (
    <aside className="left-panel panel">
      <nav className="sidebar-quickActions" aria-label="Conversation shortcuts">
        <button type="button" className="sidebar-action" onClick={() => void createConversation()}>
          <Plus size={16} aria-hidden="true" />
          <span>New chat</span>
        </button>

        <label className="sidebar-action sidebar-search">
          <Search size={16} aria-hidden="true" />
          <input
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search chats"
            aria-label="Search chats"
          />
        </label>
      </nav>

      <section className="sidebar-recents" aria-label="Recent conversations">
        <div className="sidebar-sectionHeader">
          <span>Recents</span>
        </div>
        <ConversationList />
      </section>
    </aside>
  );
}
