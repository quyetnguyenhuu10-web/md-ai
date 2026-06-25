import { useEffect, useMemo, useRef, useState } from "react";
import { MoreHorizontal, Pencil } from "lucide-react";
import { Composer } from "../../features/chat/components/Composer";
import { VirtualMessageList } from "../../features/chat/components/VirtualMessageList";
import { useChatStore } from "../../features/chat/store";
import { useConversationsStore } from "../../features/conversations/store";

export function ChatPanel() {
  const activeConversationId = useChatStore((state) => state.activeConversationId);
  const conversations = useConversationsStore((state) => state.conversations);
  const renameConversation = useConversationsStore((state) => state.renameConversation);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isRenaming, setIsRenaming] = useState(false);
  const [renameDraft, setRenameDraft] = useState("");
  const menuRef = useRef<HTMLDivElement>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);
  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeConversationId),
    [activeConversationId, conversations],
  );

  useEffect(() => {
    if (!isRenaming) {
      return;
    }

    renameInputRef.current?.focus();
    renameInputRef.current?.select();
  }, [isRenaming]);

  useEffect(() => {
    if (!isMenuOpen) {
      return;
    }

    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target;
      if (target instanceof Node && menuRef.current?.contains(target)) {
        return;
      }

      setIsMenuOpen(false);
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isMenuOpen]);

  const startRename = () => {
    if (!activeConversation) {
      return;
    }

    setIsMenuOpen(false);
    setRenameDraft(activeConversation.title);
    setIsRenaming(true);
  };

  const commitRename = () => {
    if (!activeConversation || !isRenaming) {
      return;
    }

    const nextTitle = renameDraft.trim();
    setIsRenaming(false);

    if (!nextTitle || nextTitle === activeConversation.title) {
      return;
    }

    void renameConversation(activeConversation.id, nextTitle).catch((error: unknown) => {
      console.error("Failed to rename conversation", error);
      window.alert(error instanceof Error ? error.message : "Failed to rename conversation");
    });
  };

  return (
    <section className="chat-panel">
      <header className="chat-header">
        <div ref={menuRef} className="chat-headerTitleGroup">
          {isRenaming && activeConversation ? (
            <input
              ref={renameInputRef}
              className="chat-headerRenameInput"
              value={renameDraft}
              aria-label="Rename chat"
              onChange={(event) => setRenameDraft(event.target.value)}
              onBlur={commitRename}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  commitRename();
                }

                if (event.key === "Escape") {
                  event.preventDefault();
                  setIsRenaming(false);
                  setRenameDraft(activeConversation.title);
                }
              }}
            />
          ) : (
            <h2>{activeConversation?.title ?? "No conversation selected"}</h2>
          )}
          {activeConversation ? (
            <button
              type="button"
              className="chat-headerMore"
              aria-label={`Open menu for ${activeConversation.title}`}
              aria-expanded={isMenuOpen}
              onClick={() => setIsMenuOpen((open) => !open)}
            >
              <MoreHorizontal size={16} aria-hidden="true" />
            </button>
          ) : null}

          {isMenuOpen && activeConversation ? (
            <div className="chat-headerMenu" role="menu">
              <button type="button" className="chat-headerMenuItem" role="menuitem" onClick={startRename}>
                <Pencil size={15} aria-hidden="true" />
                <span>Rename chat</span>
              </button>
            </div>
          ) : null}
        </div>
      </header>
      <VirtualMessageList />
      <Composer />
    </section>
  );
}
