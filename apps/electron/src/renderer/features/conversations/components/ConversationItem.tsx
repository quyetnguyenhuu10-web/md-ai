import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import type { ConversationView } from "../types";

interface ConversationItemProps {
  conversation: ConversationView;
  isActive: boolean;
  onSelect: () => void;
  onRename: (title: string) => Promise<void>;
  onDelete: () => void;
}

export function ConversationItem({ conversation, isActive, onSelect, onRename, onDelete }: ConversationItemProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isRenaming, setIsRenaming] = useState(false);
  const [renameDraft, setRenameDraft] = useState("");
  const [menuPosition, setMenuPosition] = useState<{ top: number; left: number } | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const moreButtonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);

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

    const updateMenuPosition = () => {
      const button = moreButtonRef.current;
      if (!button) {
        return;
      }

      const rect = button.getBoundingClientRect();
      const menuWidth = 176;
      const margin = 8;
      setMenuPosition({
        top: Math.min(rect.bottom + margin, window.innerHeight - 88),
        left: Math.min(Math.max(margin, rect.right - menuWidth), window.innerWidth - menuWidth - margin),
      });
    };

    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target;
      if (
        target instanceof Node &&
        (rootRef.current?.contains(target) || menuRef.current?.contains(target))
      ) {
        return;
      }

      setIsMenuOpen(false);
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsMenuOpen(false);
      }
    };

    updateMenuPosition();
    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    window.addEventListener("resize", updateMenuPosition);
    window.addEventListener("scroll", updateMenuPosition, true);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("resize", updateMenuPosition);
      window.removeEventListener("scroll", updateMenuPosition, true);
    };
  }, [isMenuOpen]);

  const handleSelect = () => {
    setIsMenuOpen(false);
    onSelect();
  };

  const startRename = () => {
    setIsMenuOpen(false);
    setRenameDraft(conversation.title);
    setIsRenaming(true);
  };

  const commitRename = () => {
    if (!isRenaming) {
      return;
    }

    const nextTitle = renameDraft.trim();
    setIsRenaming(false);

    if (!nextTitle || nextTitle === conversation.title) {
      return;
    }

    void onRename(nextTitle).catch((error: unknown) => {
      console.error("Failed to rename conversation", error);
      window.alert(error instanceof Error ? error.message : "Failed to rename conversation");
    });
  };

  const handleDelete = () => {
    setIsMenuOpen(false);
    onDelete();
  };

  return (
    <div ref={rootRef} className={`conversation-item ${isActive ? "active" : ""}`}>
      {isRenaming ? (
        <div className="conversation-main conversation-mainEditing">
          <input
            ref={renameInputRef}
            className="conversation-renameInput"
            value={renameDraft}
            aria-label="Rename chat"
            onClick={(event) => event.stopPropagation()}
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
                setRenameDraft(conversation.title);
              }
            }}
          />
          <span>{new Date(conversation.updatedAt).toLocaleString()}</span>
        </div>
      ) : (
        <button type="button" className="conversation-main" onClick={handleSelect}>
          <strong>{conversation.title}</strong>
          <span>{new Date(conversation.updatedAt).toLocaleString()}</span>
        </button>
      )}

      <button
        ref={moreButtonRef}
        type="button"
        className="conversation-more"
        aria-label={`Open menu for ${conversation.title}`}
        aria-expanded={isMenuOpen}
        onClick={(event) => {
          event.stopPropagation();
          setIsMenuOpen((open) => !open);
        }}
      >
        <MoreHorizontal size={16} aria-hidden="true" />
      </button>

      {isMenuOpen && menuPosition
        ? createPortal(
            <div ref={menuRef} className="conversation-menu" style={menuPosition} role="menu">
              <button
                type="button"
                className="conversation-menuItem"
                role="menuitem"
                onClick={(event) => {
                  event.stopPropagation();
                  startRename();
                }}
              >
                <Pencil size={15} aria-hidden="true" />
                <span>Rename chat</span>
              </button>
              <button
                type="button"
                className="conversation-menuItem danger"
                role="menuitem"
                onClick={(event) => {
                  event.stopPropagation();
                  handleDelete();
                }}
              >
                <Trash2 size={15} aria-hidden="true" />
                <span>Delete</span>
              </button>
            </div>,
            document.body,
          )
        : null}
    </div>
  );
}
