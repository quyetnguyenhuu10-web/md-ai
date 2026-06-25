import { create } from "zustand";
import type { ConversationView } from "./types";
import { useChatStore } from "../chat/store";
import { WINDOW_TARGET_MESSAGES } from "../chat/renderWindowPolicy";

interface ConversationsState {
  conversations: ConversationView[];
  searchQuery: string;
  isLoading: boolean;
  initialize: () => Promise<void>;
  createConversation: () => Promise<void>;
  renameConversation: (conversationId: string, title: string) => Promise<void>;
  selectConversation: (conversationId: string) => Promise<void>;
  deleteConversation: (conversationId: string) => Promise<void>;
  setSearchQuery: (query: string) => void;
  refresh: () => Promise<void>;
}

export const useConversationsStore = create<ConversationsState>((set, get) => ({
  conversations: [],
  searchQuery: "",
  isLoading: false,
  initialize: async () => {
    await get().refresh();
    const first = get().conversations[0];
    if (first) {
      await get().selectConversation(first.id);
    } else {
      await get().createConversation();
    }
  },
  refresh: async () => {
    set({ isLoading: true });
    try {
      const conversations = await window.chatAPI.listConversations();
      set({ conversations });
    } finally {
      set({ isLoading: false });
    }
  },
  createConversation: async () => {
    const conversation = await window.chatAPI.createConversation();
    set((state) => ({ conversations: [conversation, ...state.conversations] }));
    await get().selectConversation(conversation.id);
  },
  renameConversation: async (conversationId: string, title: string) => {
    const previousConversations = get().conversations;
    const optimisticUpdatedAt = Date.now();

    set((state) => ({
      conversations: state.conversations.map((item) =>
        item.id === conversationId ? { ...item, title, updatedAt: optimisticUpdatedAt } : item,
      ),
    }));

    const conversation = await window.chatAPI.renameConversation(conversationId, title).catch((error: unknown) => {
      set({ conversations: previousConversations });
      throw error;
    });

    set((state) => ({
      conversations: state.conversations.map((item) => (item.id === conversation.id ? conversation : item)),
    }));
  },
  selectConversation: async (conversationId: string) => {
    useChatStore.getState().setActiveConversation(conversationId);
    const messages = await window.chatAPI.loadRecentMessages(conversationId, WINDOW_TARGET_MESSAGES);
    useChatStore.getState().replaceVisibleMessages(messages, {
      hasOlder: messages.length >= WINDOW_TARGET_MESSAGES,
      hasNewer: false,
    });
  },
  deleteConversation: async (conversationId: string) => {
    await window.chatAPI.deleteConversation(conversationId);
    const nextConversations = get().conversations.filter((conversation) => conversation.id !== conversationId);
    set({ conversations: nextConversations });

    const chatState = useChatStore.getState();
    if (chatState.activeConversationId !== conversationId) {
      return;
    }

    const fallbackConversation = nextConversations[0];
    if (fallbackConversation) {
      await get().selectConversation(fallbackConversation.id);
      return;
    }

    chatState.clearActiveConversation();
  },
  setSearchQuery: (query: string) => set({ searchQuery: query }),
}));
