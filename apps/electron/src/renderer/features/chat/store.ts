import { create } from "zustand";
import type { ChatMessage } from "../../../contracts/types/message";

interface MessageWindowOptions {
  hasOlder?: boolean;
  hasNewer?: boolean;
}

interface ChatState {
  activeConversationId: string | null;
  visibleMessages: ChatMessage[];
  /** Overlay content for streaming messages: messageId → accumulated delta content */
  streamingMessageContentById: Record<string, string>;
  oldestMessageId: string | null;
  newestMessageId: string | null;
  hasOlder: boolean;
  hasNewer: boolean;
  isSending: boolean;
  isLoadingOlder: boolean;
  isLoadingNewer: boolean;
  scrollTop: number;
  streamUpdateIntervalMs: number;
  setActiveConversation: (conversationId: string) => void;
  clearActiveConversation: () => void;
  replaceVisibleMessages: (messages: ChatMessage[], options?: MessageWindowOptions) => void;
  prependMessages: (messages: ChatMessage[], options?: Pick<MessageWindowOptions, "hasOlder">) => void;
  appendMessages: (messages: ChatMessage[], options?: Pick<MessageWindowOptions, "hasNewer">) => void;
  /** Append delta to streaming message overlay. Triggers re-render ONLY for the StreamingMessageItem subscribing to this messageId. */
  appendToMessage: (messageId: string, delta: string) => void;
  /** Replace streaming message with final complete message. visibleMessages array ref changes → VirtualMessageList re-renders. */
  replaceMessage: (message: ChatMessage) => void;
  setSending: (isSending: boolean) => void;
  setLoadingOlder: (isLoadingOlder: boolean) => void;
  setLoadingNewer: (isLoadingNewer: boolean) => void;
  setScrollTop: (scrollTop: number) => void;
  setStreamUpdateInterval: (intervalMs: number) => void;
}

function getWindowBounds(messages: ChatMessage[]) {
  return {
    oldestMessageId: messages[0]?.id ?? null,
    newestMessageId: messages[messages.length - 1]?.id ?? null,
  };
}

function emptyWindowState() {
  return {
    visibleMessages: [],
    streamingMessageContentById: {},
    oldestMessageId: null,
    newestMessageId: null,
    hasOlder: false,
    hasNewer: false,
    isLoadingOlder: false,
    isLoadingNewer: false,
    scrollTop: 0,
  };
}

export const useChatStore = create<ChatState>((set) => ({
  activeConversationId: null,
  ...emptyWindowState(),
  isSending: false,
  streamUpdateIntervalMs: 45,
  setActiveConversation: (conversationId) => set({ activeConversationId: conversationId, ...emptyWindowState() }),
  clearActiveConversation: () => set({ activeConversationId: null, ...emptyWindowState() }),
  replaceVisibleMessages: (messages, options) =>
    set({
      visibleMessages: messages,
      streamingMessageContentById: {},
      ...getWindowBounds(messages),
      hasOlder: options?.hasOlder ?? false,
      hasNewer: options?.hasNewer ?? false,
    }),
  prependMessages: (messages, options) =>
    set((state) => {
      const existingIds = new Set(state.visibleMessages.map((message) => message.id));
      const visibleMessages = [...messages.filter((message) => !existingIds.has(message.id)), ...state.visibleMessages];
      return {
        visibleMessages,
        ...getWindowBounds(visibleMessages),
        hasOlder: options?.hasOlder ?? (messages.length > 0 ? state.hasOlder : false),
      };
    }),
  appendMessages: (messages, options) =>
    set((state) => {
      const existingIds = new Set(state.visibleMessages.map((message) => message.id));
      const visibleMessages = [...state.visibleMessages, ...messages.filter((message) => !existingIds.has(message.id))];
      return {
        visibleMessages,
        ...getWindowBounds(visibleMessages),
        hasNewer: options?.hasNewer ?? state.hasNewer,
      };
    }),
  /**
   * STREAMING OVERLAY UPDATE
   * - Creates new streamingMessageContentById object (immutable)
   * - Only the StreamingMessageItem with matching messageId subscribes to this key
   * - Zustand shallow equality → other components NOT re-rendered
   */
  appendToMessage: (messageId, delta) =>
    set((state) => {
      const currentContent =
        state.streamingMessageContentById[messageId] ?? state.visibleMessages.find((message) => message.id === messageId)?.content ?? "";

      return {
        streamingMessageContentById: {
          ...state.streamingMessageContentById,
          [messageId]: currentContent + delta,
        },
      };
    }),
  /**
   * STREAMING → COMPLETE TRANSITION
   * - visibleMessages array ref changes → VirtualMessageList re-renders
   * - streamingMessageContentById[messageId] deleted
   * - VirtualMessageList maps new messages array → row key changes (streaming → static) → new row mounts
   */
  replaceMessage: (message) =>
    set((state) => {
      const streamingMessageContentById = { ...state.streamingMessageContentById };
      delete streamingMessageContentById[message.id];

      return {
        visibleMessages: state.visibleMessages.map((item) => (item.id === message.id ? message : item)),
        streamingMessageContentById,
      };
    }),
  setSending: (isSending) => set({ isSending }),
  setLoadingOlder: (isLoadingOlder) => set({ isLoadingOlder }),
  setLoadingNewer: (isLoadingNewer) => set({ isLoadingNewer }),
  setScrollTop: (scrollTop) => set({ scrollTop }),
  setStreamUpdateInterval: (intervalMs) => set({ streamUpdateIntervalMs: intervalMs }),
}));
