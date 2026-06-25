import { memo, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ChatMessage } from "../../../../contracts/types/message";
import { useMessageVirtualizer } from "../hooks/useMessageVirtualizer";
import { useScrollAnchor } from "../hooks/useScrollAnchor";
import {
  capturePrependScrollSnapshot,
  getRestoredScrollTopAfterPrepend,
  type PrependScrollSnapshot,
} from "../prependScrollAnchor";
import { useChatStore } from "../store";
import { StaticMessageItemMemo, StreamingMessageItemMemo } from "./MessageItem";
import {
  getCachedMessageHeight,
  hydrateCachedMessageHeightsFromDisk,
  isMessageRenderStable,
  setCachedMessageHeight,
} from "../renderCache";
import { LOAD_CHUNK_SIZE } from "../renderWindowPolicy";

interface MessageRowProps {
  height: number;
  message: ChatMessage;
  top: number;
  onHeightChange: (messageId: string, height: number) => void;
}

/**
 * STATIC ROW — renders when message.status === "complete" || "error"
 * Memo guard: message.id + message.content + message.status + height + top
 * - Re-renders ONLY when: this message's content/status/id/height/top changes
 * - SKIPPED when: VirtualMessageList re-renders due to OTHER rows' measuredHeights changes
 * - This is the key optimization: history rows are completely decoupled from streaming row updates
 */
function StaticMessageRow({ height, message, top, onHeightChange }: MessageRowProps) {
  const ref = useRef<HTMLDivElement>(null);
  const latestMessageRef = useRef(message);
  latestMessageRef.current = message;

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }

    const initialMessage = latestMessageRef.current;
    const shouldSettleAndDisconnect = isMessageRenderStable(initialMessage);
    let settleFrame: number | null = null;
    let settleTimer: number | null = null;
    let observer: ResizeObserver | null = null;

    const reportHeight = () => {
      const currentMessage = latestMessageRef.current;
      const measuredHeight = Math.ceil(element.getBoundingClientRect().height);
      onHeightChange(currentMessage.id, measuredHeight);

      if (isMessageRenderStable(currentMessage)) {
        setCachedMessageHeight(currentMessage, measuredHeight);
      }
    };

    const cachedHeight = getCachedMessageHeight(initialMessage);
    if (cachedHeight !== null) {
      onHeightChange(initialMessage.id, cachedHeight);
    }

    observer = new ResizeObserver(reportHeight);
    observer.observe(element);
    reportHeight();

    if (shouldSettleAndDisconnect) {
      settleFrame = window.requestAnimationFrame(reportHeight);
      settleTimer = window.setTimeout(() => {
        reportHeight();
        observer?.disconnect();
        observer = null;
      }, 1800);
    }

    return () => {
      if (settleFrame !== null) {
        window.cancelAnimationFrame(settleFrame);
      }

      if (settleTimer !== null) {
        window.clearTimeout(settleTimer);
      }

      observer?.disconnect();
    };
    // deps: message.id, message.status, onHeightChange
  }, [message.id, message.status, onHeightChange]);

  return (
    <div ref={ref} className="virtual-row" style={{ transform: `translateY(${top}px)`, minHeight: height }}>
      <StaticMessageItemMemo message={message} isStreaming={false} />
    </div>
  );
}

/**
 * STREAMING ROW — renders while message.status === "streaming"
 * Memo guard: message.id + message.status + height + top
 * - Re-renders when: streaming row height changes (ResizeObserver), status changes, or position shifts
 * - Does NOT re-render when: other rows change, measuredHeights Map updates for other IDs
 * - Content comes from overlay store (streamingMessageContentById), NOT from message.content prop
 */
function StreamingMessageRow({ height, message, top, onHeightChange }: MessageRowProps) {
  const ref = useRef<HTMLDivElement>(null);
  const latestMessageRef = useRef(message);
  latestMessageRef.current = message;

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }

    const initialMessage = latestMessageRef.current;
    // Streaming messages are NOT stable (status === "streaming")
    let observer: ResizeObserver | null = null;

    const reportHeight = () => {
      const currentMessage = latestMessageRef.current;
      const measuredHeight = Math.ceil(element.getBoundingClientRect().height);
      onHeightChange(currentMessage.id, measuredHeight);
    };

    observer = new ResizeObserver(reportHeight);
    observer.observe(element);
    reportHeight();

    return () => {
      observer?.disconnect();
    };
    // deps: message.id, message.status, onHeightChange
    // height/top changes come from parent props → parent re-renders → this re-renders with new props
  }, [message.id, message.status, onHeightChange]);

  return (
    <div ref={ref} className="virtual-row" style={{ transform: `translateY(${top}px)`, minHeight: height }}>
      <StreamingMessageItemMemo message={message} isStreaming={true} />
    </div>
  );
}

/**
 * Memoized row components with custom equality
 * 
 * StaticMessageRowMemo: 
 *   - Compares content + status + id + height + top
 *   - VirtualMessageList re-render (measuredHeights change) passes NEW props object
 *   - But if content/status/id/height/top are === previous, memo returns true → SKIP render
 * 
 * StreamingMessageRowMemo:
 *   - Compares status + id + height + top (content comes from overlay store, not props)
 *   - Re-renders when streaming row's own height/status changes
 */
const StaticMessageRowMemo = memo(StaticMessageRow, (prev, next) => {
  return (
    prev.message.id === next.message.id &&
    prev.message.content === next.message.content &&
    prev.message.status === next.message.status &&
    prev.height === next.height &&
    prev.top === next.top
  );
});

const StreamingMessageRowMemo = memo(StreamingMessageRow, (prev, next) => {
  return (
    prev.message.id === next.message.id &&
    prev.message.status === next.message.status &&
    prev.height === next.height &&
    prev.top === next.top
  );
});

function MessageRow({ height, message, top, onHeightChange }: MessageRowProps) {
  if (message.role === "assistant" && message.status === "streaming") {
    return <StreamingMessageRowMemo height={height} message={message} top={top} onHeightChange={onHeightChange} />;
  }
  return <StaticMessageRowMemo height={height} message={message} top={top} onHeightChange={onHeightChange} />;
}

export function VirtualMessageList() {
  const ref = useRef<HTMLDivElement>(null);
  const pendingPrependAnchorRef = useRef<PrependScrollSnapshot | null>(null);
  const [height, setHeight] = useState(600);
  const [measuredHeights, setMeasuredHeights] = useState(() => new Map<string, number>());
  const activeConversationId = useChatStore((state) => state.activeConversationId);
  const messages = useChatStore((state) => state.visibleMessages);
  const oldestMessageId = useChatStore((state) => state.oldestMessageId);
  const hasOlder = useChatStore((state) => state.hasOlder);
  const scrollTop = useChatStore((state) => state.scrollTop);
  const isLoadingOlder = useChatStore((state) => state.isLoadingOlder);
  const setScrollTop = useChatStore((state) => state.setScrollTop);
  const setLoadingOlder = useChatStore((state) => state.setLoadingOlder);
  const prependMessages = useChatStore((state) => state.prependMessages);
  const virtual = useMessageVirtualizer(messages, scrollTop, height, measuredHeights);
  const lastMessage = messages[messages.length - 1];
  const scrollDependency = [
    messages.length,
    lastMessage?.id ?? "",
    lastMessage?.status ?? "",
    lastMessage?.content.length ?? 0,
    virtual.totalHeight,
  ].join(":");
  const scrollAnchor = useScrollAnchor(ref, scrollDependency, activeConversationId);

  useLayoutEffect(() => {
    const snapshot = pendingPrependAnchorRef.current;
    const element = ref.current;
    if (!snapshot || !element) {
      return;
    }

    pendingPrependAnchorRef.current = null;
    const nextScrollTop = getRestoredScrollTopAfterPrepend(snapshot, element.scrollHeight);
    element.scrollTop = nextScrollTop;
    setScrollTop(nextScrollTop);
  }, [messages.length, setScrollTop, virtual.totalHeight]);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    const observer = new ResizeObserver(() => setHeight(element.clientHeight));
    observer.observe(element);
    setHeight(element.clientHeight);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;
    const visibleIds = new Set(messages.map((message) => message.id));

    setMeasuredHeights((current) => {
      let changed = false;
      const next = new Map(current);

      for (const message of messages) {
        if (!next.has(message.id)) {
          const cachedHeight = getCachedMessageHeight(message);
          if (cachedHeight !== null) {
            next.set(message.id, cachedHeight);
            changed = true;
          }
        }
      }

      for (const messageId of current.keys()) {
        if (!visibleIds.has(messageId)) {
          next.delete(messageId);
          changed = true;
        }
      }

      return changed ? next : current;
    });

    const stableMessages = messages.filter(isMessageRenderStable);
    if (stableMessages.length > 0) {
      void hydrateCachedMessageHeightsFromDisk(stableMessages).then((hydratedHeights) => {
        if (cancelled || hydratedHeights.size === 0) {
          return;
        }

        setMeasuredHeights((current) => {
          let changed = false;
          const next = new Map(current);

          for (const [messageId, cachedHeight] of hydratedHeights) {
            if (visibleIds.has(messageId) && next.get(messageId) !== cachedHeight) {
              next.set(messageId, cachedHeight);
              changed = true;
            }
          }

          return changed ? next : current;
        });
      });
    }

    return () => {
      cancelled = true;
    };
  }, [messages]);

  const handleRowHeightChange = useCallback((messageId: string, measuredHeight: number) => {
    setMeasuredHeights((current) => {
      if (current.get(messageId) === measuredHeight) {
        return current;
      }

      const next = new Map(current);
      next.set(messageId, measuredHeight);
      return next;
    });
  }, []);

  const handleScroll = () => {
    const element = ref.current;
    if (!element) {
      return;
    }
    setScrollTop(element.scrollTop);
    scrollAnchor.onScrollAnchor();

    if (activeConversationId && oldestMessageId && hasOlder && element.scrollTop < 96 && !isLoadingOlder) {
      setLoadingOlder(true);
      window.chatAPI
        .loadMessagesBefore(activeConversationId, oldestMessageId, LOAD_CHUNK_SIZE)
        .then((older) => {
          const elementAfterLoad = ref.current;
          if (older.length > 0 && elementAfterLoad) {
            pendingPrependAnchorRef.current = capturePrependScrollSnapshot(elementAfterLoad);
          }
          prependMessages(older, { hasOlder: older.length >= LOAD_CHUNK_SIZE });
        })
        .finally(() => setLoadingOlder(false));
    }
  };

  if (!activeConversationId) {
    return <div className="message-list empty-state">Create a conversation to start</div>;
  }

  return (
    <div
      ref={ref}
      className="message-list"
      tabIndex={0}
      onScroll={handleScroll}
      onWheel={scrollAnchor.onWheelIntent}
      onTouchStart={scrollAnchor.onTouchIntent}
      onKeyDown={scrollAnchor.onKeyIntent}
    >
      <div className="virtual-spacer" style={{ height: virtual.totalHeight }}>
        {virtual.items.map((item) => (
          <MessageRow
            key={item.message.id}
            height={item.height}
            message={item.message}
            top={item.top}
            onHeightChange={handleRowHeightChange}
          />
        ))}
      </div>
    </div>
  );
}






