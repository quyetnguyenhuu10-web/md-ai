import { useEffect, useRef } from "react";
import { useChatStore } from "../store";

/**
 * RE-RENDER MECHANISM: Streaming delta flow
 * 
 * 1. IPC chunk arrives → onAssistantChunk callback
 *    - Accumulates delta in pendingChunks ref (NO render)
 *    - Schedules single RAF flush per frame
 * 
 * 2. RAF flush → appendToMessage(messageId, delta)
 *    - Immutable update: streamingMessageContentById[messageId] = newContent
 *    - Zustand selector granularity: only StreamingMessageItem subscribing to that key re-renders
 * 
 * 3. onAssistantDone → replaceMessage + setSending(false)
 *    - visibleMessages array ref changes → VirtualMessageList re-renders
 *    - streamingMessageContentById[messageId] deleted → StreamingMessageItem unmounts
 *    - Row switches from StreamingMessageRow → StaticMessageRow (new key) → mount once
 */
export function useChatStream() {
  const pendingChunks = useRef(new Map<string, string>());
  const raf = useRef<number | null>(null);
  const lastFlush = useRef(performance.now());

  useEffect(() => {
    const flush = () => {
      raf.current = null;
      const now = performance.now();
      useChatStore.getState().setStreamUpdateInterval(now - lastFlush.current);
      lastFlush.current = now;

      // Batch all pending deltas → single store update per messageId
      for (const [messageId, delta] of pendingChunks.current.entries()) {
        useChatStore.getState().appendToMessage(messageId, delta);
      }
      pendingChunks.current.clear();
    };

    const scheduleFlush = () => {
      if (raf.current !== null) {
        return; // Already scheduled for this frame
      }
      raf.current = window.requestAnimationFrame(flush);
    };

    // IPC: streaming token chunk arrives from native sidecar
    const offChunk = window.chatAPI.onAssistantChunk((event) => {
      const current = pendingChunks.current.get(event.messageId) ?? "";
      pendingChunks.current.set(event.messageId, current + event.delta);
      scheduleFlush(); // Coalesce chunks within same RAF
    });

    // IPC: generation complete → flush remaining, replace with final message
    const offDone = window.chatAPI.onAssistantDone((event) => {
      if (raf.current !== null) {
        const scheduledFrame = raf.current;
        raf.current = null;
        window.cancelAnimationFrame(scheduledFrame);
        flush(); // Flush any remaining deltas before replace
      }
      pendingChunks.current.delete(event.messageId);
      // visibleMessages ref changes → VirtualMessageList re-renders
      // streamingMessageContentById[messageId] deleted → selector returns undefined
      useChatStore.getState().replaceMessage(event.message);
      useChatStore.getState().setSending(false);
    });

    const offError = window.chatAPI.onAssistantError((message) => {
      console.error(message);
      useChatStore.getState().setSending(false);
    });

    return () => {
      offChunk();
      offDone();
      offError();
      if (raf.current !== null) {
        window.cancelAnimationFrame(raf.current);
      }
    };
  }, []);
}
