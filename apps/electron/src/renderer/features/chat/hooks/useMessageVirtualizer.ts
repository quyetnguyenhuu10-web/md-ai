import { useMemo } from "react";
import type { ChatMessage } from "../../../../contracts/types/message";
import { getCachedMessageHeight } from "../renderCache";
import { WINDOW_OVERSCAN_MESSAGES } from "../renderWindowPolicy";

export function useMessageVirtualizer(
  messages: ChatMessage[],
  scrollTop: number,
  viewportHeight: number,
  measuredHeights: ReadonlyMap<string, number>,
) {
  return useMemo(() => {
    const offsets: number[] = [];
    let totalHeight = 0;
    const getHeight = (message: ChatMessage) =>
      measuredHeights.get(message.id) ?? getCachedMessageHeight(message) ?? Math.max(72, message.layout.estimatedHeight);
    for (const message of messages) {
      offsets.push(totalHeight);
      totalHeight += getHeight(message);
    }

    let start = 0;
    while (start < offsets.length && offsets[start] + getHeight(messages[start]) < scrollTop) {
      start += 1;
    }

    let end = start;
    while (end < offsets.length && offsets[end] < scrollTop + viewportHeight) {
      end += 1;
    }

    const from = Math.max(0, start - WINDOW_OVERSCAN_MESSAGES);
    const to = Math.min(messages.length, end + WINDOW_OVERSCAN_MESSAGES);

    return {
      totalHeight,
      items: messages.slice(from, to).map((message, index) => ({
        message,
        top: offsets[from + index],
        height: getHeight(message),
      })),
    };
  }, [messages, scrollTop, viewportHeight, measuredHeights]);
}


