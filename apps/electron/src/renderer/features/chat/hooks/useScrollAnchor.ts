import { useCallback, useLayoutEffect, useRef, type KeyboardEvent, type RefObject, type WheelEvent } from "react";

const BOTTOM_THRESHOLD_PX = 32;
const PROGRAMMATIC_SCROLL_MS = 60;
const USER_IDLE_MS = 140;
const SCROLL_UP_KEYS = new Set(["ArrowUp", "PageUp", "Home"]);

function isAtBottom(element: HTMLDivElement): boolean {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= BOTTOM_THRESHOLD_PX;
}

export function useScrollAnchor(ref: RefObject<HTMLDivElement>, dependency: string, resetKey: string | null) {
  const stickToBottom = useRef(true);
  const userInteracting = useRef(false);
  const programmaticScroll = useRef(false);
  const interactionTimer = useRef<number | null>(null);
  const programmaticTimer = useRef<number | null>(null);

  const scrollToRenderedBottom = useCallback((element: HTMLDivElement) => {
    programmaticScroll.current = true;
    element.scrollTop = element.scrollHeight;

    if (programmaticTimer.current !== null) {
      window.clearTimeout(programmaticTimer.current);
    }

    programmaticTimer.current = window.setTimeout(() => {
      programmaticScroll.current = false;
    }, PROGRAMMATIC_SCROLL_MS);
  }, []);

  useLayoutEffect(() => {
    stickToBottom.current = true;
  }, [resetKey]);

  useLayoutEffect(
    () => () => {
      if (interactionTimer.current !== null) {
        window.clearTimeout(interactionTimer.current);
      }
      if (programmaticTimer.current !== null) {
        window.clearTimeout(programmaticTimer.current);
      }
    },
    [],
  );

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }

    if (stickToBottom.current && !userInteracting.current) {
      scrollToRenderedBottom(element);
    }
  }, [dependency, ref, scrollToRenderedBottom]);

  const onScrollAnchor = useCallback(() => {
    const element = ref.current;
    if (!element) {
      return;
    }

    if (programmaticScroll.current) {
      return;
    }

    stickToBottom.current = isAtBottom(element);
  }, [ref]);

  const markUserInteracting = useCallback(() => {
    const element = ref.current;
    userInteracting.current = true;

    if (interactionTimer.current !== null) {
      window.clearTimeout(interactionTimer.current);
    }

    interactionTimer.current = window.setTimeout(() => {
      userInteracting.current = false;
      if (element) {
        stickToBottom.current = isAtBottom(element);
      }
    }, USER_IDLE_MS);
  }, [ref]);

  const onWheelIntent = useCallback(
    (_event: WheelEvent<HTMLDivElement>) => {
      markUserInteracting();
    },
    [markUserInteracting],
  );

  const onTouchIntent = useCallback(() => {
    markUserInteracting();
  }, [markUserInteracting]);

  const onKeyIntent = useCallback((event: KeyboardEvent<HTMLDivElement>) => {
    if (SCROLL_UP_KEYS.has(event.key)) {
      markUserInteracting();
    }
  }, [markUserInteracting]);

  return {
    onScrollAnchor,
    onWheelIntent,
    onTouchIntent,
    onKeyIntent,
  };
}
