import type { CSSProperties, PointerEvent as ReactPointerEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AppTitleBar } from "./AppTitleBar";
import { ChatPanel } from "./ChatPanel";
import { LeftPanel } from "./LeftPanel";
import { RightPanel } from "./RightPanel";

const DEFAULT_LEFT_PANEL_WIDTH = 260;
const DEFAULT_RIGHT_PANEL_WIDTH = 320;
const MIN_LEFT_PANEL_WIDTH = 210;
const MAX_LEFT_PANEL_WIDTH = 380;
const MIN_RIGHT_PANEL_WIDTH = 260;
const MAX_RIGHT_PANEL_WIDTH = 520;
const SPLITTER_SIZE = 7;

type ResizeTarget = "left" | "right";

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function AppShell() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const shellRef = useRef<HTMLElement>(null);
  const leftPanelWidthRef = useRef(DEFAULT_LEFT_PANEL_WIDTH);
  const rightPanelWidthRef = useRef(DEFAULT_RIGHT_PANEL_WIDTH);
  const resizeGuideRef = useRef<HTMLDivElement | null>(null);
  const guideFrameRef = useRef<number | null>(null);
  const pendingGuideXRef = useRef(0);

  const shellStyle = useMemo(
    () =>
      ({
        "--left-panel-width": `${DEFAULT_LEFT_PANEL_WIDTH}px`,
        "--right-panel-width": `${DEFAULT_RIGHT_PANEL_WIDTH}px`,
      }) as CSSProperties,
    [],
  );

  const applyPanelWidths = useCallback((leftWidth: number, rightWidth: number) => {
    const shell = shellRef.current;
    if (!shell) {
      return;
    }

    shell.style.setProperty("--left-panel-width", `${Math.round(leftWidth)}px`);
    shell.style.setProperty("--right-panel-width", `${Math.round(rightWidth)}px`);
  }, []);

  const moveResizeGuide = useCallback((x: number) => {
    pendingGuideXRef.current = x;

    if (guideFrameRef.current !== null) {
      return;
    }

    guideFrameRef.current = window.requestAnimationFrame(() => {
      guideFrameRef.current = null;
      resizeGuideRef.current?.style.setProperty(
        "transform",
        `translate3d(${Math.round(pendingGuideXRef.current)}px, 0, 0)`,
      );
    });
  }, []);

  const removeResizeGuide = useCallback(() => {
    if (guideFrameRef.current !== null) {
      window.cancelAnimationFrame(guideFrameRef.current);
      guideFrameRef.current = null;
    }

    resizeGuideRef.current?.remove();
    resizeGuideRef.current = null;
  }, []);

  const createResizeGuide = useCallback(
    (shellRect: DOMRect, x: number) => {
      removeResizeGuide();

      const guide = document.createElement("div");
      guide.className = "panel-resize-guide";
      guide.style.top = `${Math.round(shellRect.top)}px`;
      guide.style.height = `${Math.round(shellRect.height)}px`;
      guide.style.transform = `translate3d(${Math.round(x)}px, 0, 0)`;

      document.body.appendChild(guide);
      resizeGuideRef.current = guide;
      pendingGuideXRef.current = x;
    },
    [removeResizeGuide],
  );

  useEffect(() => {
    return () => {
      removeResizeGuide();
      document.body.classList.remove("is-resizing-panels");
    };
  }, [removeResizeGuide]);

  const beginResize = useCallback(
    (target: ResizeTarget) => (event: ReactPointerEvent<HTMLDivElement>) => {
      event.preventDefault();

      const shell = shellRef.current;
      if (!shell) {
        return;
      }

      const shellRect = shell.getBoundingClientRect();
      const startX = event.clientX;
      const startLeftWidth = leftPanelWidthRef.current;
      const startRightWidth = rightPanelWidthRef.current;

      const initialGuideX =
        target === "left"
          ? shellRect.left + startLeftWidth
          : shellRect.right - startRightWidth;

      createResizeGuide(shellRect, initialGuideX);
      document.body.classList.add("is-resizing-panels");

      const handlePointerMove = (moveEvent: PointerEvent) => {
        moveEvent.preventDefault();

        const deltaX = moveEvent.clientX - startX;

        if (target === "left") {
          const nextLeftWidth = clamp(startLeftWidth + deltaX, MIN_LEFT_PANEL_WIDTH, MAX_LEFT_PANEL_WIDTH);
          leftPanelWidthRef.current = nextLeftWidth;
          moveResizeGuide(shellRect.left + nextLeftWidth);
          return;
        }

        const nextRightWidth = clamp(startRightWidth - deltaX, MIN_RIGHT_PANEL_WIDTH, MAX_RIGHT_PANEL_WIDTH);
        rightPanelWidthRef.current = nextRightWidth;
        moveResizeGuide(shellRect.right - nextRightWidth);
      };

      const stopResize = () => {
        document.body.classList.remove("is-resizing-panels");
        removeResizeGuide();
        applyPanelWidths(leftPanelWidthRef.current, rightPanelWidthRef.current);

        window.removeEventListener("pointermove", handlePointerMove);
        window.removeEventListener("pointerup", stopResize);
        window.removeEventListener("pointercancel", stopResize);
        window.removeEventListener("blur", stopResize);
      };

      window.addEventListener("pointermove", handlePointerMove, { passive: false });
      window.addEventListener("pointerup", stopResize);
      window.addEventListener("pointercancel", stopResize);
      window.addEventListener("blur", stopResize);
    },
    [applyPanelWidths, createResizeGuide, moveResizeGuide, removeResizeGuide],
  );

  return (
    <div className="app-root">
      <AppTitleBar sidebarCollapsed={sidebarCollapsed} onToggleSidebar={() => setSidebarCollapsed((value) => !value)} />
      <main ref={shellRef} className={`app-shell${sidebarCollapsed ? " sidebar-collapsed" : ""}`} style={shellStyle}>
        {!sidebarCollapsed && <LeftPanel />}
        {!sidebarCollapsed && (
          <div
            className="panel-splitter panel-splitterLeft"
            role="separator"
            aria-label="Resize sidebar"
            aria-orientation="vertical"
            tabIndex={0}
            onPointerDown={beginResize("left")}
          />
        )}
        <ChatPanel />
        <div
          className="panel-splitter panel-splitterRight"
          role="separator"
          aria-label="Resize inspector"
          aria-orientation="vertical"
          tabIndex={0}
          onPointerDown={beginResize("right")}
        />
        <RightPanel />
      </main>
    </div>
  );
}