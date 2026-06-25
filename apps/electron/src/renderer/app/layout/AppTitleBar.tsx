import { Minus, PanelLeftClose, PanelLeftOpen, Square, X } from "lucide-react";

interface AppTitleBarProps {
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
}

export function AppTitleBar({ sidebarCollapsed, onToggleSidebar }: AppTitleBarProps) {
  return (
    <header className="app-titlebar">
      <div className="titlebar-left">
        <button
          type="button"
          className="titlebar-iconButton"
          onClick={onToggleSidebar}
          aria-label={sidebarCollapsed ? "Show sidebar" : "Hide sidebar"}
        >
          {sidebarCollapsed ? <PanelLeftOpen size={15} aria-hidden="true" /> : <PanelLeftClose size={15} aria-hidden="true" />}
        </button>

      </div>

      <div className="titlebar-dragRegion" aria-hidden="true" />

      <div className="titlebar-windowControls">
        <button type="button" aria-label="Minimize" onClick={() => void window.windowAPI.minimizeWindow()}>
          <Minus size={14} aria-hidden="true" />
        </button>
        <button type="button" aria-label="Maximize" onClick={() => void window.windowAPI.toggleMaximizeWindow()}>
          <Square size={13} aria-hidden="true" />
        </button>
        <button type="button" className="titlebar-close" aria-label="Close" onClick={() => void window.windowAPI.closeWindow()}>
          <X size={15} aria-hidden="true" />
        </button>
      </div>
    </header>
  );
}
