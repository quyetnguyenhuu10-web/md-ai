export const WINDOW_CHANNELS = {
  minimize: "window:minimize",
  toggleMaximize: "window:toggleMaximize",
  close: "window:close",
} as const;

export interface WindowApiContract {
  minimizeWindow(): Promise<void>;
  toggleMaximizeWindow(): Promise<void>;
  closeWindow(): Promise<void>;
}
