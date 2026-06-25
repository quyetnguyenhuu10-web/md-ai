import { BrowserWindow, ipcMain } from "electron";
import { WINDOW_CHANNELS } from "../../../contracts/ipc-contracts/window.contract";

function windowFromEvent(event: Electron.IpcMainInvokeEvent): BrowserWindow {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (!win) {
    throw new Error("Window is not available");
  }
  return win;
}

export function registerWindowIpc(): void {
  ipcMain.handle(WINDOW_CHANNELS.minimize, (event) => {
    windowFromEvent(event).minimize();
  });

  ipcMain.handle(WINDOW_CHANNELS.toggleMaximize, (event) => {
    const win = windowFromEvent(event);
    if (win.isMaximized()) {
      win.unmaximize();
      return;
    }
    win.maximize();
  });

  ipcMain.handle(WINDOW_CHANNELS.close, (event) => {
    windowFromEvent(event).close();
  });
}
