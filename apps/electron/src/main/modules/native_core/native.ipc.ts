import { ipcMain } from "electron";
import { NATIVE_CHANNELS } from "../../../contracts/ipc-contracts/native.contract";
import type { NativeCoreService } from "./NativeCoreService";

export function registerNativeIpc(nativeCore: NativeCoreService): void {
  ipcMain.handle(NATIVE_CHANNELS.ping, () => nativeCore.request<{ ok: boolean }>("ping"));
  ipcMain.handle(NATIVE_CHANNELS.stats, () => nativeCore.request("stats.get"));
}
