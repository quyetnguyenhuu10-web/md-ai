import { ipcMain } from "electron";
import { UTILITIES_CHANNELS } from "../../../contracts/ipc-contracts/utilities.contract";
import type { RenderCacheArtifact } from "../../../contracts/types/tool";
import type { UtilitiesService } from "./UtilitiesService";

export function registerUtilitiesIpc(utilitiesService: UtilitiesService): void {
  ipcMain.handle(UTILITIES_CHANNELS.searchMessages, (_event, query: string) => utilitiesService.searchMessages(query));
  ipcMain.handle(UTILITIES_CHANNELS.getContextPreview, (_event, conversationId: string, currentInput: string) =>
    utilitiesService.getContextPreview(conversationId, currentInput),
  );
  ipcMain.handle(UTILITIES_CHANNELS.getTokenBudget, (_event, conversationId: string, currentInput: string) =>
    utilitiesService.getTokenBudget(conversationId, currentInput),
  );
  ipcMain.handle(UTILITIES_CHANNELS.getRenderStats, () => utilitiesService.getRenderStats());
  ipcMain.handle(UTILITIES_CHANNELS.getRenderCacheArtifacts, (_event, keys: string[]) =>
    utilitiesService.getRenderCacheArtifacts(keys),
  );
  ipcMain.handle(UTILITIES_CHANNELS.putRenderCacheArtifact, (_event, artifact: RenderCacheArtifact) =>
    utilitiesService.putRenderCacheArtifact(artifact),
  );
  ipcMain.handle(UTILITIES_CHANNELS.getRenderCacheDiskStats, () => utilitiesService.getRenderCacheDiskStats());
  ipcMain.handle(UTILITIES_CHANNELS.clearRenderCache, () => utilitiesService.clearRenderCache());
}
