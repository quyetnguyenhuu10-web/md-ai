import { app } from "electron";
import path from "node:path";

export function getProjectRoot(): string {
  if (app.isPackaged) {
    return path.dirname(process.resourcesPath);
  }

  return process.cwd();
}

export function getDataDir(): string {
  return app.isPackaged ? path.join(app.getPath("userData"), "data") : path.join(getProjectRoot(), "data", "dev");
}

export function getNativeSidecarPath(): string {
  const executable = process.platform === "win32" ? "chat_core_sidecar.exe" : "chat_core_sidecar";
  const root = getProjectRoot();
  const candidates = [
    path.join(root, "native", "build", "sidecar", "Release", executable),
    path.join(root, "native", "build", "sidecar", executable),
    path.join(root, "native", "build", "Release", executable),
    path.join(root, "native", "build", executable),
  ];

  return candidates[0];
}
