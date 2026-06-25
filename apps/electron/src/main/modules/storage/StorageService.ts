import fs from "node:fs";
import { getDataDir } from "../../platform/electronPaths";

export class StorageService {
  ensureDataDir(): string {
    const dir = getDataDir();
    fs.mkdirSync(dir, { recursive: true });
    return dir;
  }
}
