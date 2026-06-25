import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { RenderCacheArtifact, RenderCacheDiskStats } from "../../../contracts/types/tool";
import { getDataDir } from "../../platform/electronPaths";

const DISK_CACHE_VERSION = "disk-render-cache-v1";
const SOFT_LIMIT_BYTES = 256 * 1024 * 1024;
const STRONG_LIMIT_BYTES = 512 * 1024 * 1024;
const DANGER_LIMIT_BYTES = 1024 * 1024 * 1024;
const MAX_KEYS_PER_READ = 500;
const MAX_KEY_LENGTH = 4096;
const MAX_STRING_FIELD_LENGTH = 512;
const MAX_ESTIMATED_BYTES = 64 * 1024 * 1024;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function clampString(value: unknown, fallback: string, maxLength = MAX_STRING_FIELD_LENGTH): string {
  if (typeof value !== "string") {
    return fallback;
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return fallback;
  }

  return trimmed.slice(0, maxLength);
}

function normalizeEstimatedBytes(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return 0;
  }

  return Math.min(Math.ceil(value), MAX_ESTIMATED_BYTES);
}

function warningLevel(bytes: number): RenderCacheDiskStats["warningLevel"] {
  if (bytes >= DANGER_LIMIT_BYTES) {
    return "danger";
  }

  if (bytes >= STRONG_LIMIT_BYTES) {
    return "strong";
  }

  if (bytes >= SOFT_LIMIT_BYTES) {
    return "soft";
  }

  return "ok";
}

export class RenderCacheDiskService {
  private cachedStats: RenderCacheDiskStats | null = null;
  private cachedStatsUntil = 0;

  constructor(private readonly dataDir = getDataDir()) {}

  getArtifacts(keys: string[]): RenderCacheArtifact[] {
    const cleanKeys = [...new Set(Array.isArray(keys) ? keys : [])]
      .filter((key): key is string => typeof key === "string" && key.length > 0 && key.length <= MAX_KEY_LENGTH)
      .slice(0, MAX_KEYS_PER_READ);

    if (cleanKeys.length === 0) {
      return [];
    }

    this.ensureCacheDir();

    const artifacts: RenderCacheArtifact[] = [];
    for (const key of cleanKeys) {
      const artifact = this.readArtifact(key);
      if (artifact) {
        artifacts.push(artifact);
      }
    }

    return artifacts;
  }

  putArtifact(artifact: RenderCacheArtifact): void {
    const normalized = this.normalizeArtifact(artifact);
    this.ensureCacheDir();

    const artifactPath = this.artifactPath(normalized.key);
    const tempPath = `${artifactPath}.tmp-${process.pid}-${Date.now()}`;
    fs.writeFileSync(tempPath, JSON.stringify({ version: DISK_CACHE_VERSION, artifact: normalized }), "utf8");
    fs.renameSync(tempPath, artifactPath);
    this.cachedStats = null;
    this.cachedStatsUntil = 0;
  }

  getStats(): RenderCacheDiskStats {
    const now = Date.now();
    if (this.cachedStats && now < this.cachedStatsUntil) {
      return this.cachedStats;
    }

    this.ensureCacheDir();

    let artifactCount = 0;
    let diskEstimatedBytes = 0;

    for (const entry of fs.readdirSync(this.artifactDir(), { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.endsWith(".json")) {
        continue;
      }

      artifactCount += 1;
      diskEstimatedBytes += fs.statSync(path.join(this.artifactDir(), entry.name)).size;
    }

    const stats: RenderCacheDiskStats = {
      artifactCount,
      diskEstimatedBytes,
      softLimitBytes: SOFT_LIMIT_BYTES,
      strongLimitBytes: STRONG_LIMIT_BYTES,
      dangerLimitBytes: DANGER_LIMIT_BYTES,
      warningLevel: warningLevel(diskEstimatedBytes),
    };

    this.cachedStats = stats;
    this.cachedStatsUntil = now + 5000;
    return stats;
  }

  clear(): RenderCacheDiskStats {
    const artifactDir = this.artifactDir();
    this.assertSafeArtifactDir(artifactDir);

    if (fs.existsSync(artifactDir)) {
      const artifactDirStat = fs.lstatSync(artifactDir);
      if (artifactDirStat.isSymbolicLink()) {
        throw new Error("Refusing to clear render cache artifact symlink");
      }

      fs.rmSync(artifactDir, { recursive: true, force: true });
    }

    this.cachedStats = null;
    this.cachedStatsUntil = 0;
    return this.getStats();
  }

  private ensureCacheDir(): void {
    fs.mkdirSync(this.artifactDir(), { recursive: true });
  }

  private cacheRootDir(): string {
    return path.join(this.dataDir, "render-cache");
  }

  private artifactDir(): string {
    return path.join(this.cacheRootDir(), "artifacts");
  }

  private assertSafeArtifactDir(artifactDir: string): void {
    const cacheRoot = path.resolve(this.cacheRootDir());
    const target = path.resolve(artifactDir);

    if (path.basename(cacheRoot) !== "render-cache" || path.basename(target) !== "artifacts" || path.dirname(target) !== cacheRoot) {
      throw new Error(`Refusing to clear unsafe render cache path: ${target}`);
    }
  }

  private artifactPath(key: string): string {
    const digest = crypto.createHash("sha256").update(key).digest("hex");
    return path.join(this.artifactDir(), `${digest}.json`);
  }

  private readArtifact(key: string): RenderCacheArtifact | null {
    const artifactPath = this.artifactPath(key);
    if (!fs.existsSync(artifactPath)) {
      return null;
    }

    try {
      const parsed = JSON.parse(fs.readFileSync(artifactPath, "utf8")) as unknown;
      if (!isRecord(parsed) || parsed.version !== DISK_CACHE_VERSION || !isRecord(parsed.artifact)) {
        return null;
      }

      const artifact = this.normalizeArtifact(parsed.artifact as unknown as RenderCacheArtifact);
      return artifact.key === key ? artifact : null;
    } catch {
      return null;
    }
  }

  private normalizeArtifact(artifact: RenderCacheArtifact): RenderCacheArtifact {
    if (!isRecord(artifact)) {
      throw new Error("Render cache artifact must be an object");
    }

    const key = clampString(artifact.key, "", MAX_KEY_LENGTH);
    if (!key) {
      throw new Error("Render cache artifact key is required");
    }

    if (artifact.kind !== "codeHighlightHtml" && artifact.kind !== "messageHeight") {
      throw new Error(`Unsupported render cache artifact kind: ${String(artifact.kind)}`);
    }

    if (!isRecord(artifact.payload)) {
      throw new Error("Render cache artifact payload must be an object");
    }

    const normalized: RenderCacheArtifact = {
      key,
      kind: artifact.kind,
      rendererVersion: clampString(artifact.rendererVersion, "unknown"),
      contentHash: clampString(artifact.contentHash, "unknown", 1024),
      payload: artifact.payload,
      estimatedBytes: normalizeEstimatedBytes(artifact.estimatedBytes),
      updatedAt: typeof artifact.updatedAt === "number" && Number.isFinite(artifact.updatedAt) ? artifact.updatedAt : Date.now(),
    };

    if (typeof artifact.conversationId === "string" && artifact.conversationId.trim()) {
      normalized.conversationId = artifact.conversationId.trim().slice(0, MAX_STRING_FIELD_LENGTH);
    }

    if (typeof artifact.messageId === "string" && artifact.messageId.trim()) {
      normalized.messageId = artifact.messageId.trim().slice(0, MAX_STRING_FIELD_LENGTH);
    }

    return normalized;
  }
}
