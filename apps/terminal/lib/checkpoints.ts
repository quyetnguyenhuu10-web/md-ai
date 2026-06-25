import { readFileSync } from "node:fs";
import { isAbsolute, join } from "node:path";
import type { CheckpointEntry } from "../types";

export function resolveRepoPath(repoRoot: string, p: string): string {
  return isAbsolute(p) ? p : join(repoRoot, p);
}

export function loadCheckpoints(yamlPath: string): CheckpointEntry[] {
  try {
    const raw = readFileSync(yamlPath, "utf-8");
    const data = Bun.YAML.parse(raw) as Record<string, unknown> | null;
    if (!data) return [];
    const entries: CheckpointEntry[] = [];
    for (const [name, value] of Object.entries(data)) {
      const v = value as { checkpoint?: unknown[]; vocab?: unknown[] } | undefined;
      const ckpt = v?.checkpoint?.[0];
      const vocab = v?.vocab?.[0] ?? null;
      if (!ckpt) continue;
      entries.push({
        name,
        checkpoint: String(ckpt),
        vocab: vocab ? String(vocab) : null,
      });
    }
    return entries;
  } catch {
    return [];
  }
}
