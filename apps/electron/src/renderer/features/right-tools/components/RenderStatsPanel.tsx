import { Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import type { RenderCacheDiskStats, RenderStats } from "../../../../contracts/types/tool";
import { clearHotRenderCaches, getHotRenderCacheStats, type HotRenderCacheStats } from "../../chat/renderCache";
import { useChatStore } from "../../chat/store";
import { ToolPanel } from "./ToolPanel";

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }

  if (bytes < 1024) {
    return `${Math.round(bytes)} B`;
  }

  const kib = bytes / 1024;
  if (kib < 1024) {
    return `${kib.toFixed(kib >= 10 ? 0 : 1)} KiB`;
  }

  const mib = kib / 1024;
  return `${mib.toFixed(mib >= 10 ? 0 : 1)} MiB`;
}

function readHotStats(): HotRenderCacheStats {
  return getHotRenderCacheStats();
}

function formatWarning(stats: RenderCacheDiskStats | null): string {
  if (!stats || stats.warningLevel === "ok") {
    return "OK";
  }

  return stats.warningLevel.toUpperCase();
}

export function RenderStatsPanel() {
  const visibleMessages = useChatStore((state) => state.visibleMessages);
  const streamUpdateIntervalMs = useChatStore((state) => state.streamUpdateIntervalMs);
  const [nativeStats, setNativeStats] = useState<RenderStats | null>(null);
  const [hotStats, setHotStats] = useState<HotRenderCacheStats>(() => readHotStats());
  const [diskStats, setDiskStats] = useState<RenderCacheDiskStats | null>(null);
  const [isClearingCache, setIsClearingCache] = useState(false);
  const [clearCacheError, setClearCacheError] = useState<string | null>(null);

  useEffect(() => {
    const refreshStats = () => {
      setHotStats(readHotStats());
      void window.chatAPI.getRenderStats().then(setNativeStats);
      void window.chatAPI.getRenderCacheDiskStats().then(setDiskStats);
    };

    const timer = window.setInterval(refreshStats, 1000);
    refreshStats();

    return () => window.clearInterval(timer);
  }, []);

  const streamingFps = streamUpdateIntervalMs > 0 ? Math.round(1000 / streamUpdateIntervalMs) : 0;
  const handleClearRenderCache = async () => {
    setIsClearingCache(true);
    setClearCacheError(null);

    try {
      const clearedStats = await window.chatAPI.clearRenderCache();
      clearHotRenderCaches();
      setHotStats(readHotStats());
      setDiskStats(clearedStats);
    } catch (error) {
      setClearCacheError(error instanceof Error ? error.message : "Could not clear render cache");
    } finally {
      setIsClearingCache(false);
    }
  };

  return (
    <ToolPanel title="Render Stats">
      <div className="tool-stack">
        <div className="metric-row">
          <span>Visible</span>
          <strong>{visibleMessages.length}</strong>
        </div>
        <div className="metric-row">
          <span>Hot cache</span>
          <strong>{formatBytes(hotStats.estimatedBytes)}</strong>
        </div>
        <div className="metric-row">
          <span>Shiki cache</span>
          <strong>
            {hotStats.codeHighlightEntries} / {formatBytes(hotStats.codeHighlightEstimatedBytes)}
          </strong>
        </div>
        <div className="metric-row">
          <span>Height cache</span>
          <strong>
            {hotStats.messageHeightEntries} / {formatBytes(hotStats.messageHeightEstimatedBytes)}
          </strong>
        </div>
        <div className="metric-row">
          <span>Disk cache</span>
          <strong>{formatBytes(diskStats?.diskEstimatedBytes ?? 0)}</strong>
        </div>
        <div className="metric-row">
          <span>Disk artifacts</span>
          <strong>{diskStats?.artifactCount ?? 0}</strong>
        </div>
        <div className="metric-row">
          <span>Disk warning</span>
          <strong>{formatWarning(diskStats)}</strong>
        </div>
        <div className="metric-row">
          <span>Native cached</span>
          <strong>{nativeStats?.cachedMessageCount ?? 0}</strong>
        </div>
        <div className="metric-row">
          <span>Native layouts</span>
          <strong>{nativeStats?.estimatedLayoutCount ?? 0}</strong>
        </div>
        <div className="metric-row">
          <span>Stream FPS</span>
          <strong>{streamingFps}</strong>
        </div>
        <button
          type="button"
          className="tool-actionButton danger"
          disabled={isClearingCache}
          title="Clear render cache"
          onClick={() => void handleClearRenderCache()}
        >
          <Trash2 size={14} aria-hidden="true" />
          <span>{isClearingCache ? "Clearing" : "Clear cache"}</span>
        </button>
        {clearCacheError ? (
          <p className="tool-error" role="alert">
            {clearCacheError}
          </p>
        ) : null}
      </div>
    </ToolPanel>
  );
}
