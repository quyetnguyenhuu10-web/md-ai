export const NATIVE_CHANNELS = {
  ping: "native:ping",
  stats: "native:stats",
} as const;

export interface NativeApiContract {
  ping(): Promise<{ ok: boolean }>;
  stats(): Promise<unknown>;
}
