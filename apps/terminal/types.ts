export type Role = "user" | "bot";

export interface Message {
  id: number;
  role: Role;
  text: string;
}

export type WorkerStatus = "spawning" | "loading" | "warming-up" | "ready" | "error";

export type Phase = "idle" | "thinking" | "responding";

export interface CheckpointEntry {
  name: string;
  checkpoint: string;
  vocab: string | null;
}

export interface StreamHandlers {
  onToken: (text: string) => void;
  onDone: () => void;
  onError: (message: string) => void;
}

// Mirrors mintdim_lab.serving.worker_protocol.
export type WorkerEventName =
  | "loading"
  | "warming-up"
  | "ready"
  | "token"
  | "done"
  | "error";

export interface WorkerEvent {
  event: WorkerEventName;
  text?: string;
  message?: string;
  checkpoint?: string;
  vocab?: string;
}

export type WorkerCommandType = "prompt" | "shutdown";

export interface WorkerCommand {
  type: WorkerCommandType;
  text?: string;
}
