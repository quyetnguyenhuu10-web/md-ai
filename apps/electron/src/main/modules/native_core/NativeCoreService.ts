import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { getDataDir, getNativeSidecarPath } from "../../platform/electronPaths";

interface JsonRpcSuccess<T> {
  id: string;
  result: T;
}

interface JsonRpcFailure {
  id: string;
  error: {
    code: string;
    message: string;
  };
}

type JsonRpcResponse<T> = JsonRpcSuccess<T> | JsonRpcFailure;

interface PendingRequest<T> {
  resolve: (value: T) => void;
  reject: (reason: Error) => void;
  timeout: NodeJS.Timeout;
}

export class NativeCoreService {
  private child: ChildProcessWithoutNullStreams | null = null;
  private pending = new Map<string, PendingRequest<unknown>>();
  private buffer = "";
  private requestSeq = 0;
  private readonly timeoutMs = 10000;

  start(): void {
    this.ensureStarted();
  }

  async warmup(): Promise<void> {
    await this.request("ping");
  }

  async request<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    this.ensureStarted();
    const child = this.child;

    if (!child) {
      throw new Error("Native sidecar is not available");
    }

    const id = `req_${Date.now()}_${++this.requestSeq}`;
    const payload = JSON.stringify({ id, method, params });

    return new Promise<T>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Native request timed out: ${method}`));
      }, this.timeoutMs);

      this.pending.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
        timeout,
      });

      child.stdin.write(`${payload}\n`, (error) => {
        if (error) {
          clearTimeout(timeout);
          this.pending.delete(id);
          reject(error);
        }
      });
    });
  }

  stop(): void {
    if (this.child) {
      this.child.kill();
      this.child = null;
    }
  }

  private ensureStarted(): void {
    if (this.child) {
      return;
    }

    const sidecarPath = getNativeSidecarPath();
    const dataDir = getDataDir();
    const conversationStoreDir = path.join(dataDir, "conversations");
    const legacyDatabasePath = path.join(dataDir, "chat.db");
    fs.mkdirSync(dataDir, { recursive: true });
    fs.mkdirSync(conversationStoreDir, { recursive: true });

    if (!fs.existsSync(sidecarPath)) {
      throw new Error(`Native sidecar not found at ${sidecarPath}. Run npm run build:native first.`);
    }

    this.child = spawn(sidecarPath, [conversationStoreDir, legacyDatabasePath], {
      cwd: path.dirname(sidecarPath),
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });

    this.child.stdout.on("data", (chunk: Buffer) => {
      this.handleStdout(chunk.toString("utf8"));
    });

    this.child.stderr.on("data", (chunk: Buffer) => {
      console.error(`[native-core] ${chunk.toString("utf8").trim()}`);
    });

    this.child.on("exit", (code, signal) => {
      const pending = [...this.pending.values()];
      this.pending.clear();
      for (const request of pending) {
        clearTimeout(request.timeout);
        request.reject(new Error(`Native sidecar exited code=${code ?? "null"} signal=${signal ?? "null"}`));
      }
      this.child = null;
      this.buffer = "";
    });
  }

  private handleStdout(text: string): void {
    this.buffer += text;
    let newline = this.buffer.indexOf("\n");

    while (newline >= 0) {
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (line.length > 0) {
        this.handleLine(line);
      }
      newline = this.buffer.indexOf("\n");
    }
  }

  private handleLine(line: string): void {
    let response: JsonRpcResponse<unknown>;
    try {
      response = JSON.parse(line) as JsonRpcResponse<unknown>;
    } catch {
      console.error(`[native-core] invalid JSON line: ${line}`);
      return;
    }

    const pending = this.pending.get(response.id);
    if (!pending) {
      return;
    }

    clearTimeout(pending.timeout);
    this.pending.delete(response.id);

    if ("error" in response) {
      pending.reject(new Error(`${response.error.code}: ${response.error.message}`));
      return;
    }

    pending.resolve(response.result);
  }
}
