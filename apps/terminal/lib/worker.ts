import { spawn, type ChildProcessByStdio } from "node:child_process";
import readline from "node:readline";
import type { Readable, Writable } from "node:stream";

import type { CheckpointEntry, WorkerCommand, WorkerEvent } from "../types";
import { resolveRepoPath } from "./checkpoints";

export type WorkerProc = ChildProcessByStdio<Writable, Readable, Readable>;

export interface WorkerEventSink {
  onLoading(): void;
  onWarmingUp(): void;
  onReady(): void;
  onToken(text: string): void;
  onDone(): void;
  onError(message: string): void;
  onExit(code: number | null): void;
}

export interface BuildArgsOptions {
  activeCheckpoint: CheckpointEntry | null;
  userPassedCheckpoint: boolean;
  userArgs: string[];
  repoRoot: string;
}

export function buildWorkerArgs(opts: BuildArgsOptions): string[] {
  const args = ["-u", "src/mintdim_lab/cli/main.py", "chat", "--worker"];
  const c = opts.activeCheckpoint;
  if (c && !opts.userPassedCheckpoint) {
    args.push("--checkpoint", resolveRepoPath(opts.repoRoot, c.checkpoint));
    if (c.vocab) args.push("--vocab-path", resolveRepoPath(opts.repoRoot, c.vocab));
  }
  args.push(...opts.userArgs);
  return args;
}

export interface SpawnWorkerOptions {
  python: string;
  repoRoot: string;
  args: string[];
  events: WorkerEventSink;
}

export function spawnChatWorker(opts: SpawnWorkerOptions): WorkerProc {
  const proc = spawn(opts.python, opts.args, {
    cwd: opts.repoRoot,
    stdio: ["pipe", "pipe", "pipe"],
  }) as WorkerProc;

  const rl = readline.createInterface({ input: proc.stdout });
  rl.on("line", (line) => {
    if (!line.trim()) return;
    let evt: WorkerEvent;
    try {
      evt = JSON.parse(line) as WorkerEvent;
    } catch {
      return;
    }
    dispatchWorkerEvent(evt, opts.events);
  });

  // Buffer last N stderr lines so a Python traceback / spawn failure surfaces
  // in the UI instead of silently dying behind "Worker exited with code N".
  const stderrTail: string[] = [];
  const STDERR_TAIL_LINES = 60;
  const stderrRl = readline.createInterface({ input: proc.stderr });
  stderrRl.on("line", (line) => {
    stderrTail.push(line);
    if (stderrTail.length > STDERR_TAIL_LINES) stderrTail.shift();
  });

  proc.on("exit", (code) => opts.events.onExit(code));
  proc.on("close", (code) => {
    if (code !== 0 && code !== null && stderrTail.length > 0) {
      opts.events.onError(`Worker exited (code ${code}):\n${stderrTail.join("\n")}`);
    }
  });
  proc.on("error", (err) => opts.events.onError(`Spawn failed: ${err.message}`));

  return proc;
}

function dispatchWorkerEvent(evt: WorkerEvent, sink: WorkerEventSink): void {
  switch (evt.event) {
    case "loading":
      sink.onLoading();
      break;
    case "warming-up":
      sink.onWarmingUp();
      break;
    case "ready":
      sink.onReady();
      break;
    case "token":
      sink.onToken(String(evt.text ?? ""));
      break;
    case "done":
      sink.onDone();
      break;
    case "error":
      sink.onError(String(evt.message ?? "unknown error"));
      break;
  }
}

export function killChatWorker(proc: WorkerProc): void {
  const shutdown: WorkerCommand = { type: "shutdown" };
  try {
    proc.stdin.write(JSON.stringify(shutdown) + "\n");
    proc.stdin.end();
  } catch {
    // stdin already closed; nothing to do.
  }
  try {
    proc.kill();
  } catch {
    // process already exited.
  }
}

export function sendPromptCommand(proc: WorkerProc, text: string): void {
  const cmd: WorkerCommand = { type: "prompt", text };
  proc.stdin.write(JSON.stringify(cmd) + "\n");
}
