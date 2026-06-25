import { createEffect, createSignal, Show, onCleanup, onMount } from "solid-js";
import {
  BoxRenderable,
  MarkdownRenderable,
  SyntaxStyle,
  TextRenderable,
  type TextareaRenderable,
} from "@opentui/core";
import { useRenderer } from "@opentui/solid";
import { join } from "node:path";

import { InputBar } from "./components/InputBar";
import { StatusLine } from "./components/StatusLine";
import { loadCheckpoints } from "./lib/checkpoints";
import { handleSlash } from "./lib/slash";
import {
  USER_BUBBLE_BG,
  SPINNER_FRAMES,
  SPINNER_PERIOD_MS,
  SWEEP_END,
  SWEEP_PERIOD_MS,
  SWEEP_START,
  MARKDOWN_FG,
  SYNTAX_RULES,
  TICK_MS,
} from "./lib/theme";
import {
  buildWorkerArgs,
  killChatWorker,
  sendPromptCommand,
  spawnChatWorker,
  type WorkerProc,
} from "./lib/worker";
import type { CheckpointEntry, Message, Phase, StreamHandlers, WorkerStatus } from "./types";

const HERE = import.meta.dir;
const REPO_ROOT = join(HERE, "..", "..");
const CHECKPOINT_YAML = join(REPO_ROOT, "recipes", "chat", "checkpoints.yaml");
const MESSAGE_MARGIN_X = 2;
const IDLE_FOOTER_HEIGHT = 3;
const ACTIVE_FOOTER_HEIGHT = 4;

interface LauncherArgs {
  python: string;
  workerArgs: string[];
}

// Accept either `--python python` or a bare first token like `python`.
function parseLauncherArgs(argv: string[]): LauncherArgs {
  const workerArgs: string[] = [];
  let python = process.env.MINTDIM_PYTHON ?? "python";
  let pythonWasSet = false;
  let pendingValueFor: string | null = null;
  const valueFlags = new Set([
    "--checkpoint",
    "--vocab-path",
    "--max-new-tokens",
    "--temperature",
    "--top-k",
    "--seed",
  ]);

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (pendingValueFor !== null) {
      workerArgs.push(arg);
      pendingValueFor = null;
      continue;
    }
    if (arg === "--python") {
      const value = argv[i + 1];
      if (!value || value.startsWith("-")) {
        throw new Error("Missing value for --python. Pass e.g. --python python.");
      }
      python = value;
      pythonWasSet = true;
      i += 1;
      continue;
    }
    if (arg.startsWith("--python=")) {
      const value = arg.slice("--python=".length).trim();
      if (!value) {
        throw new Error("Missing value for --python. Pass e.g. --python=python.");
      }
      python = value;
      pythonWasSet = true;
      continue;
    }
    if (valueFlags.has(arg)) {
      workerArgs.push(arg);
      pendingValueFor = arg;
      continue;
    }
    if (!pythonWasSet && workerArgs.length === 0 && !arg.startsWith("-")) {
      python = arg;
      pythonWasSet = true;
      continue;
    }
    workerArgs.push(arg);
  }

  if (pendingValueFor !== null) {
    throw new Error(`Missing value for ${pendingValueFor}.`);
  }

  return { python, workerArgs };
}

export function App() {
  const renderer = useRenderer();
  const syntaxStyle = SyntaxStyle.fromTheme(SYNTAX_RULES);
  const checkpoints = loadCheckpoints(CHECKPOINT_YAML);
  const launchArgs = parseLauncherArgs(process.argv.slice(2));
  const userPassedCheckpoint = launchArgs.workerArgs.some(
    (a) => a === "--checkpoint" || a.startsWith("--checkpoint="),
  );

  const [phase, setPhase] = createSignal<Phase>("idle");
  const [tokenCount, setTokenCount] = createSignal(0);
  const [elapsedMs, setElapsedMs] = createSignal(0);
  const [spinnerFrame, setSpinnerFrame] = createSignal(0);
  const [sweepPhase, setSweepPhase] = createSignal(SWEEP_START);
  const [busy, setBusy] = createSignal(false);
  const [workerStatus, setWorkerStatus] = createSignal<WorkerStatus>("spawning");
  const [workerError, setWorkerError] = createSignal<string | null>(null);
  const [activeCheckpoint, setActiveCheckpoint] = createSignal<CheckpointEntry | null>(
    userPassedCheckpoint ? null : (checkpoints[0] ?? null),
  );

  let nextId = 1;
  let input: TextareaRenderable | undefined;
  let animationTimer: ReturnType<typeof setInterval> | undefined;
  let worker: WorkerProc | undefined;
  let activeStream: StreamHandlers | null = null;

  async function appendBotMessage(text: string) {
    const surface = renderer.createScrollbackSurface({ startOnNewLine: true });
    const contentWidth = Math.max(1, surface.width - MESSAGE_MARGIN_X * 2);
    const markdown = new MarkdownRenderable(surface.renderContext, {
      id: `chat-bot-message-${nextId++}`,
      left: MESSAGE_MARGIN_X,
      width: contentWidth,
      content: text,
      syntaxStyle,
      streaming: false,
      internalBlockMode: "top-level",
      tableOptions: { style: "grid" },
      conceal: true,
      fg: MARKDOWN_FG,
    });

    surface.root.add(markdown);
    try {
      await surface.settle();
      surface.commitRows(0, surface.height, { trailingNewline: true });
      renderer.requestRender();
      await renderer.idle().catch(() => {});
    } finally {
      if (!surface.isDestroyed) {
        surface.destroy();
      }
    }
  }

  async function appendUserMessage(text: string) {
    const surface = renderer.createScrollbackSurface({ startOnNewLine: true });
    const bubbleWidth = Math.max(1, surface.width);
    const bubble = new BoxRenderable(surface.renderContext, {
      id: `chat-user-bubble-${nextId++}`,
      left: 0,
      width: bubbleWidth,
      backgroundColor: USER_BUBBLE_BG,
      shouldFill: true,
    });
    const content = new TextRenderable(surface.renderContext, {
      id: `chat-user-message-${nextId++}`,
      width: "100%",
      content: `❯ ${text}`,
      wrapMode: "word",
      bg: USER_BUBBLE_BG,
    });

    bubble.add(content);
    surface.root.add(bubble);
    try {
      await surface.settle();
      surface.commitRows(0, surface.height, { trailingNewline: true });
      renderer.requestRender();
      await renderer.idle().catch(() => {});
    } finally {
      if (!surface.isDestroyed) {
        surface.destroy();
      }
    }
  }

  async function appendMessage(role: Message["role"], text: string) {
    if (role === "bot") {
      await appendBotMessage(text);
      return;
    }
    await appendUserMessage(text);
  }

  function pushSystemMessage(text: string) {
    void appendMessage("bot", text);
  }

  function startStatus() {
    const start = Date.now();
    let tick = 0;
    setPhase("thinking");
    setTokenCount(0);
    setElapsedMs(0);
    setSpinnerFrame(0);
    setSweepPhase(SWEEP_START);
    animationTimer = setInterval(() => {
      tick += TICK_MS;
      setSpinnerFrame(Math.floor(tick / SPINNER_PERIOD_MS) % SPINNER_FRAMES.length);
      const cycle = (tick % SWEEP_PERIOD_MS) / SWEEP_PERIOD_MS;
      setSweepPhase(SWEEP_START + cycle * (SWEEP_END - SWEEP_START));
      setElapsedMs(Date.now() - start);
    }, TICK_MS);
  }

  function stopStatus() {
    if (animationTimer !== undefined) {
      clearInterval(animationTimer);
      animationTimer = undefined;
    }
    setPhase("idle");
    setTokenCount(0);
    setElapsedMs(0);
  }

  function spawnWorker() {
    setWorkerStatus("spawning");
    setWorkerError(null);
    const args = buildWorkerArgs({
      activeCheckpoint: activeCheckpoint(),
      userPassedCheckpoint,
      userArgs: launchArgs.workerArgs,
      repoRoot: REPO_ROOT,
    });
    const proc = spawnChatWorker({
      python: launchArgs.python,
      repoRoot: REPO_ROOT,
      args,
      events: {
        onLoading: () => setWorkerStatus("loading"),
        onWarmingUp: () => setWorkerStatus("warming-up"),
        onReady: () => setWorkerStatus("ready"),
        onToken: (text) => activeStream?.onToken(text),
        onDone: () => {
          activeStream?.onDone();
          activeStream = null;
        },
        onError: (message) => {
          if (activeStream) {
            activeStream.onError(message);
            activeStream = null;
          } else {
            setWorkerStatus("error");
            setWorkerError(message);
          }
        },
        onExit: (code) => {
          if (worker !== proc) return; // replacement already in flight
          if (workerStatus() !== "ready") {
            setWorkerStatus("error");
            setWorkerError(`Worker exited with code ${code}`);
          } else if (activeStream) {
            activeStream.onError(`Worker exited mid-stream (code ${code})`);
            activeStream = null;
          }
        },
      },
    });
    worker = proc;
  }

  function killWorker() {
    if (!worker) return;
    const dying = worker;
    worker = undefined;
    if (activeStream) {
      activeStream.onError("Worker restarting");
      activeStream = null;
    }
    killChatWorker(dying);
  }

  function restartWorker() {
    killWorker();
    spawnWorker();
  }

  function sendPrompt(text: string, handlers: StreamHandlers) {
    if (!worker || workerStatus() !== "ready") {
      handlers.onError("Worker not ready");
      return;
    }
    activeStream = handlers;
    sendPromptCommand(worker, text);
  }

  onMount(spawnWorker);

  createEffect(() => {
    const next = phase() === "idle" ? IDLE_FOOTER_HEIGHT : ACTIVE_FOOTER_HEIGHT;
    if (renderer.footerHeight !== next) {
      renderer.footerHeight = next;
    }
  });

  onCleanup(() => {
    stopStatus();
    killWorker();
  });

  async function handleSubmit() {
    if (!input || busy()) return;
    const text = input.plainText.trim();
    if (!text) return;

    if (text.startsWith("/")) {
      input.clear();
      handleSlash(text, {
        checkpoints,
        activeCheckpoint,
        workerStatus,
        setActiveCheckpoint,
        pushSystemMessage,
        restartWorker,
        exit: () => process.exit(0),
      });
      return;
    }

    if (workerStatus() !== "ready") return;

    setBusy(true);
    input.clear();
    await appendMessage("user", text);

    startStatus();

    let acc = "";

    await new Promise<void>((resolve) => {
      sendPrompt(text, {
        onToken: (chunk) => {
          if (phase() === "thinking") setPhase("responding");
          acc += chunk;
          setTokenCount((n) => n + 1);
        },
        onDone: () => {
          void (async () => {
            const final = acc || "(empty response)";
            await appendMessage("bot", final);
            stopStatus();
            await renderer.idle().catch(() => {});
            resolve();
          })();
        },
        onError: (msg) => {
          void (async () => {
            await appendMessage("bot", `**Error:** ${msg}`);
            stopStatus();
            await renderer.idle().catch(() => {});
            resolve();
          })();
        },
      });
    });

    setBusy(false);
  }

  const inputPlaceholder = () => {
    const status = workerStatus();
    if (status === "spawning") return "đang spawn worker…";
    if (status === "loading") return "đang load checkpoint…";
    if (status === "warming-up") return "đang warm up inference…";
    if (status === "error") return `lỗi: ${workerError() ?? "unknown"}`;
    if (busy()) return "đang chờ phản hồi…";
    const ckpt = activeCheckpoint();
    const suffix = ckpt ? ` · ${ckpt.name}` : "";
    return `nhập tin nhắn… (/help${suffix})`;
  };

  return (
    <box flexDirection="column" width="100%" height="100%">
      <Show when={phase() !== "idle"}>
        <box flexShrink={0} height={1}>
          <StatusLine
            phase={phase()}
            spinnerFrame={spinnerFrame()}
            sweepPhase={sweepPhase()}
            tokenCount={tokenCount()}
            elapsedMs={elapsedMs()}
          />
        </box>
      </Show>
      <box flexGrow={1} />
      <InputBar
        placeholder={inputPlaceholder()}
        ready={workerStatus() === "ready"}
        busy={busy()}
        error={workerStatus() === "error"}
        onRef={(r) => (input = r)}
        onSubmit={handleSubmit}
      />
    </box>
  );
}
