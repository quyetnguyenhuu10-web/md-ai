import type { CheckpointEntry, WorkerStatus } from "../types";

export interface SlashContext {
  checkpoints: CheckpointEntry[];
  activeCheckpoint: () => CheckpointEntry | null;
  workerStatus: () => WorkerStatus;
  setActiveCheckpoint: (c: CheckpointEntry) => void;
  pushSystemMessage: (text: string) => void;
  restartWorker: () => void;
  exit: () => void;
}

export function handleSlash(text: string, ctx: SlashContext): void {
  const stripped = text.slice(1).trim();
  const [cmd, ...rest] = stripped.split(/\s+/);
  const arg = rest.join(" ");
  switch (cmd?.toLowerCase()) {
    case "help":
      ctx.pushSystemMessage(helpText());
      return;
    case "switch":
      ctx.pushSystemMessage(switchListText(ctx));
      return;
    case "use":
      applySwitch(arg, ctx);
      return;
    case "exit":
    case "quit":
      ctx.exit();
      return;
    default:
      ctx.pushSystemMessage(`Unknown command \`/${cmd}\`. Try \`/help\`.`);
  }
}

function helpText(): string {
  return [
    "**Slash commands:**",
    "- `/help` — danh sách lệnh",
    "- `/switch` — list checkpoint khả dụng",
    "- `/use <name|num>` — đổi checkpoint (restart worker)",
    "- `/exit` — thoát chat",
  ].join("\n");
}

function switchListText(ctx: SlashContext): string {
  if (ctx.checkpoints.length === 0) {
    return "Không có checkpoint nào trong `checkpoints.yaml`.";
  }
  const active = ctx.activeCheckpoint();
  const lines = ctx.checkpoints.map((c, i) => {
    const marker = active && c.name === active.name ? "●" : "○";
    return `${marker} **${i + 1}.** \`${c.name}\` — \`${c.checkpoint}\``;
  });
  return [
    "**Checkpoints khả dụng:**",
    ...lines,
    "",
    "Đổi bằng `/use <số>` hoặc `/use <name>`.",
  ].join("\n");
}

function applySwitch(arg: string, ctx: SlashContext): void {
  if (!arg) {
    ctx.pushSystemMessage("Cú pháp: `/use <số|name>`. Vd `/use 1` hoặc `/use checkpoint_1`.");
    return;
  }
  let target: CheckpointEntry | undefined;
  const asNum = Number.parseInt(arg, 10);
  if (!Number.isNaN(asNum) && asNum >= 1 && asNum <= ctx.checkpoints.length) {
    target = ctx.checkpoints[asNum - 1];
  } else {
    target = ctx.checkpoints.find((c) => c.name === arg);
  }
  if (!target) {
    ctx.pushSystemMessage(`Không tìm thấy checkpoint \`${arg}\`. Dùng \`/switch\` để xem list.`);
    return;
  }
  if (ctx.activeCheckpoint()?.name === target.name && ctx.workerStatus() === "ready") {
    ctx.pushSystemMessage(`Checkpoint \`${target.name}\` đang active rồi.`);
    return;
  }
  ctx.setActiveCheckpoint(target);
  ctx.pushSystemMessage(`Switching to \`${target.name}\`…`);
  ctx.restartWorker();
}
