function rgbToHex(r: number, g: number, b: number): string {
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
  return `#${[clamp(r), clamp(g), clamp(b)]
    .map((c) => c.toString(16).padStart(2, "0"))
    .join("")}`;
}

function lerpRgbToHex(
  base: readonly [number, number, number],
  highlight: readonly [number, number, number],
  intensity: number,
): string {
  return rgbToHex(
    base[0] + (highlight[0] - base[0]) * intensity,
    base[1] + (highlight[1] - base[1]) * intensity,
    base[2] + (highlight[2] - base[2]) * intensity,
  );
}

// Spinner animation.
export const SPINNER_FRAMES = ["✧", "✦", "✶", "✷", "✶", "✦"];
export const TICK_MS = 50;
export const SPINNER_PERIOD_MS = 130;

// Thinking-phase gradient sweep.
export const SWEEP_PERIOD_MS = 1900;
export const SWEEP_START = -0.3;
export const SWEEP_END = 1.3;
export const SWEEP_FALLOFF = 3.2;

// Status colours.
export const ICON_COLOR = "#5cdcb4";
export const RESPONDING_LABEL_COLOR = "#5cdcb4";
export const RESPONDING_META_COLOR = "#6b7280";
const TEXT_BASE_RGB: readonly [number, number, number] = [92, 220, 180];
const TEXT_HIGHLIGHT_RGB: readonly [number, number, number] = [170, 245, 220];

// Input bar.
export const PROMPT_READY_COLOR = "#4ec9b0";
export const PROMPT_MUTED_COLOR = "#666666";
export const PLACEHOLDER_COLOR = "#666666";
export const INPUT_BORDER_COLOR = "#3a3a3a";
export const INPUT_BORDER_ERROR_COLOR = "#ef4444";

// User bubble.
export const USER_BUBBLE_BG = "#3a3a3a";

// Markdown body.
export const MARKDOWN_FG = "#d4d4d4";

export function thinkingGradientColor(i: number, n: number, phase: number): string {
  const pos = n > 1 ? i / (n - 1) : 0.5;
  const intensity = Math.max(0, 1 - Math.abs(pos - phase) * SWEEP_FALLOFF);
  return lerpRgbToHex(TEXT_BASE_RGB, TEXT_HIGHLIGHT_RGB, intensity);
}

export const SYNTAX_RULES = [
  { scope: ["default"], style: { foreground: "#d4d4d4" } },
  { scope: ["markup.bold", "markup.strong"], style: { foreground: "#ffffff", bold: true } },
  { scope: ["markup.italic"], style: { foreground: "#e5e7eb", italic: true } },
  { scope: ["markup.list"], style: { foreground: "#4ade80" } },
  { scope: ["markup.quote"], style: { foreground: "#9ca3af", italic: true } },
  { scope: ["markup.raw", "markup.raw.block"], style: { foreground: "#fbbf24" } },
  { scope: ["markup.raw.inline"], style: { foreground: "#fbbf24", background: "#1f2937" } },
  { scope: ["markup.heading"], style: { foreground: "#22d3ee", bold: true } },
  { scope: ["markup.heading.1"], style: { foreground: "#22d3ee", bold: true } },
  { scope: ["markup.heading.2"], style: { foreground: "#67e8f9", bold: true } },
  { scope: ["markup.heading.3"], style: { foreground: "#a5f3fc", bold: true } },
  { scope: ["markup.link", "markup.link.url"], style: { foreground: "#60a5fa", underline: true } },
  { scope: ["markup.link.label"], style: { foreground: "#93c5fd", underline: true } },
  { scope: ["conceal"], style: { foreground: "#555555" } },
  { scope: ["comment", "comment.documentation"], style: { foreground: "#6b7280", italic: true } },
  { scope: ["string", "symbol", "character", "character.special"], style: { foreground: "#86efac" } },
  { scope: ["number", "boolean", "float"], style: { foreground: "#fbbf24" } },
  { scope: ["keyword", "keyword.operator"], style: { foreground: "#c084fc" } },
  {
    scope: ["keyword.return", "keyword.conditional", "keyword.repeat", "keyword.coroutine"],
    style: { foreground: "#c084fc", italic: true },
  },
  { scope: ["function", "function.call", "function.method"], style: { foreground: "#60a5fa" } },
  { scope: ["type", "type.builtin"], style: { foreground: "#5eead4" } },
  { scope: ["variable", "variable.parameter"], style: { foreground: "#f3f4f6" } },
  { scope: ["operator", "punctuation"], style: { foreground: "#9ca3af" } },
];

export const THINKING_LABEL = "Thinking...";
export const THINKING_CHARS = Array.from(THINKING_LABEL);
