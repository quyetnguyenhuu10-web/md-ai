import { defaultTextareaKeyBindings } from "@opentui/core";

const NEWLINE_KEYS = new Set(["return", "kpenter", "linefeed"]);

// Enter submits, Ctrl+J inserts newline.
// Drop the default `meta+return → submit` so we own the submit key.
export const enterSubmitBindings = [
  ...defaultTextareaKeyBindings.filter((b) => {
    if (NEWLINE_KEYS.has(b.name) && b.action === "newline" && !b.shift && !b.meta && !b.ctrl) {
      return false;
    }
    if (NEWLINE_KEYS.has(b.name) && b.action === "submit" && b.meta) {
      return false;
    }
    return true;
  }),
  { name: "return", action: "submit" as const },
  { name: "kpenter", action: "submit" as const },
  { name: "j", ctrl: true, action: "newline" as const },
];
