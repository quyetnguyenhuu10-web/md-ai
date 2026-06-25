import { Show } from "solid-js";
import {
  ICON_COLOR,
  RESPONDING_LABEL_COLOR,
  RESPONDING_META_COLOR,
  SPINNER_FRAMES,
  THINKING_CHARS,
  thinkingGradientColor,
} from "../lib/theme";
import type { Phase } from "../types";

export interface StatusLineProps {
  phase: Phase;
  spinnerFrame: number;
  sweepPhase: number;
  tokenCount: number;
  elapsedMs: number;
}

export function StatusLine(props: StatusLineProps) {
  return (
    <box paddingX={0} height={1}>
      <text>
        <span style={{ fg: ICON_COLOR }}>{SPINNER_FRAMES[props.spinnerFrame]}</span>{" "}
        <Show
          when={props.phase === "responding"}
          fallback={THINKING_CHARS.map((ch, i) => (
            <span
              style={{ fg: thinkingGradientColor(i, THINKING_CHARS.length, props.sweepPhase) }}
            >
              {ch}
            </span>
          ))}
        >
          <span style={{ fg: RESPONDING_LABEL_COLOR }}>Responding...</span>
          <span style={{ fg: RESPONDING_META_COLOR }}>
            {` (${Math.floor(props.elapsedMs / 1000)}s · ↑ ${props.tokenCount} tokens)`}
          </span>
        </Show>
      </text>
    </box>
  );
}
