import type { TextareaRenderable } from "@opentui/core";
import { enterSubmitBindings } from "../lib/keymap";
import {
  INPUT_BORDER_COLOR,
  INPUT_BORDER_ERROR_COLOR,
  PLACEHOLDER_COLOR,
  PROMPT_MUTED_COLOR,
  PROMPT_READY_COLOR,
} from "../lib/theme";

export interface InputBarProps {
  placeholder: string;
  ready: boolean;
  busy: boolean;
  error: boolean;
  onRef: (r: TextareaRenderable) => void;
  onSubmit: () => void;
}

export function InputBar(props: InputBarProps) {
  return (
    <box
      border={["top", "bottom"]}
      borderColor={props.error ? INPUT_BORDER_ERROR_COLOR : INPUT_BORDER_COLOR}
      paddingX={1}
      flexDirection="row"
      gap={1}
      flexShrink={0}
    >
      <text fg={props.ready && !props.busy ? PROMPT_READY_COLOR : PROMPT_MUTED_COLOR}>
        {props.busy || !props.ready ? "…" : "›"}
      </text>
      <box flexGrow={1}>
        <textarea
          ref={(r) => {
            props.onRef(r);
            setTimeout(() => r?.focus(), 0);
          }}
          placeholder={props.placeholder}
          placeholderColor={PLACEHOLDER_COLOR}
          minHeight={1}
          maxHeight={6}
          keyBindings={enterSubmitBindings}
          onSubmit={() => {
            setTimeout(() => setTimeout(() => props.onSubmit(), 0), 0);
          }}
        />
      </box>
    </box>
  );
}
