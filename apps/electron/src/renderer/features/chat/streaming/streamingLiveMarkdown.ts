export type LiveMarkdownBlock =
  | { kind: "empty"; text: "" }
  | { kind: "heading"; level: 1 | 2 | 3 | 4 | 5 | 6; text: string }
  | { kind: "unorderedList"; items: string[] }
  | { kind: "orderedList"; start: number; items: string[] }
  | { kind: "blockquote"; text: string }
  | { kind: "paragraph"; text: string };

export interface StreamingInlineChunk {
  key: string;
  text: string;
}

export interface StreamingInlineWindowState {
  chunks: StreamingInlineChunk[];
  nextChunkId: number;
  source: string;
  stableLength: number;
}

const MIN_LIVE_INLINE_CHARS = 160;
const HARD_LIVE_INLINE_CHARS = 640;

export function createStreamingInlineWindowState(): StreamingInlineWindowState {
  return {
    chunks: [],
    nextChunkId: 0,
    source: "",
    stableLength: 0,
  };
}

export function classifyLiveMarkdownBlock(markdown: string): LiveMarkdownBlock {
  const text = stripLeadingBlankLines(markdown);
  if (text.length === 0) {
    return { kind: "empty", text: "" };
  }

  const heading = /^(#{1,6})[ \t]+(.+)$/.exec(text);
  if (heading && !text.includes("\n")) {
    return {
      kind: "heading",
      level: heading[1].length as 1 | 2 | 3 | 4 | 5 | 6,
      text: heading[2],
    };
  }

  const unorderedList = readList(text, /^([ \t]{0,3})[-+*][ \t]+(.*)$/);
  if (unorderedList) {
    return { kind: "unorderedList", items: unorderedList.items };
  }

  const orderedList = readList(text, /^([ \t]{0,3})(\d+)[.)][ \t]+(.*)$/);
  if (orderedList) {
    return { kind: "orderedList", start: orderedList.start ?? 1, items: orderedList.items };
  }

  const blockquote = readBlockquote(text);
  if (blockquote !== null) {
    return { kind: "blockquote", text: blockquote };
  }

  return { kind: "paragraph", text };
}

export function updateStreamingInlineWindow(
  previous: StreamingInlineWindowState,
  source: string,
): StreamingInlineWindowState {
  if (source.length < previous.source.length || !source.startsWith(previous.source)) {
    return updateStreamingInlineWindow(createStreamingInlineWindowState(), source);
  }

  const commitEnd = findInlineCommitBoundary(source, previous.stableLength);
  if (commitEnd <= previous.stableLength) {
    return { ...previous, source };
  }

  const text = source.slice(previous.stableLength, commitEnd);
  return {
    chunks: [...previous.chunks, { key: `inline:${previous.nextChunkId}`, text }],
    nextChunkId: previous.nextChunkId + 1,
    source,
    stableLength: commitEnd,
  };
}

export function getStreamingInlineLiveText(state: StreamingInlineWindowState): string {
  return state.source.slice(state.stableLength);
}

function stripLeadingBlankLines(value: string): string {
  return value.replace(/^(?:[ \t]*(?:\r\n|\r|\n))+/, "");
}

function readList(
  text: string,
  markerPattern: RegExp,
): { items: string[]; start?: number } | null {
  const lines = text.split(/\r\n|\r|\n/);
  const items: string[] = [];
  let start: number | undefined;

  for (const line of lines) {
    if (line.trim() === "") {
      continue;
    }

    const marker = markerPattern.exec(line);
    if (marker) {
      if (start === undefined && marker.length >= 4) {
        const maybeStart = Number(marker[2]);
        if (Number.isFinite(maybeStart) && maybeStart > 0) {
          start = maybeStart;
        }
      }
      items.push(marker[marker.length - 1]);
      continue;
    }

    if (items.length === 0 || !/^[ \t]{2,}\S/.test(line)) {
      return null;
    }

    items[items.length - 1] = `${items[items.length - 1]}\n${line.trimStart()}`;
  }

  return items.length > 0 ? { items, start } : null;
}

function readBlockquote(text: string): string | null {
  const lines = text.split(/\r\n|\r|\n/);
  const stripped: string[] = [];

  for (const line of lines) {
    if (line.trim() === "") {
      stripped.push("");
      continue;
    }

    const quote = /^[ \t]{0,3}>[ \t]?(.*)$/.exec(line);
    if (!quote) {
      return null;
    }
    stripped.push(quote[1]);
  }

  return stripped.join("\n");
}

function findInlineCommitBoundary(source: string, stableLength: number): number {
  const liveLength = source.length - stableLength;
  if (liveLength <= MIN_LIVE_INLINE_CHARS) {
    return stableLength;
  }

  const softLimit = source.length - MIN_LIVE_INLINE_CHARS;
  for (let index = softLimit; index > stableLength; index -= 1) {
    if (isInlineBoundary(source, index) && isInlineStable(source.slice(stableLength, index))) {
      return index;
    }
  }

  if (liveLength >= HARD_LIVE_INLINE_CHARS) {
    return source.length - MIN_LIVE_INLINE_CHARS;
  }

  return stableLength;
}

function isInlineBoundary(source: string, index: number): boolean {
  const previous = source[index - 1] ?? "";
  const current = source[index] ?? "";
  return current === "\n" || current === " " || current === "\t" || /[.,;:!?)]/.test(previous);
}

function isInlineStable(text: string): boolean {
  return (
    countUnescaped(text, "`") % 2 === 0 &&
    countUnescaped(text, "[") === countUnescaped(text, "]") &&
    countUnescaped(text, "**") % 2 === 0 &&
    countUnescaped(text, "__") % 2 === 0
  );
}

function countUnescaped(text: string, token: string): number {
  let count = 0;
  let cursor = 0;

  while (cursor < text.length) {
    const index = text.indexOf(token, cursor);
    if (index === -1) {
      return count;
    }

    if (index === 0 || text[index - 1] !== "\\") {
      count += 1;
    }

    cursor = index + token.length;
  }

  return count;
}
