export const DISPLAY_MATH_OPEN = "\\[";
export const DISPLAY_MATH_CLOSE = "\\]";
export const DISPLAY_MATH_DOLLAR = "$$";

export type StreamSegment =
  | { kind: "text"; value: string }
  | {
      kind: "displayMath";
      value: string;
      raw: string;
      complete: boolean;
    }
  | {
      kind: "fencedCode";
      language: string;
      code: string;
      raw: string;
      fence: "```" | "~~~";
      complete: boolean;
    };

interface FenceOpen {
  index: number;
  lineEnd: number;
  marker: string;
  fence: "```" | "~~~";
  length: number;
  language: string;
}

interface FenceClose {
  index: number;
  lineEnd: number;
}

interface MathOpen {
  index: number;
  open: typeof DISPLAY_MATH_OPEN | typeof DISPLAY_MATH_DOLLAR;
  close: typeof DISPLAY_MATH_CLOSE | typeof DISPLAY_MATH_DOLLAR;
  valueStart: number;
}

export function splitProtectedStreamSegments(input: string): StreamSegment[] {
  const segments: StreamSegment[] = [];
  let cursor = 0;

  while (cursor < input.length) {
    const nextFence = findNextFenceOpen(input, cursor);
    const nextMath = findNextMathOpen(input, cursor);

    if (nextFence && (!nextMath || nextFence.index <= nextMath.index)) {
      pushText(segments, input.slice(cursor, nextFence.index));
      const codeSegment = readFencedCodeSegment(input, nextFence);
      segments.push(codeSegment.segment);
      cursor = codeSegment.nextCursor;
      continue;
    }

    if (nextMath) {
      pushText(segments, input.slice(cursor, nextMath.index));
      const closeIndex =
        nextMath.open === DISPLAY_MATH_DOLLAR
          ? findDollarMathClose(input, nextMath.valueStart)
          : input.indexOf(nextMath.close, nextMath.valueStart);

      if (closeIndex === -1) {
        segments.push({
          kind: "displayMath",
          value: input.slice(nextMath.valueStart),
          raw: input.slice(nextMath.index),
          complete: false,
        });
        cursor = input.length;
      } else {
        const raw = input.slice(nextMath.index, closeIndex + nextMath.close.length);
        segments.push({
          kind: "displayMath",
          value: input.slice(nextMath.valueStart, closeIndex),
          raw,
          complete: true,
        });
        cursor = closeIndex + nextMath.close.length;
      }
      continue;
    }

    pushText(segments, input.slice(cursor));
    cursor = input.length;
  }

  return segments;
}

function findNextMathOpen(input: string, start: number): MathOpen | null {
  const squareIndex = input.indexOf(DISPLAY_MATH_OPEN, start);
  const dollar = findNextDollarMathOpen(input, start);

  if (squareIndex === -1 && !dollar) {
    return null;
  }

  const square: MathOpen | null =
    squareIndex === -1
      ? null
      : {
          index: squareIndex,
          open: DISPLAY_MATH_OPEN,
          close: DISPLAY_MATH_CLOSE,
          valueStart: squareIndex + DISPLAY_MATH_OPEN.length,
        };

  if (!square) {
    return dollar;
  }
  if (!dollar) {
    return square;
  }
  return square.index <= dollar.index ? square : dollar;
}

function findNextDollarMathOpen(input: string, start: number): MathOpen | null {
  let lineStart = findLineStartAtOrAfter(input, start);

  while (lineStart < input.length) {
    const lineEnd = findLineEnd(input, lineStart);
    const line = input.slice(lineStart, lineEnd);
    const indentLength = countLeadingSpaces(line);

    if (indentLength <= 3 && line.slice(indentLength).trim() === DISPLAY_MATH_DOLLAR) {
      return {
        index: lineStart,
        open: DISPLAY_MATH_DOLLAR,
        close: DISPLAY_MATH_DOLLAR,
        valueStart: includeLineBreak(input, lineEnd),
      };
    }

    const nextLine = findNextLineStart(input, lineStart);
    if (nextLine <= lineStart) {
      return null;
    }
    lineStart = nextLine;
  }

  return null;
}

function findDollarMathClose(input: string, start: number): number {
  let lineStart = start;

  while (lineStart < input.length) {
    const lineEnd = findLineEnd(input, lineStart);
    const line = input.slice(lineStart, lineEnd);
    const indentLength = countLeadingSpaces(line);

    if (indentLength <= 3 && line.slice(indentLength).trim() === DISPLAY_MATH_DOLLAR) {
      return lineStart + indentLength;
    }

    const nextLine = findNextLineStart(input, lineStart);
    if (nextLine <= lineStart) {
      return -1;
    }
    lineStart = nextLine;
  }

  return -1;
}

function pushText(segments: StreamSegment[], value: string) {
  if (!value) {
    return;
  }
  const previous = segments[segments.length - 1];
  if (previous?.kind === "text") {
    previous.value += value;
  } else {
    segments.push({ kind: "text", value });
  }
}

function readFencedCodeSegment(input: string, open: FenceOpen): { segment: StreamSegment; nextCursor: number } {
  const codeStart = open.lineEnd;
  const close = findFenceClose(input, codeStart, open);

  if (!close) {
    return {
      segment: {
        kind: "fencedCode",
        language: open.language,
        code: input.slice(codeStart),
        raw: input.slice(open.index),
        fence: open.fence,
        complete: false,
      },
      nextCursor: input.length,
    };
  }

  return {
    segment: {
      kind: "fencedCode",
      language: open.language,
      code: input.slice(codeStart, close.index),
      raw: input.slice(open.index, close.lineEnd),
      fence: open.fence,
      complete: true,
    },
    nextCursor: close.lineEnd,
  };
}

function findNextFenceOpen(input: string, start: number): FenceOpen | null {
  let lineStart = findLineStartAtOrAfter(input, start);

  while (lineStart < input.length) {
    const open = readFenceOpenAtLine(input, lineStart);
    if (open) {
      return open;
    }
    const nextLine = findNextLineStart(input, lineStart);
    if (nextLine <= lineStart) {
      return null;
    }
    lineStart = nextLine;
  }

  return null;
}

function findFenceClose(input: string, start: number, open: FenceOpen): FenceClose | null {
  let lineStart = start;

  while (lineStart < input.length) {
    const lineEnd = findLineEnd(input, lineStart);
    const line = input.slice(lineStart, lineEnd);
    const indentLength = countLeadingSpaces(line);
    const trimmedStart = line.slice(indentLength);

    if (
      indentLength <= 3 &&
      trimmedStart.startsWith(open.marker[0].repeat(open.length)) &&
      sameFenceRunLength(trimmedStart, open.marker[0]) >= open.length
    ) {
      const runLength = sameFenceRunLength(trimmedStart, open.marker[0]);
      if (trimmedStart.slice(runLength).trim() === "") {
        return { index: lineStart, lineEnd: includeLineBreak(input, lineEnd) };
      }
    }

    const nextLine = findNextLineStart(input, lineStart);
    if (nextLine <= lineStart) {
      return null;
    }
    lineStart = nextLine;
  }

  return null;
}

function readFenceOpenAtLine(input: string, lineStart: number): FenceOpen | null {
  const lineEnd = findLineEnd(input, lineStart);
  const line = input.slice(lineStart, lineEnd);
  const indentLength = countLeadingSpaces(line);

  if (indentLength > 3) {
    return null;
  }

  const candidate = line.slice(indentLength);
  const fenceChar = candidate[0];
  if (fenceChar !== "`" && fenceChar !== "~") {
    return null;
  }

  const runLength = sameFenceRunLength(candidate, fenceChar);
  if (runLength < 3) {
    return null;
  }

  const marker = fenceChar.repeat(runLength);
  const info = candidate.slice(runLength).trim();
  return {
    index: lineStart,
    lineEnd: includeLineBreak(input, lineEnd),
    marker,
    fence: fenceChar === "`" ? "```" : "~~~",
    length: runLength,
    language: info.split(/\s+/)[0] ?? "",
  };
}

function findLineStartAtOrAfter(input: string, start: number): number {
  if (start === 0 || input[start - 1] === "\n" || input[start - 1] === "\r") {
    return start;
  }
  return findNextLineStart(input, start);
}

function findNextLineStart(input: string, lineStart: number): number {
  const lineEnd = findLineEnd(input, lineStart);
  return includeLineBreak(input, lineEnd);
}

function findLineEnd(input: string, lineStart: number): number {
  const nextLf = input.indexOf("\n", lineStart);
  const nextCr = input.indexOf("\r", lineStart);

  if (nextLf === -1) {
    return nextCr === -1 ? input.length : nextCr;
  }
  if (nextCr === -1) {
    return nextLf;
  }
  return Math.min(nextLf, nextCr);
}

function includeLineBreak(input: string, lineEnd: number): number {
  if (input[lineEnd] === "\r" && input[lineEnd + 1] === "\n") {
    return lineEnd + 2;
  }
  if (input[lineEnd] === "\r" || input[lineEnd] === "\n") {
    return lineEnd + 1;
  }
  return lineEnd;
}

function countLeadingSpaces(value: string): number {
  let count = 0;
  while (value[count] === " ") {
    count += 1;
  }
  return count;
}

function sameFenceRunLength(value: string, fenceChar: string): number {
  let count = 0;
  while (value[count] === fenceChar) {
    count += 1;
  }
  return count;
}
