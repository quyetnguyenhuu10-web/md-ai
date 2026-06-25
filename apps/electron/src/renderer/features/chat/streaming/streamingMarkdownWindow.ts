import { splitProtectedStreamSegments, type StreamSegment } from "./streamSegments";
import { splitStreamingMarkdownBlocks } from "./streamingMarkdownBlocks";

export type StreamingMarkdownWindowSegment = StreamSegment & {
  key: string;
  stable: boolean;
};

export interface StreamingMarkdownWindowState {
  committedSegments: StreamingMarkdownWindowSegment[];
  liveSegments: StreamingMarkdownWindowSegment[];
  nextSegmentId: number;
  source: string;
  stableLength: number;
}

export function createStreamingMarkdownWindowState(): StreamingMarkdownWindowState {
  return {
    committedSegments: [],
    liveSegments: [],
    nextSegmentId: 0,
    source: "",
    stableLength: 0,
  };
}

export function updateStreamingMarkdownWindow(
  previous: StreamingMarkdownWindowState,
  source: string,
): StreamingMarkdownWindowState {
  if (source.length < previous.source.length || !source.startsWith(previous.source)) {
    return updateStreamingMarkdownWindow(createStreamingMarkdownWindowState(), source);
  }

  let nextSegmentId = previous.nextSegmentId;
  let committedSegments = previous.committedSegments;
  const liveSegments: StreamingMarkdownWindowSegment[] = [];
  const tail = source.slice(previous.stableLength);
  const parsedTail = splitProtectedStreamSegments(tail);
  let stableLength = previous.stableLength;
  let tailCursor = 0;

  const commitSegment = (segment: StreamSegment) => {
    if (committedSegments === previous.committedSegments) {
      committedSegments = [...previous.committedSegments];
    }

    committedSegments.push({ ...segment, key: `stable:${nextSegmentId}`, stable: true } as StreamingMarkdownWindowSegment);
    nextSegmentId += 1;
  };

  const pushLiveSegment = (segment: StreamSegment, keyOffset: number) => {
    liveSegments.push({ ...segment, key: `live:${previous.stableLength + keyOffset}`, stable: false } as StreamingMarkdownWindowSegment);
  };

  for (let index = 0; index < parsedTail.length; index += 1) {
    const segment = parsedTail[index];
    const isFinalSegment = index === parsedTail.length - 1;

    if (segment.kind === "text") {
      const blocks = splitStreamingMarkdownBlocks(segment.value);
      const stableBlockCount = isFinalSegment ? Math.max(0, blocks.length - 1) : blocks.length;
      let blockCursor = 0;

      for (let blockIndex = 0; blockIndex < stableBlockCount; blockIndex += 1) {
        const block = blocks[blockIndex];
        commitSegment({ kind: "text", value: block });
        blockCursor += block.length;
        stableLength += block.length;
      }

      const liveText = segment.value.slice(blockCursor);
      if (liveText) {
        pushLiveSegment({ kind: "text", value: liveText }, tailCursor + blockCursor);
      }

      tailCursor += segment.value.length;
      continue;
    }

    if (segment.kind === "fencedCode") {
      if (!segment.complete) {
        pushLiveSegment(segment, tailCursor);
        tailCursor = tail.length;
        break;
      }

      commitSegment(segment);
      stableLength += segment.raw.length;
      tailCursor += segment.raw.length;
      continue;
    }

    if (!segment.complete) {
      pushLiveSegment(segment, tailCursor);
      tailCursor = tail.length;
      break;
    }

    commitSegment(segment);
    stableLength += segment.raw.length;
    tailCursor += segment.raw.length;
  }

  return {
    committedSegments,
    liveSegments,
    nextSegmentId,
    source,
    stableLength,
  };
}
