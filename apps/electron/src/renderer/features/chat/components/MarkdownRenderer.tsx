import { memo, useMemo, useRef } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { splitProtectedStreamSegments, type StreamSegment } from "../streaming/streamSegments";
import { splitStreamingMarkdownBlocks } from "../streaming/streamingMarkdownBlocks";
import {
  createStreamingMarkdownWindowState,
  updateStreamingMarkdownWindow,
  type StreamingMarkdownWindowSegment,
} from "../streaming/streamingMarkdownWindow";
import { CodeBlock } from "./CodeBlock";
import { StreamingLiveMarkdown } from "./StreamingLiveMarkdown";

interface MarkdownRendererProps {
  content: string;
  isStreaming?: boolean;
}

function normalizeInlineLatex(content: string): string {
  return content.replace(/\\\(([\s\S]*?)\\\)/g, (_match, math: string) => `$${math.trim()}$`);
}

function renderDisplayMath(segment: Extract<StreamSegment, { kind: "displayMath" }>): string {
  return segment.complete ? `\n$$\n${segment.value.trim()}\n$$\n` : segment.raw;
}

const MarkdownBlock = memo(function MarkdownBlock({ content, isStreaming }: MarkdownRendererProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[[rehypeKatex, { strict: false, throwOnError: false }]] as never}
      components={{
        code({ className, children }) {
          const isBlock = className?.startsWith("language-");
          const code = String(children).replace(/\n$/, "");
          const language = isBlock ? className?.replace(/^language-/, "") : null;
          return isBlock ? <CodeBlock code={code} language={language} isStreaming={isStreaming} /> : <code className="inline-code">{children}</code>;
        },
      }}
    >
      {content || "\u00a0"}
    </ReactMarkdown>
  );
});

function renderMarkdownBlocks(content: string, keyPrefix: string, isStreaming: boolean) {
  const markdown = normalizeInlineLatex(content);
  if (!isStreaming) {
    return <MarkdownBlock key={keyPrefix} content={markdown} isStreaming={false} />;
  }

  return splitStreamingMarkdownBlocks(markdown).map((block, blockIndex) => (
    <MarkdownBlock key={`${keyPrefix}:${blockIndex}`} content={block} isStreaming={isStreaming} />
  ));
}

function renderSegment(segment: StreamSegment & { key?: string; stable?: boolean }, fallbackKey: string, isStreaming: boolean) {
  const keyPrefix = segment.key ?? fallbackKey;
  const segmentIsStreaming = isStreaming && !segment.stable;

  if (segment.kind === "fencedCode") {
    return (
      <CodeBlock
        key={keyPrefix}
        code={segment.code.replace(/\n$/, "")}
        language={segment.language}
        isStreaming={segmentIsStreaming || !segment.complete}
      />
    );
  }

  if (segment.kind === "displayMath") {
    return <MarkdownBlock key={keyPrefix} content={renderDisplayMath(segment)} isStreaming={segmentIsStreaming} />;
  }

  if (segmentIsStreaming) {
    return <StreamingLiveMarkdown key={keyPrefix} content={segment.value} />;
  }

  return renderMarkdownBlocks(segment.value, keyPrefix, segmentIsStreaming);
}

const StableMarkdownSegments = memo(function StableMarkdownSegments({
  segments,
}: {
  segments: StreamingMarkdownWindowSegment[];
}) {
  return <>{segments.map((segment, index) => renderSegment(segment, `stable:${index}`, false))}</>;
});

function MarkdownRendererComponent({ content, isStreaming = false }: MarkdownRendererProps) {
  const streamingWindowRef = useRef(createStreamingMarkdownWindowState());
  const completeSegments = useMemo(() => (isStreaming ? [] : splitProtectedStreamSegments(content)), [content, isStreaming]);

  if (!isStreaming) {
    streamingWindowRef.current = createStreamingMarkdownWindowState();
  } else {
    streamingWindowRef.current = updateStreamingMarkdownWindow(streamingWindowRef.current, content);
  }

  if (isStreaming) {
    const streamingWindow = streamingWindowRef.current;

    return (
      <div className="markdown-content">
        <StableMarkdownSegments segments={streamingWindow.committedSegments} />
        {streamingWindow.liveSegments.length > 0 ? (
          streamingWindow.liveSegments.map((segment, index) => renderSegment(segment, `live:${index}`, true))
        ) : streamingWindow.committedSegments.length === 0 ? (
          <MarkdownBlock content="\u00a0" isStreaming />
        ) : (
          null
        )}
      </div>
    );
  }

  return (
    <div className="markdown-content">
      {completeSegments.length > 0 ? (
        completeSegments.map((segment, index) => renderSegment(segment, `${segment.kind}:${index}`, false))
      ) : (
        <MarkdownBlock content="\u00a0" isStreaming={false} />
      )}
    </div>
  );
}

export const MarkdownRenderer = memo(MarkdownRendererComponent);
