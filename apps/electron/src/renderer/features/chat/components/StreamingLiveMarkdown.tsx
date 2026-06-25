import { Fragment, memo, useMemo, useRef, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import {
  classifyLiveMarkdownBlock,
  createStreamingInlineWindowState,
  getStreamingInlineLiveText,
  updateStreamingInlineWindow,
} from "../streaming/streamingLiveMarkdown";
import { CodeBlock } from "./CodeBlock";

interface StreamingLiveMarkdownProps {
  content: string;
}

interface StreamingInlineTextProps {
  text: string;
}

const LIVE_BLOCK_LIBRARY_LIMIT = 1400;
const MARKDOWN_REMARK_PLUGINS = [remarkGfm, remarkMath];
const MARKDOWN_REHYPE_PLUGINS = [[rehypeKatex, { strict: false, throwOnError: false }]] as never;

const MARKDOWN_COMPONENTS = {
  code({ className, children }: { className?: string; children?: ReactNode }) {
    const isBlock = className?.startsWith("language-");
    const code = String(children).replace(/\n$/, "");
    const language = isBlock ? className?.replace(/^language-/, "") : null;
    return isBlock ? <CodeBlock code={code} language={language} isStreaming /> : <code className="inline-code">{children}</code>;
  },
};

const INLINE_MARKDOWN_COMPONENTS = {
  ...MARKDOWN_COMPONENTS,
  p({ children }: { children?: ReactNode }) {
    return <>{children}</>;
  },
};

const StandardLiveMarkdownBlock = memo(function StandardLiveMarkdownBlock({ content }: StreamingLiveMarkdownProps) {
  return (
    <ReactMarkdown
      remarkPlugins={MARKDOWN_REMARK_PLUGINS}
      rehypePlugins={MARKDOWN_REHYPE_PLUGINS}
      components={MARKDOWN_COMPONENTS}
    >
      {content || "\u00a0"}
    </ReactMarkdown>
  );
});

const InlineMarkdown = memo(function InlineMarkdown({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={MARKDOWN_REMARK_PLUGINS}
      rehypePlugins={MARKDOWN_REHYPE_PLUGINS}
      components={INLINE_MARKDOWN_COMPONENTS}
    >
      {content || "\u00a0"}
    </ReactMarkdown>
  );
});

const StableInlineChunk = memo(function StableInlineChunk({ text }: { text: string }) {
  return <InlineMarkdown content={text} />;
});

const StreamingInlineText = memo(function StreamingInlineText({ text }: StreamingInlineTextProps) {
  const stateRef = useRef(createStreamingInlineWindowState());
  stateRef.current = updateStreamingInlineWindow(stateRef.current, text);
  const liveText = getStreamingInlineLiveText(stateRef.current);
  const stableChunks = stateRef.current.chunks;

  return (
    <>
      {stableChunks.map((chunk) => (
        <StableInlineChunk key={chunk.key} text={chunk.text} />
      ))}
      <InlineMarkdown content={liveText} />
    </>
  );
});

function paragraph(text: string) {
  return (
    <p>
      <StreamingInlineText text={text} />
    </p>
  );
}

function renderMultilineText(text: string) {
  return text.split(/\r\n|\r|\n/).map((line, index, lines) => (
    <Fragment key={index}>
      <StreamingInlineText text={line} />
      {index < lines.length - 1 ? <br /> : null}
    </Fragment>
  ));
}

export function StreamingLiveMarkdown({ content }: StreamingLiveMarkdownProps) {
  const shouldUseLibraryBlock = content.length <= LIVE_BLOCK_LIBRARY_LIMIT;
  const block = useMemo(
    () => (shouldUseLibraryBlock ? null : classifyLiveMarkdownBlock(content)),
    [content, shouldUseLibraryBlock],
  );

  if (shouldUseLibraryBlock) {
    return <StandardLiveMarkdownBlock content={content} />;
  }

  if (!block) {
    return null;
  }

  switch (block.kind) {
    case "empty":
      return <p>&nbsp;</p>;
    case "heading": {
      const HeadingTag = `h${block.level}` as const;
      return (
        <HeadingTag>
          <StreamingInlineText text={block.text} />
        </HeadingTag>
      );
    }
    case "unorderedList":
      return (
        <ul>
          {block.items.map((item, index) => (
            <li key={index}>{renderMultilineText(item)}</li>
          ))}
        </ul>
      );
    case "orderedList":
      return (
        <ol start={block.start}>
          {block.items.map((item, index) => (
            <li key={index}>{renderMultilineText(item)}</li>
          ))}
        </ol>
      );
    case "blockquote":
      return <blockquote>{paragraph(block.text)}</blockquote>;
    case "paragraph":
      return paragraph(block.text);
    default:
      return null;
  }
}
