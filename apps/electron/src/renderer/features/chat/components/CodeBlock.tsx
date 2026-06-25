import { Check, Code2, Copy, X } from "lucide-react";
import type { CSSProperties } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { codeToHtml } from "shiki";
import {
  getCachedCodeHighlightHtml,
  getDiskCachedCodeHighlightHtml,
  makeCodeHighlightCacheKey,
  setCachedCodeHighlightHtml,
} from "../renderCache";
interface CodeBlockProps {
  code: string;
  isStreaming?: boolean;
  language?: string | null;
}

type CopyState = "idle" | "copied" | "failed";

interface HighlightResult {
  code: string;
  html: string;
}

const SHIKI_THEME_DARK = "slack-dark";
const SLACK_DARK_BG = "#222222";
const SLACK_DARK_FG = "#e6e6e6";

const TEXT_LANGUAGE_IDS = new Set(["plain", "plaintext", "text", "txt"]);

const LANGUAGE_LABEL_BY_ID: Record<string, string> = {
  bat: "Batch",
  bash: "Bash",
  c: "C",
  cpp: "C++",
  csharp: "C#",
  css: "CSS",
  csv: "CSV",
  diff: "Diff",
  dockerfile: "Dockerfile",
  dotenv: ".env",
  go: "Go",
  graphql: "GraphQL",
  html: "HTML",
  ini: "INI",
  java: "Java",
  javascript: "JavaScript",
  json: "JSON",
  jsonc: "JSONC",
  jsonl: "JSONL",
  jsx: "JSX",
  kotlin: "Kotlin",
  latex: "LaTeX",
  less: "Less",
  lua: "Lua",
  makefile: "Makefile",
  markdown: "Markdown",
  mdx: "MDX",
  nginx: "NGINX",
  php: "PHP",
  powershell: "PowerShell",
  python: "Python",
  r: "R",
  ruby: "Ruby",
  rust: "Rust",
  sass: "Sass",
  scala: "Scala",
  scss: "SCSS",
  sql: "SQL",
  sqlite: "SQLite",
  svelte: "Svelte",
  swift: "Swift",
  toml: "TOML",
  tsx: "TSX",
  typescript: "TypeScript",
  vue: "Vue",
  wasm: "WebAssembly",
  xml: "XML",
  yaml: "YAML",
  zig: "Zig",
};

const SHIKI_LANGUAGE_ALIASES: Record<string, string> = {
  "c#": "csharp",
  "c++": "cpp",
  batch: "bat",
  cjs: "javascript",
  cmd: "bat",
  docker: "dockerfile",
  env: "dotenv",
  hpp: "cpp",
  hxx: "cpp",
  js: "javascript",
  jsonlines: "jsonl",
  mjs: "javascript",
  md: "markdown",
  plain: "text",
  plaintext: "text",
  ps: "powershell",
  ps1: "powershell",
  pwsh: "powershell",
  py: "python",
  rb: "ruby",
  rs: "rust",
  sh: "bash",
  shell: "bash",
  shellscript: "bash",
  ts: "typescript",
  txt: "text",
  yml: "yaml",
};

const ACRONYM_LABELS = new Set([
  "api",
  "cli",
  "css",
  "csv",
  "dom",
  "html",
  "http",
  "https",
  "ini",
  "json",
  "jsonc",
  "jsonl",
  "jsx",
  "mdx",
  "sql",
  "svg",
  "toml",
  "tsx",
  "xml",
  "yaml",
  "yml",
]);

function normalizeLanguage(value?: string | null): string | null {
  const normalized = String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/^language-/, "")
    .replace(/^["'`]+|["'`]+$/g, "");

  return normalized || null;
}

function resolveLanguageId(language?: string | null): string | null {
  const normalized = normalizeLanguage(language);
  if (!normalized) {
    return null;
  }

  return SHIKI_LANGUAGE_ALIASES[normalized] ?? normalized;
}

function formatUnknownLanguageLabel(languageId: string): string {
  return languageId
    .replace(/[_-]+/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => {
      if (ACRONYM_LABELS.has(part)) {
        return part.toUpperCase();
      }

      return `${part.charAt(0).toUpperCase()}${part.slice(1)}`;
    })
    .join(" ");
}

function getLanguageLabel(language?: string | null): string {
  const languageId = resolveLanguageId(language);
  if (!languageId) {
    return "CODE";
  }

  if (TEXT_LANGUAGE_IDS.has(languageId)) {
    return "Plain Text";
  }

  return LANGUAGE_LABEL_BY_ID[languageId] ?? formatUnknownLanguageLabel(languageId);
}

function resolveShikiLanguage(language?: string | null): string {
  const languageId = resolveLanguageId(language);
  if (!languageId || TEXT_LANGUAGE_IDS.has(languageId)) {
    return "text";
  }

  return languageId;
}

function sanitizeShikiHtml(html: string): string {
  const template = document.createElement("template");
  template.innerHTML = html;

  for (const element of Array.from(template.content.querySelectorAll("*"))) {
    const tagName = element.tagName.toLowerCase();
    if (["script", "iframe", "object", "embed", "link", "meta"].includes(tagName)) {
      element.remove();
      continue;
    }

    for (const attribute of Array.from(element.attributes)) {
      const name = attribute.name.toLowerCase();
      const value = attribute.value.trim().toLowerCase();
      if (name.startsWith("on") || ((name === "href" || name === "src") && value.startsWith("javascript:"))) {
        element.removeAttribute(attribute.name);
      }
    }
  }

  const pre = template.content.querySelector("pre");
  if (pre) {
    pre.classList.add("codeBlockPre");
  }

  return template.innerHTML;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function appendRawSuffixToHighlightedHtml(html: string, suffix: string): string {
  if (!suffix) {
    return html;
  }

  const closeTag = "</code></pre>";
  const closeIndex = html.lastIndexOf(closeTag);
  if (closeIndex === -1) {
    return html;
  }

  return `${html.slice(0, closeIndex)}${escapeHtml(suffix)}${html.slice(closeIndex)}`;
}

async function highlightWithShiki(code: string, language: string): Promise<string> {
  try {
    return sanitizeShikiHtml(await codeToHtml(code, { lang: language, theme: SHIKI_THEME_DARK }));
  } catch {
    return sanitizeShikiHtml(await codeToHtml(code, { lang: "text", theme: SHIKI_THEME_DARK }));
  }
}

async function copyTextToClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  textarea.style.top = "0";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();

  try {
    document.execCommand("copy");
  } finally {
    textarea.remove();
  }
}

export function CodeBlock({ code, isStreaming = false, language }: CodeBlockProps) {
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const [highlighted, setHighlighted] = useState<HighlightResult | null>(null);
  const highlightSeqRef = useRef(0);
  const languageLabel = getLanguageLabel(language);
  const CopyIcon = copyState === "copied" ? Check : copyState === "failed" ? X : Copy;
  const blockStyle = useMemo(
    () => ({ "--code-block-bg": SLACK_DARK_BG, "--code-block-fg": SLACK_DARK_FG }) as CSSProperties,
    [],
  );
  const renderedHighlightedHtml = useMemo(() => {
    if (!highlighted || !code.startsWith(highlighted.code)) {
      return null;
    }

    return appendRawSuffixToHighlightedHtml(highlighted.html, code.slice(highlighted.code.length));
  }, [code, highlighted]);

  useEffect(() => {
    let cancelled = false;
    const seq = highlightSeqRef.current + 1;
    highlightSeqRef.current = seq;
    const highlightCode = code;
    const highlightLanguage = resolveShikiLanguage(language);

    if (isStreaming) {
      setHighlighted((current) => (current && highlightCode.startsWith(current.code) ? current : null));
      return () => {
        cancelled = true;
      };
    }

    const cacheKey = makeCodeHighlightCacheKey({
      code: highlightCode,
      language: highlightLanguage,
      theme: SHIKI_THEME_DARK,
    });

    const cachedHtml = getCachedCodeHighlightHtml(cacheKey);
    if (cachedHtml !== null) {
      setHighlighted({ code: highlightCode, html: cachedHtml });
      return () => {
        cancelled = true;
      };
    }

    setHighlighted((current) => (current && highlightCode.startsWith(current.code) ? current : null));

    void (async () => {
      const diskHtml = await getDiskCachedCodeHighlightHtml(cacheKey);
      if (diskHtml !== null) {
        if (!cancelled && highlightSeqRef.current === seq) {
          setHighlighted({ code: highlightCode, html: diskHtml });
        }
        return;
      }

      const html = await highlightWithShiki(highlightCode, highlightLanguage);
      if (!cancelled && highlightSeqRef.current === seq) {
        setCachedCodeHighlightHtml(cacheKey, html);
        setHighlighted({ code: highlightCode, html });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [code, isStreaming, language]);

  const copyCode = async () => {
    try {
      await copyTextToClipboard(code);
      setCopyState("copied");
      window.setTimeout(() => setCopyState("idle"), 1800);
    } catch {
      setCopyState("failed");
      window.setTimeout(() => setCopyState("idle"), 1200);
    }
  };

  return (
    <div className="codeBlock" style={blockStyle}>
      <div className="codeBlockHeader">
        <span className="codeHeaderMeta">
          <Code2 className="codeHeaderIcon" size={14} aria-hidden="true" />
          <span className="codeLangTag">{languageLabel}</span>
        </span>
        <button
          type="button"
          className={`codeCopyBtn ${copyState !== "idle" ? `is-${copyState}` : ""}`}
          aria-label="Copy code"
          title={copyState === "copied" ? "Copied" : copyState === "failed" ? "Copy failed" : "Copy code"}
          onMouseDown={(event) => event.preventDefault()}
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            void copyCode();
          }}
        >
          <CopyIcon size={14} aria-hidden="true" />
        </button>
      </div>
      {renderedHighlightedHtml ? (
        <div className="codeBlockHighlightHost" dangerouslySetInnerHTML={{ __html: renderedHighlightedHtml }} />
      ) : (
        <pre className="codeBlockPre">
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
}





