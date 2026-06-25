const BLANK_LINE_PATTERN = /\r?\n[ \t]*\r?\n/g;

export function splitStreamingMarkdownBlocks(markdown: string): string[] {
  if (markdown.length === 0) {
    return [markdown];
  }

  const blocks: string[] = [];
  let cursor = 0;

  for (const match of markdown.matchAll(BLANK_LINE_PATTERN)) {
    const index = match.index;
    if (index === undefined) {
      continue;
    }

    const end = index + match[0].length;
    blocks.push(markdown.slice(cursor, end));
    cursor = end;
  }

  if (cursor < markdown.length) {
    blocks.push(markdown.slice(cursor));
  }

  return blocks.length > 0 ? blocks : [markdown];
}
