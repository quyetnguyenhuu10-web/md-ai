import { useState } from "react";
import type { SearchResult } from "../../../../contracts/types/tool";
import { ToolPanel } from "./ToolPanel";

export function SearchPanel() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);

  const search = async () => {
    const trimmed = query.trim();
    if (!trimmed) {
      setResults([]);
      return;
    }
    setResults(await window.chatAPI.searchMessages(trimmed));
  };

  return (
    <ToolPanel title="Search">
      <div className="tool-search">
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search saved messages" />
        <button type="button" onClick={() => void search()}>
          Search
        </button>
      </div>
      <div className="search-results">
        {results.map((result) => (
          <article key={result.message.id}>
            <strong>{result.message.role}</strong>
            <p>{result.message.content}</p>
          </article>
        ))}
      </div>
    </ToolPanel>
  );
}
