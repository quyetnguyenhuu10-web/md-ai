import type { ReactNode } from "react";

interface ToolPanelProps {
  title: string;
  children: ReactNode;
}

export function ToolPanel({ title, children }: ToolPanelProps) {
  return (
    <section className="tool-panel">
      <h2>{title}</h2>
      {children}
    </section>
  );
}
