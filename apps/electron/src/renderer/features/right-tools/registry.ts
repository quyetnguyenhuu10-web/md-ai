import { BarChart3, Brain, Coins, FileSearch, Search } from "lucide-react";
import { ContextPreview } from "./components/ContextPreview";
import { MemoryInspector } from "./components/MemoryInspector";
import { RenderStatsPanel } from "./components/RenderStatsPanel";
import { SearchPanel } from "./components/SearchPanel";
import { TokenBudgetPanel } from "./components/TokenBudgetPanel";
import type { RightToolDefinition } from "./types";

export const rightToolRegistry: RightToolDefinition[] = [
  { id: "context", label: "Context Preview", icon: FileSearch, component: ContextPreview },
  { id: "memory", label: "Memory Inspector", icon: Brain, component: MemoryInspector },
  { id: "search", label: "Search", icon: Search, component: SearchPanel },
  { id: "budget", label: "Token Budget", icon: Coins, component: TokenBudgetPanel },
  { id: "stats", label: "Render Stats", icon: BarChart3, component: RenderStatsPanel },
];
