import type { ComponentType } from "react";
import type { LucideIcon } from "lucide-react";

export type RightToolId = "context" | "memory" | "search" | "budget" | "stats";

export interface RightToolDefinition {
  id: RightToolId;
  label: string;
  icon: LucideIcon;
  component: ComponentType;
}
