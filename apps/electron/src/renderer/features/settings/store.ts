import { create } from "zustand";
import type { RightToolId } from "../right-tools/types";

interface SettingsState {
  selectedRightTool: RightToolId;
  setSelectedRightTool: (tool: RightToolId) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  selectedRightTool: "context",
  setSelectedRightTool: (tool) => set({ selectedRightTool: tool }),
}));
