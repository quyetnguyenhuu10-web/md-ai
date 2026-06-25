import { rightToolRegistry } from "../../features/right-tools/registry";
import { useSettingsStore } from "../../features/settings/store";

export function RightPanel() {
  const selectedTool = useSettingsStore((state) => state.selectedRightTool);
  const setSelectedTool = useSettingsStore((state) => state.setSelectedRightTool);
  const tool = rightToolRegistry.find((item) => item.id === selectedTool) ?? rightToolRegistry[0];
  const ToolComponent = tool.component;

  return (
    <aside className="right-panel panel">
      <div className="tool-tabs" role="tablist" aria-label="Right tools">
        {rightToolRegistry.map((item) => (
          <button
            key={item.id}
            type="button"
            className={item.id === tool.id ? "active" : ""}
            onClick={() => setSelectedTool(item.id)}
            title={item.label}
          >
            <item.icon size={17} aria-hidden="true" />
          </button>
        ))}
      </div>
      <ToolComponent />
    </aside>
  );
}
