import { ToolSpec } from "../types";

interface Props {
  tools: ToolSpec[];
  onInsert: (tool: ToolSpec) => void;
}

export function ToolPanel({ tools, onInsert }: Props) {
  return (
    <aside className="panel tools-panel">
      <div className="panel-header">
        <h2>Tools (V2)</h2>
      </div>
      <div className="tool-list">
        {tools.map((tool) => (
          <button key={tool.name} className="tool-item" onClick={() => onInsert(tool)}>
            <div className="tool-name">{tool.name}</div>
            <div className="tool-desc">{tool.description}</div>
            <code className="tool-hint">{tool.input_hint}</code>
          </button>
        ))}
      </div>
    </aside>
  );
}
