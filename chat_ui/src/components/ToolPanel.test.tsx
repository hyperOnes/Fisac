import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ToolPanel } from "./ToolPanel";


describe("ToolPanel", () => {
  it("renders tools and inserts command", () => {
    const onInsert = vi.fn();
    render(
      <ToolPanel
        tools={[
          { name: "math.eval", description: "calc", input_hint: '{"expression":"2+2"}' },
          { name: "time.now", description: "time", input_hint: "{}" },
        ]}
        onInsert={onInsert}
      />
    );

    fireEvent.click(screen.getByText("math.eval"));
    expect(onInsert).toHaveBeenCalledTimes(1);
  });
});
