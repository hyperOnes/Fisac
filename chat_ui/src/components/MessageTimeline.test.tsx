import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { MessageTimeline } from "./MessageTimeline";


describe("MessageTimeline", () => {
  it("renders messages and shows retry on inline error", () => {
    const onRetry = vi.fn();
    render(
      <MessageTimeline
        messages={[
          {
            id: "u1",
            role: "user",
            content: "hello",
            created_at: new Date().toISOString(),
            status: "complete",
          },
          {
            id: "a1",
            role: "assistant",
            content: "world",
            created_at: new Date().toISOString(),
            status: "error",
            generation_source: "gemini",
            generation_attempts: 2,
          },
        ]}
        inlineError="stream failed"
        onRetry={onRetry}
      />
    );

    expect(screen.getByText("hello")).toBeInTheDocument();
    expect(screen.getByText("world")).toBeInTheDocument();
    expect(screen.getByText("source: gemini")).toBeInTheDocument();
    expect(screen.getByText("attempts: 2")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Retry"));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("shows quality diagnostics when flags are present", () => {
    render(
      <MessageTimeline
        messages={[
          {
            id: "a2",
            role: "assistant",
            content: "fallback answer",
            created_at: new Date().toISOString(),
            status: "complete",
            quality_flags: "liquid_native_fallback,too_short",
          },
        ]}
        inlineError={null}
      />
    );

    expect(screen.getByText("diagnostics: liquid_native_fallback, too_short")).toBeInTheDocument();
  });
});
