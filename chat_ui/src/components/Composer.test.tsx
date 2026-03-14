import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Composer } from "./Composer";


describe("Composer", () => {
  it("sends on Enter and keeps newline on Shift+Enter", () => {
    const onSend = vi.fn();
    const onChange = vi.fn();

    render(<Composer value="hello" onChange={onChange} onSend={onSend} />);

    const textbox = screen.getByPlaceholderText("Type your message...");
    fireEvent.keyDown(textbox, { key: "Enter", shiftKey: false });
    expect(onSend).toHaveBeenCalledTimes(1);

    fireEvent.keyDown(textbox, { key: "Enter", shiftKey: true });
    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it("renders an accessible text input on compact layouts", () => {
    render(<Composer value="" onChange={() => {}} onSend={() => {}} />);
    expect(screen.getByPlaceholderText("Type your message...")).toBeInTheDocument();
  });
});
