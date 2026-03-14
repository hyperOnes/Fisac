import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const api = vi.hoisted(() => ({
  createConversation: vi.fn(),
  fetchModelStatus: vi.fn(),
  listConversations: vi.fn(),
  listMessages: vi.fn(),
  listTools: vi.fn(),
  streamReply: vi.fn(),
  updateConversation: vi.fn(),
}));

vi.mock("./api", () => api);

const baseConversation = {
  id: "c1",
  title: "Chat 1",
  created_at: "2026-02-25T00:00:00Z",
  updated_at: "2026-02-25T00:00:00Z",
  gemini_enabled: true,
  last_message_preview: null,
};

const baseModelStatus = {
  ready: true,
  device: "cpu",
  feature_dim: 64,
  num_experts: 128,
  gemini_configured: true,
  gemini_available: true,
  mode: "hybrid",
};

describe("App Gemini toggle", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.fetchModelStatus.mockResolvedValue(baseModelStatus);
    api.listConversations.mockResolvedValue([baseConversation]);
    api.listMessages.mockResolvedValue([]);
    api.listTools.mockResolvedValue([]);
    api.createConversation.mockResolvedValue(baseConversation);
    api.streamReply.mockResolvedValue(undefined);
  });

  it("patches conversation gemini_enabled when toggle is clicked", async () => {
    api.updateConversation.mockResolvedValue({ ...baseConversation, gemini_enabled: false });
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Gemini")).toBeInTheDocument();
    });
    const toggle = screen.getByRole("button", { name: "Gemini" });
    expect(toggle).toHaveTextContent("ON");
    fireEvent.click(toggle);

    await waitFor(() => {
      expect(api.updateConversation).toHaveBeenCalledWith("c1", { gemini_enabled: false });
    });
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Gemini" })).toHaveTextContent("OFF");
    });
  });

  it("rolls back toggle state when patch fails", async () => {
    api.updateConversation.mockRejectedValue(new Error("patch failed"));
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Gemini")).toBeInTheDocument();
    });
    const toggle = screen.getByRole("button", { name: "Gemini" });
    expect(toggle).toHaveTextContent("ON");
    fireEvent.click(toggle);

    await waitFor(() => {
      expect(api.updateConversation).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Gemini" })).toHaveTextContent("ON");
    });
    expect(screen.getByText(/patch failed/i)).toBeInTheDocument();
  });
});
