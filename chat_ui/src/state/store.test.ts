import { describe, expect, it } from "vitest";

import { chatReducer, initialState } from "./store";


describe("chatReducer", () => {
  it("preserves per-conversation drafts when switching", () => {
    const withDraftA = chatReducer(initialState, {
      type: "setDraft",
      conversationId: "a",
      value: "draft A",
    });
    const withDraftB = chatReducer(withDraftA, {
      type: "setDraft",
      conversationId: "b",
      value: "draft B",
    });

    expect(withDraftB.drafts.a).toBe("draft A");
    expect(withDraftB.drafts.b).toBe("draft B");
  });

  it("patches a streaming message in place", () => {
    const seeded = chatReducer(initialState, {
      type: "setMessages",
      conversationId: "c1",
      messages: [
        {
          id: "m1",
          role: "assistant",
          content: "",
          status: "streaming",
          created_at: new Date().toISOString(),
        },
      ],
    });

    const patched = chatReducer(seeded, {
      type: "patchMessage",
      conversationId: "c1",
      messageId: "m1",
      patch: { content: "token", status: "streaming" },
    });

    expect(patched.messagesByConversation.c1[0].content).toBe("token");
    expect(patched.messagesByConversation.c1[0].status).toBe("streaming");
  });
});
