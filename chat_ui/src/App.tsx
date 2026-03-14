import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import {
  createConversation,
  fetchModelStatus,
  listConversations,
  listMessages,
  listTools,
  streamReply,
  updateConversation,
} from "./api";
import { Composer } from "./components/Composer";
import { ConversationList } from "./components/ConversationList";
import { MessageTimeline } from "./components/MessageTimeline";
import { ModelStatus } from "./components/ModelStatus";
import { ToolPanel } from "./components/ToolPanel";
import { CursorLabPage } from "./pages/CursorLabPage";
import { chatReducer, initialState } from "./state/store";
import { Message, ToolSpec } from "./types";

function nowIso(): string {
  return new Date().toISOString();
}

function randomId(): string {
  if (typeof globalThis.crypto !== "undefined" && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function makeOptimisticMessage(role: "user" | "assistant", content: string, status: Message["status"]): Message {
  return {
    id: `local-${role}-${randomId()}`,
    role,
    content,
    created_at: nowIso(),
    status,
  };
}

export default function App() {
  const [state, dispatch] = useReducer(chatReducer, initialState);
  const [retryPrompt, setRetryPrompt] = useState<string | null>(null);
  const [tools, setTools] = useState<ToolSpec[]>([]);
  const [activeView, setActiveView] = useState<"chat" | "cursor">("chat");
  const [toggleBusy, setToggleBusy] = useState(false);
  const inFlightRef = useRef(false);

  const activeConversationId = state.activeConversationId;
  const activeConversation = useMemo(
    () => state.conversations.find((c) => c.id === activeConversationId) ?? null,
    [state.conversations, activeConversationId]
  );
  const activeMessages = useMemo(() => {
    if (!activeConversationId) {
      return [];
    }
    return state.messagesByConversation[activeConversationId] ?? [];
  }, [activeConversationId, state.messagesByConversation]);

  const refreshConversations = useCallback(async () => {
    const conversations = await listConversations();
    dispatch({ type: "setConversations", conversations });
    if (!state.activeConversationId && conversations.length > 0) {
      dispatch({ type: "setActiveConversation", conversationId: conversations[0].id });
    }
  }, [state.activeConversationId]);

  const ensureConversation = useCallback(async () => {
    const conversations = await listConversations();
    if (conversations.length === 0) {
      const created = await createConversation();
      dispatch({ type: "setConversations", conversations: [created] });
      dispatch({ type: "setActiveConversation", conversationId: created.id });
      return created.id;
    }
    dispatch({ type: "setConversations", conversations });
    const id = state.activeConversationId ?? conversations[0].id;
    dispatch({ type: "setActiveConversation", conversationId: id });
    return id;
  }, [state.activeConversationId]);

  const refreshMessages = useCallback(async (conversationId: string) => {
    const messages = await listMessages(conversationId);
    dispatch({ type: "setMessages", conversationId, messages });
  }, []);

  useEffect(() => {
    void (async () => {
      try {
        const status = await fetchModelStatus();
        dispatch({ type: "setModelStatus", modelStatus: status });
        setTools(await listTools());
        const conversationId = await ensureConversation();
        await refreshMessages(conversationId);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to initialize chat";
        dispatch({ type: "setError", error: message });
      }
    })();
  }, [ensureConversation, refreshMessages]);

  useEffect(() => {
    if (!activeConversationId) {
      return;
    }
    void refreshMessages(activeConversationId);
  }, [activeConversationId, refreshMessages]);

  const sendPrompt = useCallback(
    async (prompt: string) => {
      if (!activeConversationId || inFlightRef.current) {
        return;
      }
      const text = prompt.trim();
      if (!text) {
        return;
      }

      inFlightRef.current = true;
      dispatch({ type: "setSending", sending: true });
      dispatch({ type: "setError", error: null });
      setRetryPrompt(null);

      const optimisticUser = makeOptimisticMessage("user", text, "complete");
      const optimisticAssistant = makeOptimisticMessage("assistant", "", "streaming");

      dispatch({ type: "appendMessage", conversationId: activeConversationId, message: optimisticUser });
      dispatch({ type: "appendMessage", conversationId: activeConversationId, message: optimisticAssistant });
      dispatch({ type: "setDraft", conversationId: activeConversationId, value: "" });

      try {
        let assistantBuffer = "";
        await streamReply(activeConversationId, text, {
          onToken: (payload) => {
            assistantBuffer += payload.delta;
            dispatch({
              type: "patchMessage",
              conversationId: activeConversationId,
              messageId: optimisticAssistant.id,
              patch: {
                content: assistantBuffer,
                status: "streaming",
              },
            });
          },
          onDone: (payload) => {
            dispatch({
              type: "patchMessage",
              conversationId: activeConversationId,
              messageId: optimisticAssistant.id,
              patch: {
                status: "complete",
                latency_ms: payload.latency_ms,
              },
            });
          },
          onError: (payload) => {
            dispatch({
              type: "patchMessage",
              conversationId: activeConversationId,
              messageId: optimisticAssistant.id,
              patch: {
                status: "error",
                content: payload.message,
              },
            });
            dispatch({ type: "setError", error: payload.message });
            dispatch({ type: "setDraft", conversationId: activeConversationId, value: text });
            setRetryPrompt(text);
          },
        });

        await refreshMessages(activeConversationId);
        await refreshConversations();
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown chat error";
        dispatch({ type: "setError", error: message });
        dispatch({
          type: "patchMessage",
          conversationId: activeConversationId,
          messageId: optimisticAssistant.id,
          patch: {
            status: "error",
            content: message,
          },
        });
        dispatch({ type: "setDraft", conversationId: activeConversationId, value: text });
        setRetryPrompt(text);
      } finally {
        dispatch({ type: "setSending", sending: false });
        inFlightRef.current = false;
      }
    },
    [activeConversationId, refreshConversations, refreshMessages]
  );

  const onCreateConversation = useCallback(async () => {
    const created = await createConversation();
    dispatch({ type: "upsertConversation", conversation: created });
    dispatch({ type: "setActiveConversation", conversationId: created.id });
    dispatch({ type: "setMessages", conversationId: created.id, messages: [] });
    setActiveView("chat");
  }, []);

  const onToggleGemini = useCallback(
    async (nextValue: boolean) => {
      if (!activeConversation || toggleBusy) {
        return;
      }
      const previous = activeConversation.gemini_enabled;
      setToggleBusy(true);
      dispatch({
        type: "upsertConversation",
        conversation: { ...activeConversation, gemini_enabled: nextValue },
      });
      try {
        const updated = await updateConversation(activeConversation.id, { gemini_enabled: nextValue });
        dispatch({ type: "upsertConversation", conversation: updated });
      } catch (error) {
        dispatch({
          type: "upsertConversation",
          conversation: { ...activeConversation, gemini_enabled: previous },
        });
        const message = error instanceof Error ? error.message : "Failed to update Gemini toggle";
        dispatch({ type: "setError", error: message });
      } finally {
        setToggleBusy(false);
      }
    },
    [activeConversation, toggleBusy]
  );

  const currentDraft = activeConversationId ? state.drafts[activeConversationId] ?? "" : "";

  return (
    <div className={`app-shell ${activeView === "cursor" ? "app-shell-cursor" : ""}`}>
      {activeView === "chat" && (
        <ConversationList
          conversations={state.conversations}
          activeConversationId={activeConversationId}
          onSelect={(conversationId) => dispatch({ type: "setActiveConversation", conversationId })}
          onNewConversation={onCreateConversation}
        />
      )}
      <section className="main-panel">
        <div className="main-header">
          <div className="header-left">
            <h1>Fisac Chat v1</h1>
            <div className="view-switch">
              <button className={`btn ${activeView === "chat" ? "btn-primary" : ""}`} onClick={() => setActiveView("chat")}>
                Chat
              </button>
              <button className={`btn ${activeView === "cursor" ? "btn-primary" : ""}`} onClick={() => setActiveView("cursor")}>
                Cursor Lab
              </button>
            </div>
          </div>
          <div className="header-right">
            <ModelStatus status={state.modelStatus} />
            {activeView === "chat" && activeConversation && (
              <label className="gemini-toggle">
                <span>Gemini</span>
                <button
                  className={`toggle-btn ${activeConversation.gemini_enabled ? "on" : "off"}`}
                  onClick={() => void onToggleGemini(!activeConversation.gemini_enabled)}
                  disabled={toggleBusy}
                >
                  {activeConversation.gemini_enabled ? "ON" : "OFF"}
                </button>
              </label>
            )}
          </div>
        </div>

        {activeView === "chat" ? (
          <>
            <MessageTimeline
              messages={activeMessages}
              inlineError={state.lastError}
              onRetry={retryPrompt ? () => void sendPrompt(retryPrompt) : undefined}
            />

            <Composer
              value={currentDraft}
              disabled={!activeConversationId || state.sending}
              onChange={(value) =>
                activeConversationId && dispatch({ type: "setDraft", conversationId: activeConversationId, value })
              }
              onSend={() => void sendPrompt(currentDraft)}
            />
          </>
        ) : (
          <CursorLabPage />
        )}
      </section>
      {activeView === "chat" && (
        <ToolPanel
          tools={tools}
          onInsert={(tool) => {
            if (!activeConversationId) {
              return;
            }
            const template =
              tool.name === "context.search"
                ? `/tool ${tool.name} {\"query\":\"keyword\"}`
                : tool.name === "math.eval"
                  ? `/tool ${tool.name} {\"expression\":\"(2+3)*4\"}`
                  : `/tool ${tool.name}`;
            dispatch({ type: "setDraft", conversationId: activeConversationId, value: template });
          }}
        />
      )}
    </div>
  );
}
