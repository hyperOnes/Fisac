import {
  Conversation,
  Message,
  ModelStatus,
  SSEAck,
  SSEDone,
  SSEError,
  SSEToken,
  StreamHandlers,
  ToolSpec,
} from "./types";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed (${res.status})`);
  }
  return (await res.json()) as T;
}

export async function createConversation(title?: string): Promise<Conversation> {
  return requestJson<Conversation>("/api/conversations", {
    method: "POST",
    body: JSON.stringify({ title }),
  });
}

export async function listConversations(): Promise<Conversation[]> {
  const data = await requestJson<{ items: Conversation[] }>("/api/conversations");
  return data.items;
}

export async function listMessages(conversationId: string): Promise<Message[]> {
  const data = await requestJson<{ items: Message[] }>(
    `/api/conversations/${conversationId}/messages?limit=200`
  );
  return data.items;
}

export async function renameConversation(conversationId: string, title: string): Promise<Conversation> {
  return requestJson<Conversation>(`/api/conversations/${conversationId}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export async function updateConversation(
  conversationId: string,
  patch: { title?: string; gemini_enabled?: boolean }
): Promise<Conversation> {
  return requestJson<Conversation>(`/api/conversations/${conversationId}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
}

export async function deleteConversation(conversationId: string): Promise<void> {
  await requestJson<{ ok: boolean }>(`/api/conversations/${conversationId}`, {
    method: "DELETE",
  });
}

export async function fetchModelStatus(): Promise<ModelStatus> {
  return requestJson<ModelStatus>("/api/model/status");
}

export async function listTools(): Promise<ToolSpec[]> {
  const data = await requestJson<{ items: ToolSpec[] }>("/api/tools");
  return data.items;
}

function parseSSEChunk(raw: string): { event?: string; data?: string } {
  const lines = raw.split("\n");
  let event: string | undefined;
  let data: string | undefined;
  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data = line.slice(5).trim();
    }
  }
  return { event, data };
}

export async function streamReply(
  conversationId: string,
  message: string,
  handlers: StreamHandlers,
  clientRequestId?: string
): Promise<void> {
  const response = await fetch("/api/chat/respond", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      conversation_id: conversationId,
      message,
      stream: true,
      client_request_id: clientRequestId,
    }),
  });

  if (!response.ok || !response.body) {
    const text = await response.text();
    throw new Error(text || `SSE request failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const sep = buffer.indexOf("\n\n");
      if (sep < 0) {
        break;
      }
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      if (!frame.trim()) {
        continue;
      }
      const parsed = parseSSEChunk(frame);
      if (!parsed.event || !parsed.data) {
        continue;
      }
      const payload = JSON.parse(parsed.data) as SSEAck | SSEToken | SSEDone | SSEError;
      if (parsed.event === "ack") {
        handlers.onAck?.(payload as SSEAck);
      } else if (parsed.event === "token") {
        handlers.onToken?.(payload as SSEToken);
      } else if (parsed.event === "done") {
        handlers.onDone?.(payload as SSEDone);
      } else if (parsed.event === "error") {
        handlers.onError?.(payload as SSEError);
      }
    }
  }
}
