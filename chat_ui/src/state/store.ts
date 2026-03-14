import { Conversation, Message, ModelStatus } from "../types";

export interface ChatState {
  conversations: Conversation[];
  activeConversationId: string | null;
  messagesByConversation: Record<string, Message[]>;
  drafts: Record<string, string>;
  sending: boolean;
  lastError: string | null;
  modelStatus: ModelStatus | null;
}

export const initialState: ChatState = {
  conversations: [],
  activeConversationId: null,
  messagesByConversation: {},
  drafts: {},
  sending: false,
  lastError: null,
  modelStatus: null,
};

export type ChatAction =
  | { type: "setConversations"; conversations: Conversation[] }
  | { type: "upsertConversation"; conversation: Conversation }
  | { type: "setActiveConversation"; conversationId: string }
  | { type: "setMessages"; conversationId: string; messages: Message[] }
  | { type: "appendMessage"; conversationId: string; message: Message }
  | { type: "patchMessage"; conversationId: string; messageId: string; patch: Partial<Message> }
  | { type: "setDraft"; conversationId: string; value: string }
  | { type: "setSending"; sending: boolean }
  | { type: "setError"; error: string | null }
  | { type: "setModelStatus"; modelStatus: ModelStatus };

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "setConversations":
      return { ...state, conversations: action.conversations };
    case "upsertConversation": {
      const idx = state.conversations.findIndex((c) => c.id === action.conversation.id);
      if (idx < 0) {
        return { ...state, conversations: [action.conversation, ...state.conversations] };
      }
      const next = [...state.conversations];
      next[idx] = action.conversation;
      return { ...state, conversations: next };
    }
    case "setActiveConversation":
      return { ...state, activeConversationId: action.conversationId, lastError: null };
    case "setMessages":
      return {
        ...state,
        messagesByConversation: {
          ...state.messagesByConversation,
          [action.conversationId]: action.messages,
        },
      };
    case "appendMessage": {
      const current = state.messagesByConversation[action.conversationId] ?? [];
      return {
        ...state,
        messagesByConversation: {
          ...state.messagesByConversation,
          [action.conversationId]: [...current, action.message],
        },
      };
    }
    case "patchMessage": {
      const current = state.messagesByConversation[action.conversationId] ?? [];
      const updated = current.map((m) => (m.id === action.messageId ? { ...m, ...action.patch } : m));
      return {
        ...state,
        messagesByConversation: {
          ...state.messagesByConversation,
          [action.conversationId]: updated,
        },
      };
    }
    case "setDraft":
      return {
        ...state,
        drafts: {
          ...state.drafts,
          [action.conversationId]: action.value,
        },
      };
    case "setSending":
      return { ...state, sending: action.sending };
    case "setError":
      return { ...state, lastError: action.error };
    case "setModelStatus":
      return { ...state, modelStatus: action.modelStatus };
    default:
      return state;
  }
}
