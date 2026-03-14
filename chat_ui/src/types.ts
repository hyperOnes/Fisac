export type Role = "user" | "assistant" | "system";
export type MessageStatus = "complete" | "error" | "streaming";

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  gemini_enabled: boolean;
  last_message_preview?: string | null;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  created_at: string;
  status: MessageStatus;
  run_id?: string | null;
  latency_ms?: number | null;
  confidence?: number | null;
  mse?: number | null;
  generation_source?: "gemini" | "gemini_regen" | "deterministic" | string | null;
  generation_attempts?: number | null;
  quality_flags?: string | null;
}

export interface ModelStatus {
  ready: boolean;
  device: "cpu" | "mps" | "cuda" | string;
  feature_dim: number;
  num_experts: number;
  gemini_configured: boolean;
  gemini_available: boolean;
  gemini_key_count?: number;
  gemini_available_key_count?: number;
  openai_configured?: boolean;
  openai_key_count?: number;
  openai_available_key_count?: number;
  external_context_enabled?: boolean;
  context_query_count?: number;
  generation_backend?: string;
  pure_liquid_active?: boolean;
  gemini_context_only?: boolean;
  mode: "deterministic" | "hybrid" | string;
}

export interface ToolSpec {
  name: string;
  description: string;
  input_hint: string;
}

export type SSEAck = { run_id: string; conversation_id: string };
export type SSEToken = { run_id: string; delta: string; index: number };
export type SSEDone = { run_id: string; assistant_message_id: string; latency_ms: number };
export type SSEError = { run_id: string; code: string; message: string };

export interface StreamHandlers {
  onAck?: (payload: SSEAck) => void;
  onToken?: (payload: SSEToken) => void;
  onDone?: (payload: SSEDone) => void;
  onError?: (payload: SSEError) => void;
}
