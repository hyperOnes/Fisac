import { ModelStatus as ModelStatusType } from "../types";

interface Props {
  status: ModelStatusType | null;
}

export function ModelStatus({ status }: Props) {
  if (!status) {
    return <div className="model-status">Model: loading...</div>;
  }
  return (
    <div className="model-status">
      <span className={`dot ${status.ready ? "ready" : "not-ready"}`} />
      <span>
        Fisac {status.ready ? "ready" : "booting"} | {status.device} | D={status.feature_dim} | E={status.num_experts} |{" "}
        {status.mode}
      </span>
      {status.pure_liquid_active ? <span className="pill pill-on">Pure Liquid</span> : null}
      {status.gemini_context_only ? <span className="pill pill-off">Gemini context-only</span> : null}
      {status.external_context_enabled ? (
        <span className="pill pill-on">Context probes x{status.context_query_count ?? 0}</span>
      ) : null}
      {typeof status.openai_configured === "boolean" ? (
        <span className={`pill ${status.openai_configured ? "pill-on" : "pill-off"}`}>
          OpenAI {status.openai_configured ? "available" : "off"}
        </span>
      ) : null}
      <span className={`pill ${status.gemini_available ? "pill-on" : "pill-off"}`}>
        Gemini {status.gemini_available ? "available" : "local-only"}
        {typeof status.gemini_key_count === "number" ? ` (${status.gemini_available_key_count ?? 0}/${status.gemini_key_count})` : ""}
      </span>
      {typeof status.openai_key_count === "number" ? (
        <span className={`pill ${status.openai_configured ? "pill-on" : "pill-off"}`}>
          OpenAI keys {status.openai_available_key_count ?? 0}/{status.openai_key_count}
        </span>
      ) : null}
    </div>
  );
}
