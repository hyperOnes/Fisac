import { Message } from "../types";

interface Props {
  messages: Message[];
  inlineError: string | null;
  onRetry?: () => void;
}

function formatMetric(label: string, value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return `${label}: n/a`;
  }
  if (label === "latency_ms") {
    return `${label}: ${value.toFixed(1)} ms`;
  }
  return `${label}: ${value.toFixed(4)}`;
}

function parseQualityFlags(raw: string | null | undefined): string[] {
  if (!raw) {
    return [];
  }
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

export function MessageTimeline({ messages, inlineError, onRetry }: Props) {
  return (
    <main className="panel timeline-panel">
      <div className="timeline">
        {messages.map((message) => {
          const qualityFlags = parseQualityFlags(message.quality_flags);
          return (
            <article key={message.id} className={`message message-${message.role} status-${message.status}`}>
              <header className="message-header">
                <span>{message.role}</span>
                <span>{new Date(message.created_at).toLocaleTimeString()}</span>
              </header>
              <div className="message-content">{message.content}</div>
              {message.role === "assistant" && (
                <>
                  {qualityFlags.length > 0 && (
                    <div className="message-diagnostics">
                      diagnostics: {qualityFlags.join(", ")}
                    </div>
                  )}
                  <footer className="message-metrics">
                    <span>source: {message.generation_source ?? "n/a"}</span>
                    <span>attempts: {message.generation_attempts ?? "n/a"}</span>
                    <span>{formatMetric("latency_ms", message.latency_ms)}</span>
                    <span>{formatMetric("confidence", message.confidence)}</span>
                    <span>{formatMetric("mse", message.mse)}</span>
                  </footer>
                </>
              )}
            </article>
          );
        })}
      </div>
      {inlineError && (
        <div className="inline-error">
          <span>{inlineError}</span>
          {onRetry && (
            <button className="btn" onClick={onRetry}>
              Retry
            </button>
          )}
        </div>
      )}
    </main>
  );
}
