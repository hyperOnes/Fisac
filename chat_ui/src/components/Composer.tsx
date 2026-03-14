import { KeyboardEvent } from "react";

interface Props {
  value: string;
  disabled?: boolean;
  onChange: (value: string) => void;
  onSend: () => void;
}

export function Composer({ value, disabled, onChange, onSend }: Props) {
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  };

  return (
    <div className="composer-wrap">
      <textarea
        className="composer-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Type your message..."
        disabled={disabled}
      />
      <button className="btn btn-primary" onClick={onSend} disabled={disabled || !value.trim()}>
        Send
      </button>
    </div>
  );
}
