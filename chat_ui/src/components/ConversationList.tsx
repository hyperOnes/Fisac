import { Conversation } from "../types";

interface Props {
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelect: (conversationId: string) => void;
  onNewConversation: () => void;
}

function displayConversationTitle(title: string): string {
  return title === "Fiscal Chat" ? "Fisac Chat" : title;
}

export function ConversationList({
  conversations,
  activeConversationId,
  onSelect,
  onNewConversation,
}: Props) {
  return (
    <aside className="panel conversations-panel">
      <div className="panel-header">
        <h2>Chats</h2>
        <button className="btn btn-primary" onClick={onNewConversation}>
          New
        </button>
      </div>
      <div className="conversation-list">
        {conversations.map((conversation) => {
          const active = conversation.id === activeConversationId;
          return (
            <button
              key={conversation.id}
              className={`conversation-item ${active ? "active" : ""}`}
              onClick={() => onSelect(conversation.id)}
            >
              <div className="conversation-title">{displayConversationTitle(conversation.title)}</div>
              <div className="conversation-preview">{conversation.last_message_preview ?? "No messages yet"}</div>
            </button>
          );
        })}
      </div>
    </aside>
  );
}
