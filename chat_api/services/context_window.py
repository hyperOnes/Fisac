from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from chat_api.models import MessageRecord


@dataclass(frozen=True)
class ContextWindowPolicy:
    keep_last_messages: int = 20
    summary_every_user_turns: int = 6

    def trim_messages(self, messages: Sequence[MessageRecord]) -> list[MessageRecord]:
        if self.keep_last_messages <= 0:
            return list(messages)
        return list(messages[-self.keep_last_messages :])

    def should_refresh_summary(self, user_turn_count: int) -> bool:
        if self.summary_every_user_turns <= 0:
            return False
        return user_turn_count > 0 and (user_turn_count % self.summary_every_user_turns == 0)

    def to_text(self, messages: Sequence[MessageRecord]) -> str:
        clipped = self.trim_messages(messages)
        return "\n".join(f"{m.role}: {m.content}" for m in clipped)
