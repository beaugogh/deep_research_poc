from typing_extensions import TypedDict
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    def __str__(self):
        # Override str() to return the value directly, making it easier to use in f-strings/prints
        return self.value


class Dialog:
    ROLE = "role"
    CONTENT = "content"

    def __init__(self, limit: int = 3):
        super().__init__()
        self._dialog = []
        self._limit = limit  # max number of USER turns to keep

    @property
    def dialog(self):
        return self._dialog

    @dialog.setter
    def dialog(self, value):
        self._dialog = value

    @property
    def limit(self):
        return self._limit

    def _remove_one_turn(self):
        while self.dialog and self.dialog[0][self.ROLE] != Role.USER.value:
            self._dialog.pop(0)

        if self.dialog and self.dialog[0][self.ROLE] == Role.USER.value:
            self._dialog.pop(0)

        while self.dialog and self.dialog[0][self.ROLE] != Role.USER.value:
            self._dialog.pop(0)

    def _count_turns(self):
        count = 0
        for turn in self.dialog:
            if turn[self.ROLE] == Role.USER.value:
                count += 1
        return count

    def clear(self):
        self._dialog = []

    def add_message(self, role: Role, content: str):
        # if a user utterance is added, clean the oldest turn when the total number of turns exceeds the limit
        if role == Role.USER:
            num_turns = self._count_turns()
            while num_turns >= self.limit:
                self._remove_one_turn()
                num_turns = self._count_turns()

        self._dialog.append({self.ROLE: role.value, self.CONTENT: content})
        # logger.info(f"{role.value}: {content}")

    def get_last_turn(self):
        sub_dialog = Dialog(limit=self.limit)
        last_turn = []
        if self.dialog:
            n = len(self.dialog)
            for i in range(n):
                utterance = self.dialog[n - 1 - i]
                last_turn.append(utterance)
                if utterance[self.ROLE] == Role.USER.value:
                    break

            last_turn = last_turn[::-1]
        sub_dialog.dialog = last_turn
        return sub_dialog

    def to_string(self):
        return json.dumps(self.dialog, indent=2)
    
    def to_messages(self):
        return self.dialog


class State(TypedDict):
    dialog: Dialog
    tool_call_iterations: int


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dialog = Dialog(limit=3)
    dialog.add_message(Role.SYSTEM, "You are a helpful assistant.")
    dialog.add_message(Role.USER, "Hello!")
    dialog.add_message(Role.ASSISTANT, "Hi! How can I help you?")
    dialog.add_message(Role.USER, "What's the weather like today?")
    messages = dialog.to_string()
    logger.info(messages)
