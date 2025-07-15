from typing import List, Dict, Any
from tinydb import TinyDB
from datetime import datetime


class ConversationManager:
    def __init__(self, db_path: str = "chat_history.json", use_last_conversation: bool = False):
        self.db_path = db_path
        self.db = TinyDB(db_path)
        self.conversations_table = self.db.table("conversations")

        all = self.conversations_table.all()
        if use_last_conversation and len(all) > 0:
            last = all[-1]
            self.doc_id = last.doc_id
            self.messages = self.conversations_table.get(doc_id=last.doc_id)["messages"]
        else:
            self.doc_id, self.messages = self.create_new_conversation()

    def __del__(self):
        self.db.close()

    def create_new_conversation(self) -> tuple[str, list]:
        messages = []
        return self.conversations_table.insert({"messages": messages}), messages

    def update_conversation(self, messages: List[Dict[str, Any]]) -> None:
        # Add timestamp to new messages that don't already have one
        timestamp = datetime.now().isoformat()
        for message in messages[len(self.messages) :]:
            if "timestamp" not in message:
                message["timestamp"] = timestamp
                self.messages.append(message)

        self.conversations_table.update({"messages": self.messages}, doc_ids=[self.doc_id])

    def close(self) -> None:
        self.db.close()
