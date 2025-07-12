import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from tinydb import TinyDB, Query


class ChatHistory:
    def __init__(self, db_path: str = "chat_history.json"):
        self.db_path = db_path
        self.db = TinyDB(db_path)
        self.messages_table = self.db.table("messages")
        self.current_session_id: Optional[str] = None

    def __del__(self):
        self.db.close()

    def create_or_retrieve_latest_messages(self) -> tuple[str, List[Dict[str, Any]]]:
        try:
            last = self.db.all()[-1]
            messages = self.messages_table.get(doc_id=last.doc_id)
            doc_id = last.doc_id
        except Exception:
            messages = []
            doc_id = self.messages_table.insert(messages)

        return doc_id, messages

    def update_messages(self, messages: List[Dict[str, Any]], doc_id: str) -> None:
        self.messages_table.update(messages, doc_ids=[doc_id])

    def close(self) -> None:
        self.db.close()
