import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from tinydb import TinyDB, Query


class ChatHistoryManager:
    def __init__(self, db_path: str = "chat_history.json"):
        self.db_path = db_path
        self.db = TinyDB(db_path)
        self.messages_table = self.db.table("messages")

    def __del__(self):
        self.db.close()

    def create_new_conversation(self) -> str:
        """
        Create a new conversation with an empty message history.
        Returns the document ID of the new conversation.
        """

        doc_id = self.messages_table.insert([])
        return doc_id

    def update_conversation(self, messages: List[Dict[str, Any]], doc_id: str) -> None:
        self.messages_table.update(messages, doc_ids=[doc_id])

    def retrieve_last_conversation(self) -> tuple[str, List[Dict[str, Any]]]:
        try:
            last = self.db.all()[-1]
            messages = self.messages_table.get(doc_id=last.doc_id)
            doc_id = last.doc_id
        except Exception:
            messages = []
            doc_id = self.messages_table.insert(messages)

        return doc_id, messages

    def close(self) -> None:
        self.db.close()
