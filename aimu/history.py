from typing import List, Dict, Any
from tinydb import TinyDB
from datetime import datetime


class ConversationManager:
    def __init__(self, db_path: str = "chat_history.json", use_last_conversation: bool = False):
        self.__db_path = db_path

        self.__db = TinyDB(db_path)
        self.__conversations_table = self.__db.table("conversations")

        all = self.__conversations_table.all()
        if use_last_conversation and len(all) > 0:
            last = all[-1]
            doc_id = last.doc_id
            messages = self.__conversations_table.get(doc_id=last.doc_id)["messages"]
        else:
            doc_id, messages = self.create_new_conversation()

        self.__doc_id = doc_id
        self.__messages = messages

    def __del__(self):
        self.__db.close()

    def create_new_conversation(self) -> tuple[str, list]:
        messages = []
        return self.__conversations_table.insert({"messages": messages}), messages

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return self.__messages.copy()

    def update_conversation(self, messages: List[Dict[str, Any]]) -> None:
        # Add a timestamp to new messages
        timestamp = datetime.now().isoformat()

        for message in messages[len(self.messages) :]:
            message["timestamp"] = timestamp
            self.__messages.append(message)

        self.__conversations_table.update({"messages": self.__messages}, doc_ids=[self.__doc_id])

    def close(self) -> None:
        self.__db.close()
