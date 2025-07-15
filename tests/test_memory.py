"""
Tests for ConversationManager class.
"""

import pytest
import os
import tempfile
from unittest.mock import patch
from tinydb import TinyDB

from aimu.memory import ConversationManager


@pytest.fixture
def temp_db_path():
    """Create a temporary database file for testing."""

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        db_path = f.name

    yield db_path

    if os.path.exists(db_path):
        os.unlink(db_path)


def test_init_new_conversation(temp_db_path):
    """Test initialization with new conversation."""

    manager = ConversationManager(db_path=temp_db_path, use_last_conversation=False)

    assert manager.messages == []

    manager.close()


def test_init_use_last_conversation_empty_db(temp_db_path):
    """Test initialization with use_last_conversation=True on empty database."""

    manager = ConversationManager(db_path=temp_db_path, use_last_conversation=True)

    assert manager.messages == []

    manager.close()


def test_init_use_last_conversation_existing_db(temp_db_path):
    """Test initialization with use_last_conversation=True on existing database."""

    # First, create a conversation in the main table to simulate existing data
    db = TinyDB(temp_db_path)
    conversations_table = db.table("conversations")
    doc_id = conversations_table.insert(
        {"messages": [{"role": "user", "content": "Hello", "timestamp": "2023-01-01T00:00:00"}]}
    )

    # Also add to main table for the use_last_conversation logic to find it
    db.insert({"doc_id": doc_id})
    db.close()

    # Now test loading the last conversation
    manager = ConversationManager(db_path=temp_db_path, use_last_conversation=True)

    assert len(manager.messages) == 1
    assert manager.messages[0]["role"] == "user"
    assert manager.messages[0]["content"] == "Hello"
    assert "timestamp" in manager.messages[0]

    manager.close()


def test_create_new_conversation(temp_db_path):
    """Test _create_new_conversation method."""

    manager = ConversationManager(db_path=temp_db_path)

    doc_id, messages = manager.create_new_conversation()

    assert isinstance(doc_id, int)
    assert messages == []

    manager.close()


def test_update_conversation(temp_db_path):
    """Test update_conversation method."""

    manager = ConversationManager(db_path=temp_db_path)

    test_messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]

    manager.update_conversation(test_messages)

    # Verify the messages were stored with timestamps
    updated_messages = manager.messages
    assert len(updated_messages) == 2
    assert updated_messages[0]["role"] == "user"
    assert updated_messages[0]["content"] == "Hello"
    assert "timestamp" in updated_messages[0]
    assert updated_messages[1]["role"] == "assistant"
    assert updated_messages[1]["content"] == "Hi there!"
    assert "timestamp" in updated_messages[1]
    # Both messages should have the same timestamp
    assert updated_messages[0]["timestamp"] == updated_messages[1]["timestamp"]

    manager.close()


def test_update_conversation_multiple_times(temp_db_path):
    """Test updating conversation multiple times."""

    manager = ConversationManager(db_path=temp_db_path)

    # First update
    messages1 = [{"role": "user", "content": "Hello"}]
    manager.update_conversation(messages1)

    # Second update - only adding new messages
    messages2 = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
    manager.update_conversation(messages2)

    # Verify that only the new message was appended
    stored_messages = manager.messages
    assert len(stored_messages) == 2
    assert stored_messages[0]["role"] == "user"
    assert stored_messages[0]["content"] == "Hello"
    assert "timestamp" in stored_messages[0]
    assert stored_messages[1]["role"] == "assistant"
    assert stored_messages[1]["content"] == "Hi there!"
    assert "timestamp" in stored_messages[1]

    manager.close()


def test_default_db_path():
    """Test default database path."""

    manager = ConversationManager()

    assert os.path.exists("chat_history.json")

    manager.close()

    # Clean up default file
    os.unlink("chat_history.json")


def test_multiple_conversations_in_same_db(temp_db_path):
    """Test multiple conversations can be stored in the same database."""

    # Create first conversation
    manager1 = ConversationManager(db_path=temp_db_path, use_last_conversation=False)
    messages1 = [{"role": "user", "content": "First conversation"}]
    manager1.update_conversation(messages1)
    manager1.close()

    # Create second conversation
    manager2 = ConversationManager(db_path=temp_db_path, use_last_conversation=False)
    messages2 = [{"role": "user", "content": "Second conversation"}]
    manager2.update_conversation(messages2)
    manager2.close()

    # Verify we can still access both conversations
    db = TinyDB(temp_db_path)
    table = db.table("conversations")

    assert len(table.all()) == 2

    stored_data1 = table.all()[0]  # First conversation
    stored_data2 = table.all()[1]  # Second conversation
    stored_messages1 = stored_data1["messages"]
    stored_messages2 = stored_data2["messages"]

    assert len(stored_messages1) == 1
    assert stored_messages1[0]["role"] == "user"
    assert stored_messages1[0]["content"] == "First conversation"
    assert "timestamp" in stored_messages1[0]

    assert len(stored_messages2) == 1
    assert stored_messages2[0]["role"] == "user"
    assert stored_messages2[0]["content"] == "Second conversation"
    assert "timestamp" in stored_messages2[0]

    db.close()


def test_conversation_persistence(temp_db_path):
    """Test that conversations persist across manager instances."""

    # Create and populate a conversation
    manager = ConversationManager(db_path=temp_db_path)
    test_messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
    manager.update_conversation(test_messages)

    # Create new manager instance and verify persistence
    manager = ConversationManager(db_path=temp_db_path, use_last_conversation=True)

    assert len(manager.messages) == 2
    assert manager.messages[0]["role"] == "user"
    assert manager.messages[0]["content"] == "Hello"
    assert "timestamp" in manager.messages[0]
    assert manager.messages[1]["role"] == "assistant"
    assert manager.messages[1]["content"] == "Hi there!"
    assert "timestamp" in manager.messages[1]

    manager.close()


def test_complex_message_structure(temp_db_path):
    """Test with complex message structures."""

    manager = ConversationManager(db_path=temp_db_path)

    complex_messages = [
        {"role": "user", "content": "Complex message", "metadata": {"timestamp": "2023-01-01T00:00:00Z"}},
        {
            "role": "assistant",
            "content": "Complex response",
            "tool_calls": [{"id": "call_1", "function": {"name": "test_func"}}],
        },
    ]

    manager.update_conversation(complex_messages)

    # Verify complex structure is preserved and timestamps are added
    updated_messages = manager.messages
    assert len(updated_messages) == 2
    assert updated_messages[0]["role"] == "user"
    assert updated_messages[0]["content"] == "Complex message"
    assert "timestamp" in updated_messages[0]
    assert updated_messages[0]["metadata"] == {"timestamp": "2023-01-01T00:00:00Z"}

    assert updated_messages[1]["role"] == "assistant"
    assert updated_messages[1]["content"] == "Complex response"
    assert "timestamp" in updated_messages[1]
    assert updated_messages[1]["tool_calls"] == [{"id": "call_1", "function": {"name": "test_func"}}]

    manager.close()


def test_timestamp_format(temp_db_path):
    """Test that timestamps are in ISO format."""

    manager = ConversationManager(db_path=temp_db_path)

    messages = [{"role": "user", "content": "Hello"}]
    manager.update_conversation(messages)

    updated_messages = manager.messages
    timestamp = updated_messages[0]["timestamp"]

    # Verify it's a valid ISO format timestamp
    from datetime import datetime

    datetime.fromisoformat(timestamp)  # This will raise an exception if invalid

    manager.close()
