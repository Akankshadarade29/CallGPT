"""
Purpose: Thread and conversation history management utilities for multi-threaded chat sessions.

This module provides utilities to:
- Generate unique thread IDs
- Manage conversation threads (create, list, switch)
- Load conversation history from checkpointer
- Reset chat sessions
"""

from uuid import uuid4
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, AIMessageChunk


def generate_thread_id() -> str:
    """
    Purpose: Generate a unique thread ID for a new conversation.

    Return Value:
    - str: A unique thread identifier.

    Side Effects:
    - None.

    Examples:
    >>> thread_id = generate_thread_id()
    >>> isinstance(thread_id, str)
    True
    """
    return str(uuid4())


def add_thread(thread_list: List[str], thread_id: str) -> List[str]:
    """
    Purpose: Add a new thread ID to the thread list if it doesn't already exist.

    Parameters:
    - thread_list (List[str]): Current list of thread IDs.
    - thread_id (str): Thread ID to add.

    Return Value:
    - List[str]: Updated thread list.

    Side Effects:
    - None (returns new list, caller must persist).

    Examples:
    >>> threads = []
    >>> threads = add_thread(threads, "thread-1")
    >>> len(threads)
    1
    """
    if thread_id not in thread_list:
        return [*thread_list, thread_id]
    return thread_list


def load_conversation(chatbot: Any, thread_id: str) -> List[BaseMessage]:
    """
    Purpose: Load conversation history (messages) for a given thread from the graph's checkpointer.

    Parameters:
    - chatbot (Any): The compiled LangGraph application with checkpointer.
    - thread_id (str): The thread ID to load.

    Return Value:
    - List[BaseMessage]: List of messages (HumanMessage, AIMessage) from the thread.

    Side Effects:
    - Reads from the checkpointer.

    Examples:
    # messages = load_conversation(chatbot, "thread-123")
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = chatbot.get_state(config=config)
        # Return messages if present, otherwise empty list
        return list(state.values.get("messages", []))
    except Exception as e:
        # If thread doesn't exist or error, return empty
        return []


def convert_messages_to_chat_history(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """
    Purpose: Convert LangChain messages to a simple dict format for UI rendering.

    Parameters:
    - messages (List[BaseMessage]): List of LangChain message objects.

    Return Value:
    - List[Dict[str, str]]: List of dicts with 'role' and 'content' keys.

    Side Effects:
    - None.

    Examples:
    msgs = [HumanMessage(content="Hi"), AIMessage(content="Hello")]
    converted = convert_messages_to_chat_history(msgs)
    converted[0]['role']
    'user'
    """
    chat_history = []
    for msg in messages:
        if isinstance(msg, AIMessageChunk):
            continue
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "system"
        chat_history.append({"role": role, "content": msg.content})
    return chat_history


def reset_chat(
    thread_list: List[str],
    message_histories: Dict[str, List[Dict[str, str]]],
) -> tuple[str, List[str], Dict[str, List[Dict[str, str]]]]:
    """
    Purpose: Create a new chat thread and reset the active conversation.

    Parameters:
    - thread_list (List[str]): Current list of thread IDs.
    - message_histories (Dict[str, List[Dict]]): Dict mapping thread_id to UI message history.

    Return Value:
    - tuple: (new_thread_id, updated_thread_list, updated_message_histories)

    Side Effects:
    - None (returns new values, caller must persist).

    Examples:
    threads = ["thread-1"]
    histories = {"thread-1": [{"role": "user", "content": "Hi"}]}
    new_tid, new_threads, new_histories = reset_chat(threads, histories)
    len(new_threads)
    2
    """
    new_thread_id = generate_thread_id()
    updated_threads = add_thread(thread_list, new_thread_id)
    # Initialize empty history for the new thread
    updated_histories = {**message_histories, new_thread_id: []}
    return new_thread_id, updated_threads, updated_histories


def get_thread_preview(
    chatbot: Any, thread_id: str, max_length: int = 40
) -> str:
    """
    Purpose: Get a short preview/title for a thread based on its first user message.

    Parameters:
    - chatbot (Any): The compiled LangGraph application with checkpointer.
    - thread_id (str): Thread ID.
    - max_length (int): Maximum characters for preview.

    Return Value:
    - str: Preview text or thread_id if no messages.

    Side Effects:
    - Reads from checkpointer.

    Examples:
    # preview = get_thread_preview(chatbot, "thread-123")
    """
    messages = load_conversation(chatbot, thread_id)
    if not messages:
        return f"Thread {thread_id[:8]}..."
    
    # Find first human message
    for msg in messages:
        if isinstance(msg, HumanMessage):
            preview = msg.content[:max_length]
            if len(msg.content) > max_length:
                preview += "..."
            return preview
    
    return f"Thread {thread_id[:8]}..."
