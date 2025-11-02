"""
Purpose: Streaming utilities for LangGraph chatbot outputs.
"""

from typing import Any, Dict, Generator, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk


def build_messages_state(
    user_input: str,
    base_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Purpose: Build a state dictionary that carries a 'messages' channel with the
    current user input, optionally merging additional keys from an existing base state.

    Parameters:
    - user_input (str): The user's input for this turn.
    - base_state (Optional[Dict[str, Any]]): Existing state to merge (e.g., index_dir,
      model params). If provided, its keys are copied into the returned state.

    Return Value:
    - Dict[str, Any]: State including a 'messages' key suitable for `chatbot.stream`.

    Side Effects:
    - None.

    Examples:
    s = build_messages_state("Hello")
    isinstance(s["messages"], list)
    True
    """
    state: Dict[str, Any] = dict(base_state) if base_state else {}
    state["messages"] = [HumanMessage(content=user_input)]
    # Ensure compatibility with current pipeline nodes which expect 'question'
    # for retrieval and prompt formatting.
    state.setdefault("question", user_input)
    return state


def stream_messages(
    chatbot: Any,
    state: Dict[str, Any],
    thread_id: str,
    stream_mode: str = "messages",
) -> Generator[BaseMessage, None, None]:
    """
    Purpose: Stream message chunks emitted by a LangGraph chatbot run.

    Parameters:
    - chatbot (Any): Compiled LangGraph app.
    - state (Dict[str, Any]): Initial state for the graph (should include
      'messages' or any required fields by the graph).
    - thread_id (str): Thread identifier used by the LangGraph checkpointer.
    - stream_mode (str): Streaming mode; default 'messages' to stream message events.

    Return Value:
    - Generator[BaseMessage, None, None]: Yields `BaseMessage` chunks as produced by the graph.

    Side Effects:
    - Reads from the graph and its checkpointer; no writes performed by this helper.

    Examples:
    for chunk, meta in chatbot.stream(state, config=..., stream_mode='messages'):
    #     ...
    """
    config = {"configurable": {"thread_id": thread_id}}
    for message_chunk, _metadata in chatbot.stream(
        state, config=config, stream_mode=stream_mode
    ):
        # message_chunk is a BaseMessage (HumanMessage/AIMessage/SystemMessage/...)
        yield message_chunk


def stream_ai_tokens(
    chatbot: Any,
    state: Dict[str, Any],
    thread_id: str,
    stream_mode: str = "messages",
) -> Generator[str, None, None]:
    """
    Purpose: Stream only assistant (AI) message chunks' content.

    Parameters:
    - chatbot (Any): Compiled LangGraph app.
    - state (Dict[str, Any]): Initial state for the graph.
    - thread_id (str): Thread identifier used by the LangGraph checkpointer.
    - stream_mode (str): Streaming mode; default 'messages'.

    Return Value:
    - Generator[str, None, None]: Yields strings representing AI message content chunks.

    Side Effects:
    - None directly; depends on graph execution.

    Examples:
    >>> # def ai_only():
    >>> #     yield from stream_ai_tokens(chatbot, state, thread_id)
    >>> # text = st.write_stream(ai_only())
    """
    config = {"configurable": {"thread_id": thread_id}}
    for message_chunk, _metadata in chatbot.stream(
        state, config=config, stream_mode=stream_mode
    ):
        if isinstance(message_chunk, (AIMessage, AIMessageChunk)):
            # Yield only assistant tokens or message fragments
            yield message_chunk.content
