from openhands.events.event import Event
from openhands.llm.metrics import Metrics, TokenUsage


def get_token_usage_for_event(event: Event, metrics: Metrics) -> TokenUsage | None:
    """Returns at most one token usage record for either:
      - `tool_call_metadata.model_response.id`, if possible
      - otherwise event.response_id, if set

    If neither exist or none matches in metrics.token_usages, returns None.
    """
    # 1) Use the tool_call_metadata's response.id if present
    if event.tool_call_metadata and event.tool_call_metadata.model_response:
        tool_response_id = event.tool_call_metadata.model_response.get('id')
        if tool_response_id:
            usage_rec = next(
                (u for u in metrics.token_usages if u.response_id == tool_response_id),
                None,
            )
            if usage_rec is not None:
                return usage_rec

    # 2) Fallback to the top-level event.response_id if present
    if event.response_id:
        return next(
            (u for u in metrics.token_usages if u.response_id == event.response_id),
            None,
        )

    return None


def get_token_usage_for_event_id(
    events: list[Event], event_id: int, metrics: Metrics
) -> TokenUsage | None:
    """Starting from the event with .id == event_id and moving backwards in `events`,
    find the first TokenUsage record (if any) associated either with:
      - tool_call_metadata.model_response.id, or
      - event.response_id
    Returns the first match found, or None if none is found.
    """
    # Find the index of the event with the given id
    idx = None
    # Avoid enumerate, use a single pass index lookup for performance
    for i in range(len(events)):
        if events[i].id == event_id:
            idx = i
            break
    if idx is None:
        return None

    # Search backward from idx down to 0
    # Inline get_token_usage_for_event for this loop and avoid extra function calls for hot path
    if not hasattr(metrics, '_token_usage_by_id'):
        metrics._token_usage_by_id = {u.response_id: u for u in metrics.token_usages}
    token_usage_by_id = metrics._token_usage_by_id

    for i in range(idx, -1, -1):
        event = events[i]

        # 1) Use the tool_call_metadata's response.id if present
        usage_rec = None
        if event.tool_call_metadata and event.tool_call_metadata.model_response:
            tool_response_id = event.tool_call_metadata.model_response.get('id')
            if tool_response_id:
                usage_rec = token_usage_by_id.get(tool_response_id)
                if usage_rec is not None:
                    return usage_rec

        # 2) Fallback to the top-level event.response_id if present
        if event.response_id:
            usage_rec = token_usage_by_id.get(event.response_id)
            if usage_rec is not None:
                return usage_rec

    return None
