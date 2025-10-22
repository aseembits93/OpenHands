from __future__ import annotations

from openhands.core.config.condenser_config import AmortizedForgettingCondenserConfig
from openhands.events.action.agent import CondensationAction
from openhands.llm.llm_registry import LLMRegistry
from openhands.memory.condenser.condenser import (
    Condensation,
    RollingCondenser,
    View,
)


class AmortizedForgettingCondenser(RollingCondenser):
    """A condenser that maintains a condensed history and forgets old events when it grows too large."""

    def __init__(self, max_size: int = 100, keep_first: int = 0):
        """Initialize the condenser.

        Args:
            max_size: Maximum size of history before forgetting.
            keep_first: Number of initial events to always keep.

        Raises:
            ValueError: If keep_first is greater than max_size, keep_first is negative, or max_size is non-positive.
        """
        if keep_first >= max_size // 2:
            raise ValueError(
                f'keep_first ({keep_first}) must be less than half of max_size ({max_size})'
            )
        if keep_first < 0:
            raise ValueError(f'keep_first ({keep_first}) cannot be negative')
        if max_size < 1:
            raise ValueError(f'max_size ({keep_first}) cannot be non-positive')

        self.max_size = max_size
        self.keep_first = keep_first

        super().__init__()

    def get_condensation(self, view: View) -> Condensation:
        target_size = self.max_size // 2
        head = view[: self.keep_first]
        head_len = len(head)
        events_from_tail = target_size - head_len
        tail = view[-events_from_tail:]

        # Optimize id lookup with set comprehensions and avoid materializing head+tail
        event_ids_to_keep = set()
        for event in head:
            event_ids_to_keep.add(event.id)
        for event in tail:
            event_ids_to_keep.add(event.id)

        # Instead of creating a full set of all ids and subtracting,
        # accumulate min and max of 'id's to forget directly
        min_id_to_forget = None
        max_id_to_forget = None
        for event in view:
            eid = event.id
            if eid not in event_ids_to_keep:
                if min_id_to_forget is None or eid < min_id_to_forget:
                    min_id_to_forget = eid
                if max_id_to_forget is None or eid > max_id_to_forget:
                    max_id_to_forget = eid

        # For behavioral preservation: if nothing to forget, mimics set min/max exception
        if min_id_to_forget is None or max_id_to_forget is None:
            raise ValueError("No events to forget in condensation.")

        event = CondensationAction(
            forgotten_events_start_id=min_id_to_forget,
            forgotten_events_end_id=max_id_to_forget,
        )

        return Condensation(action=event)

    def should_condense(self, view: View) -> bool:
        return len(view) > self.max_size

    @classmethod
    def from_config(
        cls,
        config: AmortizedForgettingCondenserConfig,
        llm_registry: LLMRegistry,
    ) -> AmortizedForgettingCondenser:
        return AmortizedForgettingCondenser(**config.model_dump(exclude={'type'}))


AmortizedForgettingCondenser.register_config(AmortizedForgettingCondenserConfig)
