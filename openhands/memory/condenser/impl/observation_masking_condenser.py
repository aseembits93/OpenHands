from __future__ import annotations

from openhands.core.config.condenser_config import ObservationMaskingCondenserConfig
from openhands.events.event import Event
from openhands.events.observation import Observation
from openhands.events.observation.agent import AgentCondensationObservation
from openhands.llm.llm_registry import LLMRegistry
from openhands.memory.condenser.condenser import Condensation, Condenser, View


class ObservationMaskingCondenser(Condenser):
    """A condenser that masks the values of observations outside of a recent attention window."""

    def __init__(self, attention_window: int = 5):
        self.attention_window = attention_window

        super().__init__()

    def condense(self, view: View) -> View | Condensation:
        """Replace the content of observations outside of the attention window with a placeholder."""
        # Compute cutoff once for faster access
        cutoff = max(len(view) - self.attention_window, 0)
        results: list[Event] = [AgentCondensationObservation('<MASKED>') if isinstance(event, Observation) and i < cutoff else event
                                for i, event in enumerate(view)]
        return View(events=results)

    @classmethod
    def from_config(
        cls,
        config: ObservationMaskingCondenserConfig,
        llm_registry: LLMRegistry,
    ) -> ObservationMaskingCondenser:
        return ObservationMaskingCondenser(**config.model_dump(exclude={'type'}))


ObservationMaskingCondenser.register_config(ObservationMaskingCondenserConfig)
