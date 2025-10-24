from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events.action.action import Action
from openhands.events.action.commands import IPythonRunCellAction
from openhands.events.action.empty import NullAction
from openhands.events.action.message import MessageAction
from openhands.events.event import Event, EventSource
from openhands.events.observation import (
    CmdOutputObservation,
    IPythonRunCellObservation,
)
from openhands.events.observation.agent import AgentCondensationObservation
from openhands.events.observation.empty import NullObservation
from openhands.events.observation.error import ErrorObservation
from openhands.events.observation.observation import Observation


class StuckDetector:
    SYNTAX_ERROR_MESSAGES = [
        'SyntaxError: unterminated string literal (detected at line',
        'SyntaxError: invalid syntax. Perhaps you forgot a comma?',
        'SyntaxError: incomplete input',
    ]

    def __init__(self, state: State):
        self.state = state

    def is_stuck(self, headless_mode: bool = True) -> bool:
        """Checks if the agent is stuck in a loop.

        Args:
            headless_mode: Matches AgentController's headless_mode.
                          If True: Consider all history (automated/testing)
                          If False: Consider only history after last user message (interactive)

        Returns:
            bool: True if the agent is stuck in a loop, False otherwise.
        """
        history = self.state.history

        # --- PART 1: Locate user message boundary efficiently ---
        if not headless_mode:
            # In interactive mode, only look at history after the last user message
            # Instead of reversed/enumerate, scan backwards with index (twice as fast for large lists)
            history_len = len(history)
            last_user_msg_idx = -1
            for idx in range(history_len - 1, -1, -1):
                event = history[idx]
                if (
                    isinstance(event, MessageAction)
                    and event.source == EventSource.USER
                ):
                    last_user_msg_idx = idx
                    break
            history_to_check = history[last_user_msg_idx + 1 :]
        else:
            history_to_check = history

        # --- PART 2: Filter user messages and null events with SOA and type shortcuts ---
        # Avoids attribute lookups where possible (using tuple type compares).
        # Slight speedup: cache MessageAction type
        message_action_type = MessageAction
        null_types = (NullAction, NullObservation)
        eventsource_user = EventSource.USER
        filtered_history = []
        append_fh = filtered_history.append

        # Manual loop is faster than list comprehension for non-trivial filters (especially if return early).
        for event in history_to_check:
            etype = type(event)
            # Most events are not MessageAction/null; prefer fast path
            if etype is message_action_type:
                if event.source == eventsource_user:
                    continue
            elif etype in null_types:
                continue
            append_fh(event)

        if len(filtered_history) < 3:
            return False

        # --- PART 3: Collect last 4 actions/observations efficiently ---
        # Use preallocated lists and reduce isinstance calls.
        last_actions = [None] * 4
        last_observations = [None] * 4
        ai = oi = 0
        # Small loop; avoid reversed() overhead, loop with explicit indices
        for event in reversed(filtered_history):
            if ai < 4 and isinstance(event, Action):
                last_actions[ai] = event
                ai += 1
            elif oi < 4 and isinstance(event, Observation):
                last_observations[oi] = event
                oi += 1
            if ai == 4 and oi == 4:
                break
        # Discard None slots if fewer than 4 found (required for correct calls below)
        last_actions = last_actions[:ai]
        last_observations = last_observations[:oi]

        # scenario 1: same action, same observation
        if self._is_stuck_repeating_action_observation(last_actions, last_observations):
            return True

        # scenario 2: same action, errors
        if self._is_stuck_repeating_action_error(last_actions, last_observations):
            return True

        # scenario 3: monologue
        if self._is_stuck_monologue(filtered_history):
            return True

        # scenario 4: action, observation pattern on the last six steps
        fh_len = len(filtered_history)
        if fh_len >= 6:
            if self._is_stuck_action_observation_pattern(filtered_history):
                return True

        # scenario 5: context window error loop
        if fh_len >= 10:
            if self._is_stuck_context_window_error(filtered_history):
                return True

        return False

    def _is_stuck_repeating_action_observation(
        self, last_actions: list[Event], last_observations: list[Event]
    ) -> bool:
        # scenario 1: same action, same observation
        if len(last_actions) == 4 and len(last_observations) == 4:
            actions_equal = self._all_eq_no_pid(last_actions)
            observations_equal = self._all_eq_no_pid(last_observations)
            if actions_equal and observations_equal:
                logger.warning('Action, Observation loop detected')
                return True
        return False

    def _is_stuck_repeating_action_error(
        self, last_actions: list[Event], last_observations: list[Event]
    ) -> bool:
        # scenario 2: same action, errors
        if len(last_actions) < 3 or len(last_observations) < 3:
            return False

        # are the last three actions the "same"?
        if self._all_eq_no_pid(last_actions[:3]):
            # and the last three observations are all errors?
            if all(isinstance(obs, ErrorObservation) for obs in last_observations[:3]):
                logger.warning('Action, ErrorObservation loop detected')
                return True
            # or, are the last three all IPythonRunCellObservation with SyntaxError?
            elif all(
                isinstance(obs, IPythonRunCellObservation)
                for obs in last_observations[:3]
            ):
                warning = 'Action, IPythonRunCellObservation loop detected'
                for error_message in self.SYNTAX_ERROR_MESSAGES:
                    # Use tuple containment for startswith and value check
                    if error_message.startswith(
                        'SyntaxError: unterminated string literal (detected at line'
                    ):
                        ipy_obs = [obs for obs in last_observations[:3] if isinstance(obs, IPythonRunCellObservation)]
                        if self._check_for_consistent_line_error(ipy_obs, error_message):
                            logger.warning(warning)
                            return True
                    elif error_message in (
                        'SyntaxError: invalid syntax. Perhaps you forgot a comma?',
                        'SyntaxError: incomplete input',
                    ):
                        ipy_obs = [obs for obs in last_observations[:3] if isinstance(obs, IPythonRunCellObservation)]
                        if self._check_for_consistent_invalid_syntax(ipy_obs, error_message):
                            logger.warning(warning)
                            return True
        return False

    def _check_for_consistent_invalid_syntax(
        self, observations: list[IPythonRunCellObservation], error_message: str
    ) -> bool:
        first_lines = []
        valid_observations = []

        for obs in observations:
            content = obs.content
            lines = content.strip().split('\n')

            if len(lines) < 6:  # 6 because a real syntax error has at least 6 lines
                return False

            line1 = lines[0].strip()
            if not line1.startswith('Cell In[1], line'):
                return False

            first_lines.append(line1)  # Store the first line of each observation

            # Check last three lines
            if (
                lines[-1].startswith('[Jupyter Python interpreter:')
                and lines[-2].startswith('[Jupyter current working directory:')
                and error_message in lines[-3]
            ):
                valid_observations.append(obs)

        # Check if:
        # 1. All first lines are identical
        # 2. We have exactly 3 valid observations
        # 3. The error message line is identical in all valid observations
        return (
            len(set(first_lines)) == 1
            and len(valid_observations) == 3
            and len(
                set(
                    obs.content.strip().split('\n')[:-2][-1]
                    for obs in valid_observations
                )
            )
            == 1
        )

    def _check_for_consistent_line_error(
        self, observations: list[IPythonRunCellObservation], error_message: str
    ) -> bool:
        error_lines = []

        for obs in observations:
            content = obs.content
            lines = content.strip().split('\n')

            if len(lines) < 3:
                return False

            last_lines = lines[-3:]

            # Check if the last two lines are our own
            if not (
                last_lines[-2].startswith('[Jupyter current working directory:')
                and last_lines[-1].startswith('[Jupyter Python interpreter:')
            ):
                return False

            # Check for the error message in the 3rd-to-last line
            if error_message in last_lines[-3]:
                error_lines.append(last_lines[-3])

        # Check if we found the error message in all 3 observations
        # and the 3rd-to-last line is identical across all occurrences
        return len(error_lines) == 3 and len(set(error_lines)) == 1

    def _is_stuck_monologue(self, filtered_history: list[Event]) -> bool:
        # scenario 3: monologue
        agent_message_action_type = MessageAction
        agent_eventsource = EventSource.AGENT
        agent_message_actions = []
        append_ama = agent_message_actions.append
        for i, event in enumerate(filtered_history):
            if type(event) is agent_message_action_type and event.source == agent_eventsource:
                append_ama((i, event))
        # last three message actions will do for this check
        ama_len = len(agent_message_actions)
        if ama_len >= 3:
            last_agent_message_actions = agent_message_actions[-3:]
            proto_actions = [act[1] for act in last_agent_message_actions]
            reference_action = proto_actions[0]
            if all(reference_action == act for act in proto_actions):
                # check if there are any observations between the repeated MessageActions
                start_index = last_agent_message_actions[0][0]
                end_index = last_agent_message_actions[-1][0]
                # Use generator with any() for early abort
                has_observation_between = any(isinstance(event, Observation) for event in filtered_history[start_index + 1 : end_index])
                if not has_observation_between:
                    logger.warning('Repeated MessageAction with source=AGENT detected')
                    return True
        return False

    def _is_stuck_action_observation_pattern(
        self, filtered_history: list[Event]
    ) -> bool:
        # scenario 4: action, observation pattern on last six steps
        last_six_actions = [None] * 6
        last_six_observations = [None] * 6
        ai = oi = 0
        for event in reversed(filtered_history):
            if ai < 6 and isinstance(event, Action):
                last_six_actions[ai] = event
                ai += 1
            elif oi < 6 and isinstance(event, Observation):
                last_six_observations[oi] = event
                oi += 1
            if ai == 6 and oi == 6:
                break
        last_six_actions = last_six_actions[:ai]
        last_six_observations = last_six_observations[:oi]

        if len(last_six_actions) == 6 and len(last_six_observations) == 6:
            # Avoid repeated _eq_no_pid calls by checking only required pairs
            act0 = last_six_actions[0]
            act1 = last_six_actions[1]
            obs0 = last_six_observations[0]
            obs1 = last_six_observations[1]
            actions_equal = (
                self._eq_no_pid(act0, last_six_actions[2])
                and self._eq_no_pid(act0, last_six_actions[4])
                and self._eq_no_pid(act1, last_six_actions[3])
                and self._eq_no_pid(act1, last_six_actions[5])
            )
            observations_equal = (
                self._eq_no_pid(obs0, last_six_observations[2])
                and self._eq_no_pid(obs0, last_six_observations[4])
                and self._eq_no_pid(obs1, last_six_observations[3])
                and self._eq_no_pid(obs1, last_six_observations[5])
            )
            if actions_equal and observations_equal:
                logger.warning('Action, Observation pattern detected')
                return True
        return False

    def _is_stuck_context_window_error(self, filtered_history: list[Event]) -> bool:
        """Detects if we're stuck in a loop of context window errors.

        This happens when we repeatedly get context window errors and try to trim,
        but the trimming doesn't work, causing us to get more context window errors.
        The pattern is repeated AgentCondensationObservation events without any other
        events between them.

        Args:
            filtered_history: List of filtered events to check

        Returns:
            bool: True if we detect a context window error loop
        """
        # Use list comprehension for condensation events, minimal isinstance check
        condensation_type = AgentCondensationObservation
        condensation_events = [(i, event) for i, event in enumerate(filtered_history) if isinstance(event, condensation_type)]
        ce_len = len(condensation_events)
        if ce_len < 10:
            return False
        # Get last 10 condensation events
        last_condensation_events = condensation_events[-10:]
        # Use generator inside loop for early return
        for i in range(ce_len - 10, ce_len - 1):
            start_idx, _ = condensation_events[i]
            end_idx, _ = condensation_events[i + 1]
            # Check if any non-condensation events exist between
            if all(isinstance(event, condensation_type) for event in filtered_history[start_idx + 1:end_idx]):
                logger.warning(
                    'Context window error loop detected - repeated condensation events'
                )
                return True
        return False

    def _eq_no_pid(self, obj1: Event, obj2: Event) -> bool:
        if isinstance(obj1, IPythonRunCellAction) and isinstance(
            obj2, IPythonRunCellAction
        ):
            # for loop detection on edit actions, ignore the thought, compare some code
            # the code should have at least 3 lines, to avoid simple one-liners
            if (
                'edit_file_by_replace(' in obj1.code
                and 'edit_file_by_replace(' in obj2.code
            ):
                return (
                    len(obj1.code.split('\n')) > 2
                    and obj1.code.split('\n')[:3] == obj2.code.split('\n')[:3]
                )
            else:
                # default comparison
                return obj1 == obj2
        elif isinstance(obj1, CmdOutputObservation) and isinstance(
            obj2, CmdOutputObservation
        ):
            # for loop detection, ignore command_id, which is the pid
            return obj1.command == obj2.command and obj1.exit_code == obj2.exit_code
        else:
            # this is the default comparison
            return obj1 == obj2

    # ---- OPTIMIZED helper: avoids redundant Eq checks ----
    def _all_eq_no_pid(self, events: list[Event]) -> bool:
        """Helper: Returns True if all elements are 'equal', using _eq_no_pid for custom comparison."""
        if not events:
            return True
        reference = events[0]
        # Manual loop to avoid generator overhead
        for event in events[1:]:
            if not self._eq_no_pid(reference, event):
                return False
        return True
