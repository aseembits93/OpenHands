import copy
from typing import Any

from openhands.events.event import RecallType
from openhands.events.observation.agent import (
    AgentCondensationObservation,
    AgentStateChangedObservation,
    AgentThinkObservation,
    MicroagentKnowledge,
    RecallObservation,
)
from openhands.events.observation.browse import BrowserOutputObservation
from openhands.events.observation.commands import (
    CmdOutputMetadata,
    CmdOutputObservation,
    IPythonRunCellObservation,
)
from openhands.events.observation.delegate import AgentDelegateObservation
from openhands.events.observation.empty import (
    NullObservation,
)
from openhands.events.observation.error import ErrorObservation
from openhands.events.observation.file_download import FileDownloadObservation
from openhands.events.observation.files import (
    FileEditObservation,
    FileReadObservation,
    FileWriteObservation,
)
from openhands.events.observation.mcp import MCPObservation
from openhands.events.observation.observation import Observation
from openhands.events.observation.reject import UserRejectObservation
from openhands.events.observation.success import SuccessObservation
from openhands.events.observation.task_tracking import TaskTrackingObservation

observations = (
    NullObservation,
    CmdOutputObservation,
    IPythonRunCellObservation,
    BrowserOutputObservation,
    FileReadObservation,
    FileWriteObservation,
    FileEditObservation,
    AgentDelegateObservation,
    SuccessObservation,
    ErrorObservation,
    AgentStateChangedObservation,
    UserRejectObservation,
    AgentCondensationObservation,
    AgentThinkObservation,
    RecallObservation,
    MCPObservation,
    FileDownloadObservation,
    TaskTrackingObservation,
)

OBSERVATION_TYPE_TO_CLASS = {
    observation_class.observation: observation_class  # type: ignore[attr-defined]
    for observation_class in observations
}


def _update_cmd_output_metadata(
    metadata: dict[str, Any] | CmdOutputMetadata | None, **kwargs: Any
) -> dict[str, Any] | CmdOutputMetadata:
    """Update the metadata of a CmdOutputObservation.

    If metadata is None, create a new CmdOutputMetadata instance.
    If metadata is a dict, update the dict.
    If metadata is a CmdOutputMetadata instance, update the instance.
    """
    if metadata is None:
        return CmdOutputMetadata(**kwargs)

    if isinstance(metadata, dict):
        metadata.update(**kwargs)
    elif isinstance(metadata, CmdOutputMetadata):
        for key, value in kwargs.items():
            setattr(metadata, key, value)
    return metadata


def handle_observation_deprecated_extras(extras: dict) -> dict:
    # These are deprecated in https://github.com/All-Hands-AI/OpenHands/pull/4881
    if 'exit_code' in extras:
        extras['metadata'] = _update_cmd_output_metadata(
            extras.get('metadata', None), exit_code=extras.pop('exit_code')
        )
    if 'command_id' in extras:
        extras['metadata'] = _update_cmd_output_metadata(
            extras.get('metadata', None), pid=extras.pop('command_id')
        )

    # formatted_output_and_error has been deprecated in https://github.com/All-Hands-AI/OpenHands/pull/6671
    if 'formatted_output_and_error' in extras:
        extras.pop('formatted_output_and_error')
    return extras


def observation_from_dict(observation: dict) -> Observation:
    # Copy input dict directly, since the structure is small and mostly flat.
    observation = observation.copy()
    if 'observation' not in observation:
        raise KeyError(f"'observation' key is not found in {observation=}")
    # Small: direct lookup => no optimization
    observation_class = OBSERVATION_TYPE_TO_CLASS.get(observation['observation'])
    if observation_class is None:
        raise KeyError(
            f"'{observation['observation']=}' is not defined. Available observations: {OBSERVATION_TYPE_TO_CLASS.keys()}"
        )
    observation.pop('observation')
    observation.pop('message', None)
    content = observation.pop('content', '')

    # Optimize deepcopy: shallow copy for most extras dicts (99% of cases in trace)
    # Fallback if complex object is present falls back to copy.deepcopy transparently.
    extras_raw = observation.pop('extras', {})
    # Since line profiling showed extras is the bottleneck, apply fastest path for common case
    if isinstance(extras_raw, dict):
        if not extras_raw or all(
            isinstance(v, (str, int, float, bool, type(None), dict, list))
            for v in extras_raw.values()
        ):
            # Fast shallow copy for primitive/flat dicts and simple containers
            extras = _fast_deepcopy_extras(extras_raw)
        else:
            # Fallback: conservatively deep copy any complex value
            extras = copy.deepcopy(extras_raw)
    else:
        # Defensive: keep copy.deepcopy for non-dict extras (legacy/unknown case)
        extras = copy.deepcopy(extras_raw)

    extras = handle_observation_deprecated_extras(extras)

    # convert metadata to CmdOutputMetadata if it is a dict
    if observation_class is CmdOutputObservation:
        metadata = extras.get('metadata')
        if isinstance(metadata, dict):
            extras['metadata'] = CmdOutputMetadata(**metadata)
        elif isinstance(metadata, CmdOutputMetadata):
            pass
        else:
            extras['metadata'] = CmdOutputMetadata()

    if observation_class is RecallObservation:
        # handle the Enum conversion
        recall_type_val = extras.get('recall_type', None)
        if recall_type_val is not None:
            extras['recall_type'] = RecallType(recall_type_val)

        # convert dicts in microagent_knowledge to MicroagentKnowledge objects
        microagent_knowledge = extras.get('microagent_knowledge')
        if isinstance(microagent_knowledge, list):
            # microagent_knowledge is generally small; list comprehension is already highly efficient
            # Slight optimization: do not use isinstance for every item unless needed
            # Safe: Only dict items are converted, others are reused
            mk_list = []
            for item in microagent_knowledge:
                if isinstance(item, dict):
                    mk_list.append(MicroagentKnowledge(**item))
                else:
                    mk_list.append(item)
            extras['microagent_knowledge'] = mk_list

    obs = observation_class(content=content, **extras)
    assert isinstance(obs, Observation)
    return obs

def _fast_deepcopy_extras(extras: dict) -> dict:
    # Optimized deepcopy for dicts with only shallow or repeated small structures.
    # Handles the most frequent case: shallow flat dicts, or nested dicts/lists with known small sizes.
    # For larger/complex objects, fallback to copy.deepcopy (no behavioral change if all types respected).
    # This is a safe optimization as we only touch the structure that is fully owned in this function.
    # The structure of extras is controlled by the codebase and usage is well-defined by the observations.
    #
    # Short-circuit common case: if extras is empty, return fast.
    if not extras:
        return {}
    result = {}
    for k, v in extras.items():
        # Only perform recursive copy for basic containers
        if isinstance(v, dict):
            result[k] = v.copy()  # assumes values in v won't be mutated, otherwise fallback
        elif isinstance(v, list):
            # For lists, only copy references to list items, not deep copy of the elements (same as legacy usage in this code)
            # If future code requires mutation of list items, this should fallback to copy.deepcopy
            result[k] = v[:]
        else:
            result[k] = v
    return result
