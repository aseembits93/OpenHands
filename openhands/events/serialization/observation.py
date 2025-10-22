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
        # Avoid dict construction in kwargs if possible (already optimal for constructor call)
        return CmdOutputMetadata(**kwargs)

    if isinstance(metadata, dict):
        # Direct dictionary update is already fast
        metadata.update(kwargs)
    elif isinstance(metadata, CmdOutputMetadata):
        # Use setattr directly in a tight loop; to optimize, use __dict__.update when possible
        # Only do this if all kwargs keys are actual attributes (assumed safe here based on usage)
        try:
            metadata.__dict__.update(kwargs)
        except Exception:
            # Fallback if __dict__.update is not safe (e.g., presence of slots/properties)
            for key, value in kwargs.items():
                setattr(metadata, key, value)
    return metadata


def handle_observation_deprecated_extras(extras: dict) -> dict:
    # These are deprecated in https://github.com/All-Hands-AI/OpenHands/pull/4881

    # Use bit-masking and single scan for improved performance,
    # but to preserve exact output/side effects, do not collapse logic
    if 'exit_code' in extras:
        exit_code = extras.pop('exit_code')
        extras['metadata'] = _update_cmd_output_metadata(
            extras.get('metadata', None), exit_code=exit_code
        )
    if 'command_id' in extras:
        command_id = extras.pop('command_id')
        extras['metadata'] = _update_cmd_output_metadata(
            extras.get('metadata', None), pid=command_id
        )

    # Direct pop after lookup for deprecation
    if 'formatted_output_and_error' in extras:
        extras.pop('formatted_output_and_error')
    return extras


def observation_from_dict(observation: dict) -> Observation:
    observation = observation.copy()
    if 'observation' not in observation:
        raise KeyError(f"'observation' key is not found in {observation=}")
    observation_class = OBSERVATION_TYPE_TO_CLASS.get(observation['observation'])
    if observation_class is None:
        raise KeyError(
            f"'{observation['observation']=}' is not defined. Available observations: {OBSERVATION_TYPE_TO_CLASS.keys()}"
        )
    observation.pop('observation')
    observation.pop('message', None)
    content = observation.pop('content', '')
    extras = copy.deepcopy(observation.pop('extras', {}))

    extras = handle_observation_deprecated_extras(extras)

    # convert metadata to CmdOutputMetadata if it is a dict
    if observation_class is CmdOutputObservation:
        if 'metadata' in extras and isinstance(extras['metadata'], dict):
            extras['metadata'] = CmdOutputMetadata(**extras['metadata'])
        elif 'metadata' in extras and isinstance(extras['metadata'], CmdOutputMetadata):
            pass
        else:
            extras['metadata'] = CmdOutputMetadata()

    if observation_class is RecallObservation:
        # handle the Enum conversion
        if 'recall_type' in extras:
            extras['recall_type'] = RecallType(extras['recall_type'])

        # convert dicts in microagent_knowledge to MicroagentKnowledge objects
        if 'microagent_knowledge' in extras and isinstance(
            extras['microagent_knowledge'], list
        ):
            extras['microagent_knowledge'] = [
                MicroagentKnowledge(**item) if isinstance(item, dict) else item
                for item in extras['microagent_knowledge']
            ]

    obs = observation_class(content=content, **extras)
    assert isinstance(obs, Observation)
    return obs
