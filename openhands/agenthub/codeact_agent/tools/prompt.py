import re
import sys

_REFINE_PROMPT_REGEX = re.compile(
    r'\bexecute_bash\b|\bbash\b', re.IGNORECASE
)


def refine_prompt(prompt: str):
    """Refines the prompt based on the platform.

    On Windows systems, replaces 'bash' with 'powershell' and 'execute_bash' with 'execute_powershell'
    to ensure commands work correctly on the Windows platform.

    Args:
        prompt: The prompt text to refine

    Returns:
        The refined prompt text
    """
    if sys.platform == 'win32':
        # Run a single regex substitution with callback for both patterns
        return _REFINE_PROMPT_REGEX.sub(_refine_replacement, prompt)
    return prompt

def _refine_replacement(match):
    # 'execute_bash' becomes 'execute_powershell'
    text = match.group()
    if text.lower() == 'execute_bash':
        return 'execute_powershell'
    # standalone 'bash' should NOT match 'execute_bash'
    return 'powershell'
