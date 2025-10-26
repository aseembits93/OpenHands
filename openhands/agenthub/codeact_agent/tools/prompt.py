import re
import sys

_RE_EXECUTE_BASH = re.compile(r'\bexecute_bash\b', flags=re.IGNORECASE)

_RE_BASH = re.compile(r'\bbash\b', flags=re.IGNORECASE)


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
        # Replace 'bash' with 'powershell' including tool names like 'execute_bash'
        # First replace 'execute_bash' with 'execute_powershell' to handle tool names
        result = _RE_EXECUTE_BASH.sub('execute_powershell', prompt)
        # Then replace standalone 'bash' with 'powershell'
        result = _RE_BASH.sub('powershell', result)
        return result
    return prompt
