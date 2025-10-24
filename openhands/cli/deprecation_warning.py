"""Deprecation warning utilities for the old OpenHands CLI."""

import sys

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import HTML


def display_deprecation_warning() -> None:
    """Display a prominent deprecation warning for the old CLI interface."""
    warning_lines = [
        '',
        '⚠️  DEPRECATION WARNING ⚠️',
        '',
        'This CLI interface is deprecated and will be removed in a future version.',
        'Please migrate to the new OpenHands CLI:',
        '',
        'For more information, visit: https://docs.all-hands.dev/usage/how-to/cli-mode',
        '',
        '=' * 70,
        '',
    ]

    style_map = {
        'DEPRECATION WARNING': '<ansired><b>{}</b></ansired>',
        'https://': '<ansiblue>{}</ansiblue>',
        '=': '<ansiyellow>{}</ansiyellow>',
        'default': '<ansiyellow>{}</ansiyellow>',
    }

    # Pre-create HTML objects for fixed lines to avoid recreating
    formatted_lines = []
    for line in warning_lines:
        if 'DEPRECATION WARNING' in line:
            formatted_lines.append(HTML(style_map['DEPRECATION WARNING'].format(line)))
        elif line.startswith('  •'):
            # There are no lines that match this, skip for now
            formatted_lines.append(HTML(f'<ansigreen>{line}</ansigreen>'))
        elif 'https://' in line:
            formatted_lines.append(HTML(style_map['https://'].format(line)))
        elif line.startswith('='):
            formatted_lines.append(HTML(style_map['='].format(line)))
        else:
            formatted_lines.append(HTML(style_map['default'].format(line)))

    for html_line in formatted_lines:
        print_formatted_text(html_line)

    sys.stdout.flush()
