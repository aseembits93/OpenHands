"""Fast help module for OpenHands CLI.

This module provides a lightweight implementation of the CLI help and version commands
without loading all the dependencies, which significantly improves the
performance of `openhands --help` and `openhands --version`.

The approach is to create a simplified version of the CLI parser that only includes
the necessary options for displaying help and version information. This avoids loading
the full OpenHands codebase, which can take several seconds.

This implementation addresses GitHub issue #10698, which reported that
`openhands --help` was taking around 20 seconds to run.
"""

import argparse
import sys

from openhands.cli.deprecation_warning import display_deprecation_warning


def get_fast_cli_parser() -> argparse.ArgumentParser:
    """Create a lightweight argument parser for CLI help command."""
    description = (
        'Welcome to OpenHands: Code Less, Make More\n\n'
        'OpenHands supports two main commands:\n'
        '  serve - Launch the OpenHands GUI server (web interface)\n'
        '  cli   - Run OpenHands in CLI mode (terminal interface)\n\n'
        'Running "openhands" without a command is the same as "openhands cli"'
    )

    parser = argparse.ArgumentParser(
        description=description,
        prog='openhands',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='For more information about a command, run: openhands COMMAND --help',
    )
    subparsers = parser.add_subparsers(
        dest='command',
        title='commands',
        description='OpenHands supports two main commands:',
        metavar='COMMAND',
    )

    # serve subcommand
    serve_parser = subparsers.add_parser(
        'serve', help='Launch the OpenHands GUI server using Docker (web interface)'
    )
    serve_parser.add_argument(
        '--mount-cwd',
        help='Mount the current working directory into the GUI server container',
        action='store_true',
        default=False,
    )
    serve_parser.add_argument(
        '--gpu',
        help='Enable GPU support by mounting all GPUs into the Docker container via nvidia-docker',
        action='store_true',
        default=False,
    )

    # cli subcommand, bulk add arguments for efficiency
    cli_parser = subparsers.add_parser(
        'cli', help='Run OpenHands in CLI mode (terminal interface)'
    )
    cli_arguments = [
        {
            "args": ('--config-file',),
            "kwargs": {
                "type": str,
                "default": 'config.toml',
                "help": 'Path to the config file (default: config.toml in the current directory)'
            }
        },
        {
            "args": ('-t', '--task'),
            "kwargs": {
                "type": str,
                "default": '',
                "help": 'The task for the agent to perform'
            }
        },
        {
            "args": ('-f', '--file'),
            "kwargs": {
                "type": str,
                "help": 'Path to a file containing the task. Overrides -t if both are provided.'
            }
        },
        {
            "args": ('-n', '--name'),
            "kwargs": {
                "help": 'Session name',
                "type": str,
                "default": '',
            }
        },
        {
            "args": ('--log-level',),
            "kwargs": {
                "help": 'Set the log level',
                "type": str,
                "default": None,
            }
        },
        {
            "args": ('-l', '--llm-config'),
            "kwargs": {
                "default": None,
                "type": str,
                "help": 'Replace default LLM ([llm] section in config.toml) config with the specified LLM config, e.g. "llama3" for [llm.llama3] section in config.toml'
            }
        },
        {
            "args": ('--agent-config',),
            "kwargs": {
                "default": None,
                "type": str,
                "help": 'Replace default Agent ([agent] section in config.toml) config with the specified Agent config, e.g. "CodeAct" for [agent.CodeAct] section in config.toml'
            }
        },
        {
            "args": ('-v', '--version'),
            "kwargs": {
                "action": 'store_true',
                "help": 'Show version information'
            }
        },
        {
            "args": ('--override-cli-mode',),
            "kwargs": {
                "help": 'Override the default settings for CLI mode',
                "type": bool,
                "default": False,
            }
        },
    ]

    for arg_spec in cli_arguments:
        cli_parser.add_argument(*arg_spec["args"], **arg_spec["kwargs"])

    parser.add_argument(
        '--conversation',
        help='The conversation id to continue',
        type=str,
        default=None,
    )
    return parser


def get_fast_subparser(
    parser: argparse.ArgumentParser, name: str
) -> argparse.ArgumentParser:
    """Get a subparser by name."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            if name in action.choices:
                return action.choices[name]
    raise ValueError(f"Subparser '{name}' not found")


def handle_fast_commands() -> bool:
    """Handle fast path commands like help and version.

    Returns:
        bool: True if a command was handled, False otherwise.
    """
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg in ('--help', '-h'):
            display_deprecation_warning()
            parser = get_fast_cli_parser()
            # Print top-level help
            print(parser.format_help())
            # Also print help for `cli` subcommand
            print('\n' + '=' * 80)
            print('CLI command help:\n')
            cli_parser = get_fast_subparser(parser, 'cli')
            print(cli_parser.format_help())
            return True

        if arg in ('--version', '-v'):
            from openhands import get_version
            print(f'OpenHands CLI version: {get_version()}')
            display_deprecation_warning()
            return True

    return False
