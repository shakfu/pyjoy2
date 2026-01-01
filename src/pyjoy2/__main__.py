"""
pyjoy2 CLI entry point.

Usage:
    python -m pyjoy2              # Start REPL
    python -m pyjoy2 file.joy     # Run file
    python -m pyjoy2 -e "expr"    # Evaluate expression
    python -m pyjoy2 --help       # Show help
"""

import argparse
import sys
import traceback

from .repl import HybridREPL, run


def _print_stack(stack):
    """Print stack contents."""
    for item in stack:
        print(repr(item))


def _handle_error(e: Exception, debug: bool) -> None:
    """Print error and optionally show traceback."""
    print(f"Error: {e}", file=sys.stderr)
    if debug:
        traceback.print_exc()
    sys.exit(1)


def _run_expression(expr: str, debug: bool) -> None:
    """Evaluate a Joy expression."""
    try:
        stack = run(expr)
        if stack:
            _print_stack(stack)
    except Exception as e:
        _handle_error(e, debug)


def _run_file(filename: str, debug: bool) -> None:
    """Execute a Joy source file."""
    try:
        with open(filename) as f:
            source = f.read()
        stack = run(source)
        if stack:
            _print_stack(stack)
    except FileNotFoundError:
        print(f"Error: file not found: {filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        _handle_error(e, debug)


def main():
    parser = argparse.ArgumentParser(
        prog="pyjoy2", description="PyJoy2 - A Pythonic Joy implementation"
    )
    parser.add_argument("file", nargs="?", help="Joy source file to execute")
    parser.add_argument("-e", "--eval", metavar="EXPR", help="Evaluate Joy expression")
    parser.add_argument(
        "-c", "--command", metavar="CMD", help="Execute Joy program (alias for -e)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (show tracebacks)"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    args = parser.parse_args()

    if args.debug:
        sys.argv.append("--debug")

    if args.eval or args.command:
        _run_expression(args.eval or args.command, args.debug)
    elif args.file:
        _run_file(args.file, args.debug)
    else:
        HybridREPL().run()


if __name__ == "__main__":
    main()
