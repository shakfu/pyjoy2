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

from .repl import HybridREPL, run
from .core import Stack


def main():
    parser = argparse.ArgumentParser(
        prog='pyjoy2',
        description='PyJoy2 - A Pythonic Joy implementation'
    )
    parser.add_argument(
        'file',
        nargs='?',
        help='Joy source file to execute'
    )
    parser.add_argument(
        '-e', '--eval',
        metavar='EXPR',
        help='Evaluate Joy expression'
    )
    parser.add_argument(
        '-c', '--command',
        metavar='CMD',
        help='Execute Joy program (alias for -e)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (show tracebacks)'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )

    args = parser.parse_args()

    if args.debug:
        sys.argv.append('--debug')

    # Evaluate expression
    if args.eval or args.command:
        expr = args.eval or args.command
        try:
            stack = run(expr)
            if stack:
                for item in stack:
                    print(repr(item))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if args.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        return

    # Run file
    if args.file:
        try:
            with open(args.file) as f:
                source = f.read()
            stack = run(source)
            if stack:
                for item in stack:
                    print(repr(item))
        except FileNotFoundError:
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if args.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        return

    # Start REPL
    repl_instance = HybridREPL()
    repl_instance.run()


if __name__ == '__main__':
    main()
