"""
pyjoy2.repl - Hybrid REPL with Python integration.

Supports:
- Joy words and quotations
- Python expressions via `expr` or $(expr)
- Python statements via !stmt
- Multi-line function definitions
- Commands: .s, .c, .w, .help, .def, .import, .load
"""

from __future__ import annotations
from typing import Dict, Any, List
import re
import sys
import traceback

from .core import Stack, WORDS, word, define, execute
from .parser import parse

__all__ = ["HybridREPL", "repl", "run"]


class HybridREPL:
    """
    Hybrid Joy/Python REPL.

    Syntax:
    - Joy words: executed as usual
    - `expr`: backtick evaluates Python expression, pushes result
    - $(expr): alternative Python expression syntax
    - !statement: execute Python statement (no push)
    - def/class: multi-line Python definitions
    - .s: show stack
    - .c: clear stack
    - .w: list words
    - .def name [...]: define new word
    - .import module: import Python module
    """

    def __init__(self, stack: Stack | None = None):
        self.stack = stack if stack is not None else Stack()
        self.pending_lines: List[str] = []

        # Python namespace with useful bindings
        self.globals: Dict[str, Any] = {
            "stack": self.stack,
            "S": self.stack,
            "word": word,
            "define": define,
            "WORDS": WORDS,
            "Stack": Stack,
            "execute": execute,
            "__builtins__": __builtins__,
        }
        self.locals: Dict[str, Any] = {}

        # Pre-import common modules
        self._exec("import math")
        self._exec("import json")
        self._exec("import os")
        self._exec("import re as re_module")

    def _exec(self, code: str) -> None:
        """Execute Python statement."""
        exec(code, self.globals, self.locals)
        self.globals.update(self.locals)

    def _eval(self, expr: str) -> Any:
        """Evaluate Python expression and return result."""
        return eval(expr, self.globals, self.locals)

    def _is_incomplete(self, code: str) -> bool:
        """Check if Python code is incomplete (needs more lines)."""
        try:
            compile(code, "<input>", "exec")
            return False
        except SyntaxError as e:
            msg = str(e)
            return "unexpected EOF" in msg or "EOF while scanning" in msg

    def _handle_python_block(self, line: str) -> bool:
        """
        Handle multi-line Python blocks (def, class, etc.)
        Returns True if line was handled as Python block.
        """
        block_starters = (
            "def ",
            "class ",
            "if ",
            "for ",
            "while ",
            "with ",
            "try:",
            "async ",
            "@",
        )
        is_block_start = any(line.lstrip().startswith(s) for s in block_starters)

        if self.pending_lines or is_block_start:
            self.pending_lines.append(line)
            code = "\n".join(self.pending_lines)

            if self._is_incomplete(code):
                return True  # Need more lines

            # Execute complete block
            self._exec(code)
            self.pending_lines = []
            self._register_new_words()
            return True

        return False

    def _register_new_words(self) -> None:
        """Check for newly defined functions with decorators."""
        for name, obj in list(self.locals.items()):
            if callable(obj) and hasattr(obj, "__name__"):
                if hasattr(obj, "joy_word"):
                    print(f"  Registered word: {obj.joy_word}")
        self.globals.update(self.locals)

    def process_line(self, line: str) -> None:
        """Process a single line of input."""
        stripped = line.strip()
        if not stripped:
            return

        # Handle REPL commands
        if stripped == ".s":
            self._show_stack()
            return
        if stripped == ".c":
            self.stack.clear()
            print("  Stack cleared")
            return
        if stripped == ".w":
            self._show_words()
            return
        if stripped.startswith(".def "):
            self._define_word(stripped[5:])
            return
        if stripped.startswith(".import "):
            mod = stripped[8:].strip()
            self._exec(f"import {mod}")
            print(f"  Imported {mod}")
            return
        if stripped.startswith(".load "):
            self._load_file(stripped[6:].strip())
            return
        if stripped.startswith(".help "):
            self._show_word_help(stripped[6:].strip())
            return
        if stripped == ".help":
            self._show_help()
            return

        # Parse and execute mixed Joy/Python tokens
        self._execute_line(stripped)

    def _execute_line(self, line: str) -> None:
        """Parse and execute mixed Joy/Python line."""
        tokens = self._tokenize_hybrid(line)

        for token in tokens:
            if token["type"] == "python_expr":
                result = self._eval(token["value"])
                self.stack.push(result)

            elif token["type"] == "python_stmt":
                self._exec(token["value"])

            elif token["type"] == "word":
                name = token["value"]
                if name in WORDS:
                    WORDS[name](self.stack)
                elif name in self.globals:
                    self.stack.push(self.globals[name])
                elif name in self.locals:
                    self.stack.push(self.locals[name])
                else:
                    raise NameError(f"Unknown word: {name}")

            elif token["type"] == "quotation":
                inner = self._parse_quotation(token["value"])
                self.stack.push(inner)

            elif token["type"] == "literal":
                self.stack.push(token["value"])

    def _find_balanced(self, line: str, start: int, open_ch: str, close_ch: str) -> int:
        """Find end of balanced delimiters, return position after closing."""
        depth = 1
        j = start
        while depth > 0 and j < len(line):
            if line[j] == open_ch:
                depth += 1
            elif line[j] == close_ch:
                depth -= 1
            j += 1
        return j

    def _tokenize_hybrid(self, line: str) -> List[Dict[str, Any]]:
        """
        Tokenize line into Joy words and Python expressions.

        Patterns:
          `...`     -> Python expression
          $(...)    -> Python expression (alternative)
          !...      -> Python statement
          [...]     -> Quotation
          "..."     -> String
          123, 3.14 -> Numbers
          word      -> Joy word
        """
        tokens = []
        i = 0

        while i < len(line):
            # Skip whitespace
            while i < len(line) and line[i].isspace():
                i += 1
            if i >= len(line):
                break

            # Backtick Python expression: `expr`
            if line[i] == "`":
                j = line.index("`", i + 1)
                tokens.append({"type": "python_expr", "value": line[i + 1 : j]})
                i = j + 1

            # Dollar Python expression: $(expr)
            elif line[i : i + 2] == "$(":
                j = self._find_balanced(line, i + 2, "(", ")")
                tokens.append({"type": "python_expr", "value": line[i + 2 : j - 1]})
                i = j

            # Python statement: !statement
            elif line[i] == "!":
                j = i + 1
                while j < len(line) and line[j] not in "[]`":
                    j += 1
                tokens.append({"type": "python_stmt", "value": line[i + 1 : j].strip()})
                i = j

            # Quotation: [...]
            elif line[i] == "[":
                j = self._find_balanced(line, i + 1, "[", "]")
                tokens.append({"type": "quotation", "value": line[i + 1 : j - 1]})
                i = j

            # String: "..."
            elif line[i] == '"':
                j = i + 1
                while j < len(line) and line[j] != '"':
                    if line[j] == "\\":
                        j += 1
                    j += 1
                tokens.append({"type": "literal", "value": line[i + 1 : j]})
                i = j + 1

            # Number or word
            else:
                j = i
                while j < len(line) and not line[j].isspace() and line[j] not in '[]`"':
                    j += 1
                token_str = line[i:j]

                # Try to parse as number
                try:
                    if "." in token_str or "e" in token_str.lower():
                        tokens.append({"type": "literal", "value": float(token_str)})
                    else:
                        tokens.append({"type": "literal", "value": int(token_str)})
                except ValueError:
                    # Handle booleans
                    if token_str == "true":
                        tokens.append({"type": "literal", "value": True})
                    elif token_str == "false":
                        tokens.append({"type": "literal", "value": False})
                    else:
                        tokens.append({"type": "word", "value": token_str})

                i = j

        return tokens

    def _parse_quotation(self, inner: str) -> List[Any]:
        """Parse quotation content into executable list."""
        if not inner.strip():
            return []

        tokens = self._tokenize_hybrid(inner)
        result = []

        for token in tokens:
            if token["type"] == "word":
                name = token["value"]
                if name in WORDS:
                    result.append(WORDS[name])
                else:
                    # Keep as string for late binding
                    result.append(name)
            elif token["type"] == "quotation":
                result.append(self._parse_quotation(token["value"]))
            elif token["type"] == "python_expr":
                # Create a closure to evaluate later
                expr = token["value"]

                def make_eval_fn(e):
                    return lambda s: s.push(self._eval(e))

                result.append(make_eval_fn(expr))
            else:
                result.append(token["value"])

        return result

    def _show_stack(self) -> None:
        """Display stack with types."""
        if not self.stack:
            print("  (empty)")
            return
        print("  Stack (bottom to top):")
        for i, item in enumerate(self.stack):
            type_name = type(item).__name__
            repr_str = repr(item)
            if len(repr_str) > 60:
                repr_str = repr_str[:57] + "..."
            print(f"    {i}: ({type_name}) {repr_str}")

    def _show_words(self) -> None:
        """List available words."""
        words = sorted(WORDS.keys())
        # Group by first character
        print(f"  {len(words)} words available")
        line = "  "
        for w in words:
            if len(line) + len(w) + 2 > 78:
                print(line)
                line = "  "
            line += w + " "
        if line.strip():
            print(line)

    def _define_word(self, defn: str) -> None:
        """Define new word: .def name [body]

        Supports recursive definitions by registering a forward reference
        before parsing the body.
        """
        match = re.match(r"(\w[\w\-]*)\s+\[(.+)\]", defn)
        if not match:
            print("  Usage: .def name [body]")
            return
        name, body = match.groups()

        # For recursive definitions: register forward reference first
        # Use a mutable container to allow updating after parse
        impl_holder: List[Any] = [[]]

        def forward_ref(s: Stack, holder=impl_holder):
            execute(s, holder[0])

        WORDS[name] = forward_ref

        # Now parse (recursive references will resolve to forward_ref)
        quot = self._parse_quotation(body)

        # Update the implementation
        impl_holder[0] = quot
        print(f"  Defined: {name}")

    def _load_file(self, filename: str) -> None:
        """Load and execute a Joy file."""
        try:
            with open(filename) as f:
                source = f.read()
            program = parse(source)
            self._execute_program(program)
            print(f"  Loaded: {filename}")
        except FileNotFoundError:
            print(f"  Error: file not found: {filename}")
        except Exception as e:
            print(f"  Error loading {filename}: {e}")

    def _execute_program(self, program: List[Any]) -> None:
        """Execute a parsed program."""
        for item in program:
            if callable(item):
                item(self.stack)
            elif isinstance(item, list):
                self.stack.push(item)
            elif isinstance(item, str):
                if item in WORDS:
                    WORDS[item](self.stack)
                else:
                    raise NameError(f"Unknown word: {item}")
            else:
                self.stack.push(item)

    def _show_word_help(self, word_name: str) -> None:
        """Show help for a specific word."""
        if word_name not in WORDS:
            # Try to find similar words
            similar = [w for w in WORDS if word_name in w or w in word_name]
            print(f"  Unknown word: {word_name}")
            if similar:
                print(f"  Did you mean: {', '.join(sorted(similar)[:5])}")
            return

        word_func = WORDS[word_name]
        doc = getattr(word_func, "__doc__", None)

        print(f"\n  {word_name}")
        if doc:
            # Parse stack effect from docstring (format: "X Y -> Z : Description")
            doc = doc.strip()
            print(f"    {doc}")
        else:
            print("    (no documentation)")

        # Show aliases
        aliases = [name for name, func in WORDS.items() if func is word_func and name != word_name]
        if aliases:
            print(f"    Aliases: {', '.join(sorted(aliases))}")
        print()

    def _show_help(self) -> None:
        """Show help information."""
        print("""
PyJoy2 Hybrid REPL

Syntax:
  word          Execute Joy word (dup, +, map, etc.)
  `expr`        Evaluate Python expression, push result
  $(expr)       Alternative Python expression syntax
  !stmt         Execute Python statement
  [...]         Quotation (unevaluated program)
  "..."         String literal
  123, 3.14     Number literals
  true, false   Boolean literals

Commands:
  .s            Show stack
  .c            Clear stack
  .w            List all words
  .help W       Show help for word W
  .def N [...]  Define word N with body [...]
  .import M     Import Python module M
  .load F       Load Joy file F
  .help         Show this help

Defining words:
  .def square [dup *]           Joy-style
  @define("cube")               Python decorator
  def cube(x): return x**3

  @word                         Auto-pop args, push result
  def add3(a, b, c):
      return a + b + c

  WORDS["double"] = lambda s: s.push(s.pop() * 2)
""")

    def run(self) -> None:
        """Main REPL loop."""
        print("PyJoy2 - A Pythonic Joy")
        print("Type .help for help, 'quit' to exit\n")

        while True:
            try:
                prompt = "... " if self.pending_lines else "> "
                line = input(prompt)

                # Handle quit
                if line.strip() == "quit" and not self.pending_lines:
                    break

                # Handle empty line in pending block
                if not line.strip() and self.pending_lines:
                    code = "\n".join(self.pending_lines)
                    self._exec(code)
                    self.pending_lines = []
                    self._register_new_words()
                    continue

                # Try to handle as Python block
                if self._handle_python_block(line):
                    continue

                # Process as Joy/hybrid line
                self.process_line(line)

                # Show top of stack
                if self.stack:
                    top = self.stack.peek()
                    repr_str = repr(top)
                    if len(repr_str) > 70:
                        repr_str = repr_str[:67] + "..."
                    print(repr_str)

            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\n  Interrupted")
                self.pending_lines = []
            except Exception as e:
                self.pending_lines = []
                print(f"  Error: {e}")
                if "--debug" in sys.argv:
                    traceback.print_exc()


def repl(stack: Stack | None = None) -> None:
    """Start the hybrid REPL."""
    HybridREPL(stack).run()


def run(source: str, stack: Stack | None = None) -> Stack:
    """
    Run a Joy program and return the stack.

    Example:
        >>> run("3 4 + dup *")
        Stack([49])
    """
    if stack is None:
        stack = Stack()
    repl_instance = HybridREPL(stack)

    # Process each line
    for line in source.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            repl_instance.process_line(line)

    return stack
