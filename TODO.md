# TODO

Prioritized enhancements to make pyjoy2 more valuable and usable.

## Priority 1: Core Usability

### Better Error Messages
- [ ] Include source location (line, column) in runtime errors
- [ ] Show stack state at point of error
- [ ] Suggest fixes for common mistakes (e.g., "did you mean 'dup'?")

### REPL Improvements
- [ ] Command history persistence across sessions (save to ~/.pyjoy2_history)
- [ ] Tab completion for words and commands
- [ ] Multi-line input for complex definitions
- [ ] `.undo` command to revert last operation

### Word Documentation
- [x] `.help word` to show docstring for any word
- [x] `.words pattern` to filter word list (e.g., `.words map` shows map-related)
- [ ] Include stack effect signatures in `.w` output

## Priority 2: Language Completeness

### Additional Recursion Combinators
- [ ] `genrec` - general recursion (most flexible)
- [ ] `tailrec` - tail-recursive combinator (enables optimization)
- [ ] `condlinrec` - conditional linear recursion

### Control Flow
- [x] `cond` - multi-way conditional (list of [test action] pairs)
- [x] `case` - pattern matching on values

### String Operations
- [x] `chars` - string to list of characters
- [x] `unchars` - list of characters to string
- [ ] `format` - string formatting with stack values
- [x] `upper`, `lower`, `trim` - common string operations

### Set Operations
- [ ] `union`, `intersection`, `difference`
- [ ] `member` - set membership test
- [ ] `subset` - subset test

## Priority 3: Developer Experience

### Debugging Support
- [ ] `.trace on/off` - trace execution showing each word and stack
- [ ] `.break word` - breakpoint when word is called
- [ ] `.step` - single-step execution mode
- [ ] Stack visualization in trace output

### File and Module System
- [ ] `DEFINE name == body.` syntax (Joy compatibility)
- [ ] `.save file` - save current word definitions
- [ ] Module/namespace support for organizing code
- [ ] Standard library directory (~/.pyjoy2/lib/)

### Testing Support
- [x] `assert` word for inline assertions
- [ ] `.test file` - run tests from a Joy file
- [ ] Test runner that checks expected stack results

## Priority 4: Performance

### Optimization
- [ ] Tail-call optimization for recursive definitions
- [ ] Compile frequent quotations to Python bytecode
- [ ] Memoization combinator for pure functions

### Profiling
- [ ] `.profile on/off` - time spent in each word
- [ ] `.bench expr` - benchmark an expression

## Priority 5: Extended Features

### Parallel/Concurrent Operations
- [ ] `pmap` - parallel map using multiprocessing
- [ ] `async` combinator for async/await integration

### Data Structure Support
- [ ] Dictionary/map operations (`get`, `put`, `keys`, `values`)
- [ ] Record/struct support with named fields
- [ ] `table` type for tabular data

### I/O Words
- [ ] `read-file`, `write-file` - file operations
- [ ] `slurp`, `spit` - read/write entire files
- [ ] `lines` - file to list of lines

### Interop Enhancements
- [ ] `@joyword` Python decorator that auto-converts Joy lists to Python
- [ ] Better error messages for Python exceptions
- [ ] `py-call` for calling Python methods with Joy syntax

## Priority 6: Documentation and Examples

### Example Programs
- [ ] examples/ directory with practical programs
- [ ] Advent of Code solutions
- [ ] Data processing pipelines
- [ ] Algorithm implementations (sorting, searching, trees)

### Documentation
- [ ] Tutorial for Joy newcomers
- [ ] Cookbook of common patterns
- [ ] API documentation for Python integration

## Quick Wins (Low Effort, High Value)

- [x] `show` - print without popping (already exists, but add `.show` alias)
- [x] `clr` alias for `clear`
- [x] `id` or `nop` - no-operation word
- [x] `dup2`, `pop2` - operate on pairs (already implemented)
- [x] `neg?`, `pos?`, `zero?` - numeric predicates
- [x] `empty?` alias for `null`
- [x] `inc`, `dec` aliases for `succ`, `pred`

## Notes

### Design Principles
1. **Python-first**: When in doubt, use Python semantics and stick to the Zen of Python 
2. **Practical over pure**: Pragmatic features over theoretical purity
3. **Interop is key**: Make it trivial to use Python libraries
4. **Discoverable**: Good error messages and help system

### Non-Goals
- Full Joy compatibility (we diverge intentionally for Python integration)
- High-performance numerical computing (use NumPy via Python interop)
- Static typing (dynamic typing matches both Joy and Python philosophy)
