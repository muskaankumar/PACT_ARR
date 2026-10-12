The log parser in /app/src/parser.py mishandles multi-line stack traces: it emits
one record per physical line instead of grouping the trace with its header.

Fix parse_lines() so a stack trace is attached to the record that precedes it.
