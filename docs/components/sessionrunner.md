## Component: SessionRunner

Purpose:
Executing multiple runs of a GraphRunner object, and maintaining state variables that are kept between runs. GraphRunner will not retain the `state_dict` between calls to `execute()`, so the SessionRunner is used to do this.

Key Responsibilities:
- Run the same GraphRunner across multiple calls
- Maintain copies of keys that are meant to be seen by multiple calls
- Keep a list of the `trace_log` output from each call
- Accumulate token usage across all executions

### Constructor

`SessionRunner(graph, session_keys=None)`

- `graph`: the `GraphRunner` instance to execute.
- `session_keys` (optional): list of state dict keys whose values carry over between `execute()` calls. Defaults to `[]`.

### Attributes
- `graph`: the `GraphRunner` instance being used
- `session_keys`: list of keys retained between executions
- `session_dict`: current persisted key-value pairs (only the keys from `session_keys`)
- `trace_logs`: list of trace log lists, one entry per `execute()` call
- `in_tokens`: cumulative prompt tokens across all executions
- `out_tokens`: cumulative completion tokens across all executions

### Method

**`execute(input) -> dict`**

Merges `session_dict` with `input` (input takes precedence on conflicts), runs `graph.execute()`, updates `session_dict` with any result keys that are in `session_keys`, appends the trace log, and returns the raw result from `graph.execute()`.

### Usage

```python
# Assumes graphrunner is already defined
session_keys = ["message_history"]
sessionrunner = SessionRunner(
    graph=graphrunner,
    session_keys=session_keys,
)
sessionrunner.execute({"user_query": "What is the capital of France?"})
sessionrunner.execute({"user_query": "What is a notable landmark there?"})

print(sessionrunner.in_tokens, sessionrunner.out_tokens)
```
