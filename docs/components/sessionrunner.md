## Component : SessionRunner

Purpose:
Executing multiple runs of a GraphRunner object, and maintaining state variables that are kept between runs. GraphRunner will not retain the `state_dict` between calls to `execute()`, so the SessionRunner is used to do this

Key Responsibilities:
- Run the same GraphRunner across multiple calls
- maintain copies of keys that are meant to be seen by multiple calls
- keep a list of the `trace_log` output from each call

### Attributes
- graphrunner : the GraphRunner instance being used
- session_keys : a list of the keys that are meant to be held between execution calls
- session_dict : the dictionary of key-value pairs with keys from session_keys only
- trace_logs : a list of the trace_logs from all the calls to the graphrunner

### Method
- execute : takes the session_dict and input to execute combined to pass to graphrunner and have graphrunner execute on that input. Update any keys in session_dict and update the trace_logs

### Usage
```python
# we assume here that graphrunner is already defined
session_keys = ['message_history']
sessionrunner = SessionRunner(
    graphrunner=graphrunner,
    session_keys=session_keys,
)
sessionrunner.execute({"user query": "What is the capital of France?"})
sessionrunner.execute({"user_query": "What is a notable landmark there?"})
```