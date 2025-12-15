# src_alt — Standalone Research Agent Demo

This directory holds a self-contained, lightweight reimplementation of the
original agent code using plain Python (no `langgraph` dependency).

Quick start

- Run the local mock demo (no network calls):

```bash
python -m src_alt.demo --mock
```

- Run the demo using the SiliconFlow API (requires `siliconflow.py` and an API key):

```bash
python -m src_alt.demo --mock --siliconflow --siliconflow-api-key YOUR_KEY
```

Tests

The package includes lightweight integration tests. You can run them without
pytest by importing the test module directly:

```bash
python -c "import src_alt.tests.test_demo_run as t; t.test_demo_run_imports_and_executes_run_mock(); print('OK')"
```

Notes

- The demo injects minimal stubs for `langchain` / `langchain_core` so the
  original code imports continue to work. For integration testing with a
  real LLM, use the `--siliconflow` option which will call
  `siliconflow.call_siliconflow_llm` (the `siliconflow.py` in the repo).
- The code favors readability and explicitness over clever abstractions so
  it's easy to modify and test locally.
