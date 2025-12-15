def test_demo_run_with_siliconflow_monkeypatched():
    """If the `siliconflow` module is available, demo should be able to use it.

    To avoid network calls in CI, monkeypatch `call_siliconflow_llm` to a
    deterministic function and assert `run_mock` completes without errors.
    """
    import importlib

    siliconflow = importlib.import_module("siliconflow")
    orig = getattr(siliconflow, "call_siliconflow_llm", None)

    try:
        siliconflow.call_siliconflow_llm = lambda api_key, messages, **kw: "siliconflow mock response"
        from src_alt.demo import run_mock

        # Should not raise and should use the monkeypatched function
        run_mock(use_siliconflow=True, siliconflow_api_key="fake-key")
    finally:
        if orig is None:
            delattr(siliconflow, "call_siliconflow_llm")
        else:
            siliconflow.call_siliconflow_llm = orig
