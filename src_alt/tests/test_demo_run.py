def test_demo_run_imports_and_executes_run_mock():
    """Import and run the demo's run_mock function to ensure the mock
    environment and the alt agents run without raising exceptions.
    """
    from src_alt.demo import run_mock

    # Should not raise
    run_mock()
