def test_source_contains_expected_classes_and_no_langgraph_imports():
    import pathlib

    base = pathlib.Path(__file__).parents[1]

    files_to_check = {
        'research_agent.py': 'ResearcherAgent',
        'research_agent_scope.py': 'ScopeAgent',
        'multi_agent_supervisor.py': 'SupervisorAgent',
        'research_agent_full.py': 'FullAgent',
    }

    for fname, clsname in files_to_check.items():
        p = base / fname
        src = p.read_text()
        assert clsname in src, f"{clsname} not found in {fname}"
        assert 'from langgraph' not in src, f"Found langgraph import in {fname}"

    # Also ensure the package doesn't contain a local langgraph directory
    assert not (base / 'langgraph').exists(), "src_alt/langgraph directory should be removed"
