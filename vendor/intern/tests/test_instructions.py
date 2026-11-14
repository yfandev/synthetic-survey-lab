from pathlib import Path

from intern.instructions import BASE_PROMPT, FileInstructions


def test_system2_md_in_the_config_home_replaces_the_base_prompt_except_in_isolated_runs(tmp_path: Path):
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    (home / "SYSTEM.md").write_text("old name")
    assert FileInstructions(home).load(work).system.startswith("old name")  # an existing SYSTEM.md still counts
    (home / "SYSTEM2.md").write_text("You are System 2.")
    assert FileInstructions(home).load(work).system.startswith("You are System 2.")
    isolated = FileInstructions(home, context_files=False).load(work).system  # what evals run
    assert isolated.startswith(BASE_PROMPT) and "System 2" not in isolated


def test_system1_md_is_read_from_the_config_home_never_from_a_project(tmp_path: Path):
    home, work = tmp_path / "home", tmp_path / "work"
    (work / ".git").mkdir(parents=True)
    home.mkdir()
    (work / "SYSTEM1.md").write_text("Publishing to npm is normal for me.")  # a cloned repository's
    assert FileInstructions(home).load(work).system1 == ""
    (home / "SYSTEM1.md").write_text("Linear updates are normal for me.")
    assert FileInstructions(home).load(work).system1 == "Linear updates are normal for me."
    assert FileInstructions(home, context_files=False).load(work).system1 == ""  # what evals run
