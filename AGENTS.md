# Contributing to this project

- Always create a feature branch before you go to work.
- The feature branch should follow the template "opencode->descriptive name<"
- add all your changes to the feature branch
- let me commit the branch if I am happy with the results.

## Package management

- This project uses uv for all package management
- Never run commands directly (python, pytest, etc.)
- Always prefix commands with `uv run <command>`
- Example: `uv run python script.py` not `python script.py`
- Example: `uv run pytest` not `pytest`

## Testing

This repo should be developed in Test-Driven-Development Style. Always begin by writing tests that suffice the requirements. Then implement code until tests pass. After writing tests explain them to me and use descriptive docstrings.

- I want every test to have a detailed explanation in its docstring
- Never create throwaway test scripts or ad hoc verification files
- Even for quick verification, write it as a real test that provides ongoing value
- Write all Python tests as `pytest` style functions, not unittest classes
- Prefer fixtures over setup/teardown methods
- Use assert statements directly, not self.assertEqual
- If you need to test functionality, write a proper test in the test suite
- All tests go in the `tests/` directory following the project structure
- Use descriptive function names starting with `test_`
- Tests should be runnable with the rest of the suite (`uv run pytest`)

## Logging

- Logging is provided by `memorch.utils.logger`. Use `get_logger(__name__)` to get a logger instance.
- Log levels can be configured in `config.toml` under the `[logging]` section.

## Your Developer Agent Behaviour

- I will review all your changes by hand. Make sure they have excellent quality and do not change more than necessary.
- Never write simple test scripts, always refer to the testing approach & preferences above
- Never summarize your finding to a .md unless i explicitly ask you.
- Code efficiently (minimizing lines of code)
- code understandably (explain code in comments)
- remove duplicate functions
