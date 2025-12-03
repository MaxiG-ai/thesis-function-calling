# Contributing to this project

- Always create a feature branch before you go to work.
- The feature branch should follow the template "ai-<descriptive name>"
- add all your changes to the feature branch
- let me commit the branch if I am happy with the results. 

## Package management

- This project uses uv for all package management
- Never run commands directly (python, pytest, etc.)
- Always prefix commands with `uv run <command>`
- Example: `uv run python script.py` not `python script.py`
- Example: `uv run pytest` not `pytest`

## Testing approach

- Never create throwaway test scripts or ad hoc verification files
- If you need to test functionality, write a proper test in the test suite
- All tests go in the `tests/` directory following the project structure
- Tests should be runnable with the rest of the suite (`uv run pytest`)
- Even for quick verification, write it as a real test that provides ongoing value

## Testing preferences

- Write all Python tests as `pytest` style functions, not unittest classes
- Use descriptive function names starting with `test_`
- Prefer fixtures over setup/teardown methods
- Use assert statements directly, not self.assertEqual

## Logging Style

- There are two main layers of logging, `logging.INFO` and `logging.DEBUG`
- `INFO` is supposed to give highlevel information about the benchmark run and will always be displayed to the user.
- `DEBUG` is for detecting errors or logical mistakes in code implementation
- The levels are set in `config.toml`.

## Your Developer Agent Behaviour

- Never write simple test scripts, always refer to the testing approach & preferences above
- Never summarize your finding to a .md unless i explicitly ask you.
