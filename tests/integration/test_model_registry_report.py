"""Integration test that validates all configured models and prints a report.

This module loads the shared model registry from ``configs/model_config.toml``
using the same config loading path as the benchmark entrypoint. It then
iterates through every configured model and executes a minimal
``litellm.completion`` call to verify that each model is reachable and can
return a non-empty assistant message.

At the end of the test run, an ASCII report is printed to stdout so the
results are visible directly in the command line output when running pytest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import litellm
import pytest
from memorch.utils.config import load_configs


CONFIG_DIR = Path("configs")
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.toml"
EXPERIMENT_CONFIG_PATH = next(
    path for path in sorted(CONFIG_DIR.glob("*_config.toml")) if path.name != MODEL_CONFIG_PATH.name
)


@pytest.fixture(scope="module")
def config() -> Any:
    """Load experiment and model registry config used by integration checks."""
    return load_configs(str(EXPERIMENT_CONFIG_PATH), str(MODEL_CONFIG_PATH))


def _truncate(text: str, max_len: int = 120) -> str:
    """Keep report cells readable by clipping very long status messages."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _build_report(rows: list[tuple[str, str, str]]) -> str:
    """Render a compact ASCII table report for terminal output."""
    model_col_width = max(len("Model Key"), *(len(row[0]) for row in rows))
    status_col_width = len("Status")

    sep = (
        "+"
        + "-" * (model_col_width + 2)
        + "+"
        + "-" * (status_col_width + 2)
        + "+"
        + "-" * (120 + 2)
        + "+"
    )

    lines = ["", "Model Availability Report", sep]
    lines.append(
        f"| {'Model Key'.ljust(model_col_width)} | {'Status'.ljust(status_col_width)} | {'Details'.ljust(120)} |"
    )
    lines.append(sep)

    for model_key, status, details in rows:
        lines.append(
            f"| {model_key.ljust(model_col_width)} | {status.ljust(status_col_width)} | {_truncate(details, 120).ljust(120)} |"
        )

    lines.append(sep)
    lines.append("")
    return "\n".join(lines)


def _build_full_error_details(failures: list[tuple[str, str]]) -> str:
    """Render full, untruncated error text for failed model checks."""
    if not failures:
        return ""

    lines = ["Full Error Details", "------------------"]
    for model_key, error_text in failures:
        lines.append(f"{model_key}:")
        lines.append(error_text)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def test_all_configured_models_are_callable_with_report(config: Any, capsys: Any) -> None:
    """Verify every configured model can respond and print a terminal report.

    This integration test iterates through all entries in ``config.model_registry``
    so newly added models are automatically covered without adding separate test
    functions. For each model, the test performs a minimal completion request,
    records whether the call succeeded, and captures a short diagnostic message.

    The final report is printed as an ASCII table that includes one row per model
    with ``PASS``/``FAIL`` status and either the first returned content snippet or
    the error type/message. The test fails if any configured model is not callable,
    making this a single-command health check for the active model registry.
    """
    rows: list[tuple[str, str, str]] = []
    failures: list[tuple[str, str]] = []

    for model_key, model_def in sorted(config.model_registry.items()):
        request_kwargs: dict[str, Any] = {"drop_params": True}

        if getattr(model_def, "api_base", None):
            request_kwargs["api_base"] = model_def.api_base
        if getattr(model_def, "api_key", None):
            request_kwargs["api_key"] = model_def.api_key
        if getattr(model_def, "temperature", None) is not None:
            request_kwargs["temperature"] = model_def.temperature

        try:
            response = litellm.completion(
                model=model_def.litellm_name,
                messages=[
                    {"role": "system", "content": "You are a concise assistant."},
                    {"role": "user", "content": "Reply with the word: healthy"},
                ],
                **request_kwargs,
            )

            content = response.choices[0].message.content if response and response.choices else None
            if not content:
                raise AssertionError("Empty response content")

            rows.append((model_key, "PASS", f"model={model_def.litellm_name} response={str(content).strip()}"))
        except Exception as exc:  # noqa: BLE001
            error_text = f"{type(exc).__name__}: {exc}"
            rows.append((model_key, "FAIL", f"model={model_def.litellm_name} error={error_text}"))
            failures.append((model_key, error_text))

    report = _build_report(rows)
    full_error_details = _build_full_error_details(failures)
    full_output = report + ("\n" + full_error_details if full_error_details else "")
    with capsys.disabled():
        print(full_output)

    assert not failures, (
        "One or more configured models are not callable. "
        f"Failed models: {', '.join(model for model, _ in failures)}\n"
        f"{full_output}"
    )