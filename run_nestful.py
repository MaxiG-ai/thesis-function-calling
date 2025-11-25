import os
import subprocess
import sys
import logging
from src.llm_orchestrator import LLMOrchestrator

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NESTFUL-Adapter")

# Configuration
NESTFUL_DIR = "benchmarks/nestful"
PROXY_URL = "http://localhost:8000/v1"  # Where your FastAPI middleware will run
PROXY_API_KEY = "sk-thesis-proxy"  # Dummy key for your local proxy


def check_proxy_health():
    """Ensures your middleware is actually running before starting the benchmark."""
    import requests

    try:
        # Assuming your proxy has a health endpoint
        resp = requests.get("http://localhost:8000/health")
        if resp.status_code == 200:
            return True
    except:
        return False
    return False


def run_nestful_eval(model_name: str, output_dir: str):
    """
    Runs the NESTFUL evaluation script with environment variables
    that redirect traffic to our Proxy.
    """

    # 1. Define the Redirect Environment
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = PROXY_API_KEY
    env["OPENAI_BASE_URL"] = PROXY_URL  # The magic switch

    # Force NESTFUL to use the model name we want (so it shows up in your logs correctly)
    # Some benchmarks accept a --model arg, others hardcode.
    # We'll try passing it via args if their script supports it.

    cmd = [
        sys.executable,  # Use the current python interpreter (with installed deps)
        "benchmarks/nestful/src/eval.py",
        "--model",
        model_name,
        "--save_directory",
        f"results/{NESTFUL_DIR}/{model_name}",
        "--dataset",
        f"{NESTFUL_DIR}/data/nestful_data.jsonl",
        "--icl_count",
        "3",
    ]

    logger.info(f"🚀 Launching NESTFUL for {model_name} via Proxy...")

    # 2. Execute as a subprocess
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        # 3. Log Results
        if result.returncode == 0:
            logger.info("✅ NESTFUL Run Complete.")
            print(result.stdout)
        else:
            logger.error("❌ NESTFUL Failed.")
            print(result.stderr)

    except Exception as e:
        logger.error(f"Execution Error: {e}")


if __name__ == "__main__":
    # ⚠️ PRE-REQUISITE: Your Middleware (main.py) must be running in a separate terminal!

    # Example: Run for the active model in your config
    orchestrator = LLMOrchestrator()
    active_model = orchestrator.active_model_key

    if not check_proxy_health():
        logger.error("⛔ Middleware Proxy is NOT running on port 8000.")
        logger.error(
            "   Please run 'uv run fastapi dev main.py' in another terminal first."
        )
        sys.exit(1)

    run_nestful_eval(model_name=active_model, output_dir="results/nestful")
