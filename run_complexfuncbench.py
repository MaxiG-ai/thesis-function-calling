import os
import subprocess
import sys
import logging
from src.llm_orchestrator import LLMOrchestrator

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComplexFunc-Adapter")

# --- CONFIGURATION ---
BENCH_DIR = "benchmarks/ComplexFuncBench"
PROXY_URL = "http://localhost:8000/v1"
PROXY_KEY = "sk-thesis-proxy"


def run_complex_eval(model_key: str):
    """
    Runs ComplexFuncBench against our local Middleware Proxy.
    """

    # 1. Setup the Environment to Trick the Benchmark
    env = os.environ.copy()

    # The Benchmark likely uses the standard OpenAI client.
    # We override the base URL to point to OUR server.
    env["OPENAI_BASE_URL"] = PROXY_URL
    env["OPENAI_API_KEY"] = PROXY_KEY

    # Some benchmarks look for specific "Eval" keys
    env["EVAL_MODEL"] = model_key

    logger.info(f"🚀 Starting ComplexFuncBench for {model_key}...")
    logger.info(f"   Target: {PROXY_URL}")

    # 2. Construct the Command
    # Note: I am inferring the script name based on standard patterns.
    # You MUST check the repo for the exact python file (e.g., eval.py, main.py).
    # Based on their docs, it's likely `evaluate.py` or similar in the root.

    cmd = [
        sys.executable,
        "evaluation.py",  # <--- ⚠️ VERIFY THIS FILENAME in the repo!
        "--model_name",
        model_key,
    ]

    # 3. Run it
    try:
        # We stream output so you can see progress
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=BENCH_DIR,  # Run from inside the benchmark dir to find relative paths
        )

        for line in process.stdout:
            print(f"[BENCH] {line.strip()}")

        process.wait()

        if process.returncode == 0:
            logger.info("✅ Benchmark Finished Successfully.")
        else:
            logger.error(f"❌ Benchmark Failed with code {process.returncode}")

    except Exception as e:
        logger.error(f"Execution Error: {e}")


if __name__ == "__main__":
    # 1. Check if Proxy is Alive
    import requests

    try:
        requests.get("http://localhost:8000/health")
    except Exception:
        logger.error("⛔ Middleware is NOT running. Start 'main.py' first!")
        sys.exit(1)

    # 2. Get Active Model from Config
    orchestrator = LLMOrchestrator()
    active_model = orchestrator.active_model_key

    # 3. Run
    run_complex_eval(active_model)
