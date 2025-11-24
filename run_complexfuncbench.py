import sys
import os

from benchmarks.ComplexFuncBench.utils.utils import load_json
from benchmarks.ComplexFuncBench.evaluation import process_example, get_args, MODEL_MAPPING
from multiprocessing import Pool, Manager
from functools import partial

# Import our custom adapter
from benchmarks.AdapterComplexFuncBench.runner import LocalRunner

# --- CONFIGURATION ---
# Update this path to point to where you cloned the ComplexFuncBench repo
COMPLEX_BENCH_PATH = os.path.abspath(".benchmarks/ComplexFuncBench")
DATASET_PATH = os.path.join(COMPLEX_BENCH_PATH, "data/ComplexFuncBench.jsonl")
# ---------------------


def setup_environment():
    """Injects ComplexFuncBench into sys.path so imports work natively."""
    if not os.path.exists(COMPLEX_BENCH_PATH):
        raise FileNotFoundError(
            f"Could not find ComplexFuncBench at: {COMPLEX_BENCH_PATH}"
        )
    sys.path.insert(0, COMPLEX_BENCH_PATH)


def main():
    # Register our LocalRunner into the benchmark's mapping
    # We use a custom key "local-proxy" that we will pass as --model_name
    MODEL_MAPPING["local-proxy"] = LocalRunner

    # Parse arguments (simulating command line args for the benchmark)
    # You can pass arguments to this script, and they will be parsed by get_args()
    try:
        args = get_args()
    except SystemExit:
        # Handle case where get_args might try to exit if required args are missing
        # We force defaults if running without args
        class Args:
            model_name = "local-proxy"
            input_file = DATASET_PATH
            log_dir = "logs/complex_bench_proxy.log"
            output_dir = "results/complex_bench_proxy.jsonl"
            exp_name = "local-test"
            proc_num = 1  # Use 1 for local debugging/proxy safety
            debug = False

        args = Args()

    # Ensure output dirs exist (replicating logic from evaluation.py)
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)

    print(f"🚀 Starting ComplexFuncBench on Local Proxy...")
    print(f"📂 Dataset: {args.input_file}")
    print(f"💾 Results: {args.output_dir}")

    # Load Data
    test_data = load_json(args.input_file)

    # Filter already processed IDs
    if os.path.exists(args.output_dir):
        finished_data = load_json(args.output_dir)
        finished_ids = {d["id"] for d in finished_data}
        test_data = [d for d in test_data if d["id"] not in finished_ids]
        print(f"⏭️  Skipping {len(finished_ids)} already completed items.")

    if not test_data:
        print("✅ All items completed.")
        return

    # Run Evaluation Loop
    # We use 1 process by default to avoid overwhelming the local proxy state
    with Manager() as manager:
        # Note: If your proxy handles concurrency well, you can increase processes
        pool = Pool(processes=args.proc_num)
        process_func = partial(process_example)

        # Map the process function
        pool.starmap(process_func, [(data, args) for data in test_data])

        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
