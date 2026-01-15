import os

from benchmarks.complex_func_bench.utils.utils import load_json

from langfuse import get_client

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY","")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY","")
os.environ["LANGFUSE_BASE_URL"] = os.getenv("LANGFUSE_BASE_URL","")

langfuse_client = get_client()
# Verify connection
if  langfuse_client.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

# Load dataset
data_path = os.path.join("benchmarks", "complex_func_bench", "data", "demoComplexFuncBench.jsonl")
dataset = load_json(data_path)


langfuse_client.create_dataset(
    name="ComplexFuncBench",
    description="Dataset for evaluating complex function calling capabilities of LLMs.",
)

for item in dataset:
    langfuse_client.create_dataset_item(
        id = item["id"],
        dataset_name="DemoComplexFuncBench",
        input=item["conversations"][0],
        expected_output=item["conversations"][1:-1],
        metadata=item["functions"],
    )
