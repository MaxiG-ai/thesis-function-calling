import os
import logging
import time

from benchmarks.complex_func_bench.utils.utils import load_json
from src.utils.logger import get_logger

from langfuse import get_client

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = get_logger("DatasetToLangfuse")

os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
os.environ["LANGFUSE_BASE_URL"] = os.getenv("LANGFUSE_BASE_URL", "")

langfuse_client = get_client()

# Verify connection
if langfuse_client.auth_check():
    logger.info("✅ Langfuse client is authenticated and ready!")
else:
    logger.error("❌ Authentication failed. Please check your credentials and host.")
    raise RuntimeError("Failed to authenticate with Langfuse")

# Load full dataset
data_path = os.path.join("benchmarks", "complex_func_bench", "data", "ComplexFuncBench.jsonl")
logger.info(f"📂 Loading dataset from {data_path}...")
dataset = load_json(data_path)
logger.info(f"📊 Loaded {len(dataset)} items from dataset")

# Create or get dataset
dataset_name = "ComplexFuncBench"
logger.info(f"📤 Creating/updating dataset '{dataset_name}' in Langfuse...")

try:
    langfuse_client.create_dataset(
        name=dataset_name,
        description="Dataset for evaluating complex function calling capabilities of LLMs.",
    )
    logger.info(f"✅ Created new dataset '{dataset_name}'")
except Exception as e:
    # Dataset might already exist
    logger.info(f"⚠️  Dataset '{dataset_name}' may already exist (this is OK): {e}")

# Upload items to dataset
uploaded_count = 0
skipped_count = 0
error_count = 0

logger.info("📥 Starting item upload...")
for idx, item in enumerate(dataset, 1):
    try:
        # Extract metadata - handle missing 'functions' field gracefully
        metadata = item.get("functions", {})
        
        # Create dataset item
        langfuse_client.create_dataset_item(
            id=item["id"],
            dataset_name=dataset_name,
            input=item["conversations"][0],
            expected_output=item["conversations"][1:-1],
            metadata=metadata,
        )
        uploaded_count += 1
        
        # Log progress every 100 items
        if idx % 50 == 0:
            logger.info(f"  Progress: {idx}/{len(dataset)} items processed...")
        
        time.sleep(0.5) # To avoid rate limiting
            
    except Exception as e:
        error_msg = str(e)
        # Check if it's a duplicate key error (item already exists)
        if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
            skipped_count += 1
        else:
            logger.warning(f"❌ Error uploading item {item.get('id', 'unknown')}: {e}")
            error_count += 1

logger.info(f"\n{'='*60}")
logger.info(f"✅ Upload Complete!")
logger.info(f"  ✓ Successfully uploaded: {uploaded_count} items")
if skipped_count > 0:
    logger.info(f"  ⊘ Skipped (already exist): {skipped_count} items")
if error_count > 0:
    logger.warning(f"  ✗ Errors: {error_count} items")
logger.info(f"{'='*60}\n")
