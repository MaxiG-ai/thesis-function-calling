import random
import time


def generate_verbose_logs(service_name: str, num_lines: int = 500) -> str:
    """
    Simulates retrieving a massive log file from a server.
    Generates ~20-30 tokens per line.
    500 lines ≈ 10k-15k tokens.
    """
    log_levels = ["INFO", "DEBUG", "WARN", "ERROR"]
    messages = [
        "Connection reset by peer",
        "Garbage collection started",
        "User session authenticated",
        "Cache miss for key: user_metadata_v2",
        "Transaction committed successfully",
        "Payload validation failed: missing 'id'",
        "Retrying database connection (attempt 2/5)",
    ]

    output = []
    output.append(f"--- LOGS START: {service_name} ---")

    for i in range(num_lines):
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        level = random.choice(log_levels)
        msg = random.choice(messages)
        # Add random hex noise to make it harder to compress
        trace_id = f"{random.getrandbits(64):016x}"

        line = f"[{timestamp}] [{level}] [TraceID:{trace_id}] {msg}"
        output.append(line)

    output.append(f"--- LOGS END: {service_name} ---")
    return "\n".join(output)


# Tool Definition Schema (OpenAI Format)
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "fetch_server_logs",
            "description": "Fetches the raw stdout logs from a specific backend service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "The service to debug (e.g., 'auth-service', 'payment-gateway').",
                    },
                    "num_lines": {
                        "type": "integer",
                        "description": "Number of log lines to retrieve. Default is 500.",
                    },
                },
                "required": ["service_name"],
            },
        },
    }
]
