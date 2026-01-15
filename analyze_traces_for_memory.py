import streamlit as st
from langfuse import Langfuse
import json

st.set_page_config(layout="wide", page_title="🔍 Memory Strategy Trace Viewer")
st.title("🔍 Memory Strategy Trace Viewer")

# --- HELPERS ---
def unwrap(data):
    """Recursively convert Langfuse objects to native Python types."""
    if hasattr(data, "data"):
        return unwrap(data.data)
    if isinstance(data, (list, tuple)):
        return [unwrap(x) for x in data]
    if isinstance(data, dict):
        return {k: unwrap(v) for k, v in data.items()}
    return data

def get_messages(inputs):
    """Extract messages list from inputs."""
    data = unwrap(inputs)
    return data.get("messages", []) if isinstance(data, dict) else []

def get_strategy_key(observation):
    """Extract strategy_key from observation inputs."""
    inputs = unwrap(observation.input) if observation.input else {}
    return inputs.get("strategy_key", "unknown") if isinstance(inputs, dict) else "unknown"

# --- SIDEBAR: Project & Trace ID ---
st.sidebar.header("Load Trace")
st.sidebar.info("Configure LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST as environment variables")
trace_id = st.sidebar.text_input("Trace ID", placeholder="Enter Langfuse trace ID")

if not trace_id:
    st.info("Enter a Trace ID in the sidebar to load a trace.")
    st.stop()

# --- FETCH TRACE ---
try:
    client = Langfuse()
    trace = client.fetch_trace(trace_id)
    if not trace or not trace.data:
        st.error(f"Trace {trace_id} not found.")
        st.stop()
    trace_data = trace.data
except Exception as e:
    st.error(f"Error loading trace: {e}")
    st.stop()

# --- EXTRACT DATA ---
inputs = unwrap(trace_data.input) if trace_data.input else {}
output = unwrap(trace_data.output) if trace_data.output else {}

in_messages = [unwrap(m) for m in get_messages(trace_data.input)] if trace_data.input else []
out_messages = [unwrap(m) for m in output] if isinstance(output, list) else []

# --- SIDEBAR: METADATA ---
st.sidebar.header("Trace Metadata")
latency_ms = trace_data.latency if hasattr(trace_data, 'latency') and trace_data.latency else 0
memory_key = inputs.get("strategy_key", "N/A") if isinstance(inputs, dict) else "N/A"
time_str = trace_data.timestamp.strftime("%Y-%m-%d %H:%M:%S") if hasattr(trace_data, 'timestamp') and trace_data.timestamp else "N/A"

st.sidebar.metric("Memory Key", memory_key)
st.sidebar.metric("Messages", f"{len(in_messages)} → {len(out_messages)}")
st.sidebar.metric("Latency", f"{latency_ms:.0f}ms")
st.sidebar.text(f"Started: {time_str}")

# --- RENDER MESSAGE ---
def render_message(msg, idx=None):
    msg = unwrap(msg)
    if not isinstance(msg, dict):
        return
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    icons = {"user": "👤", "assistant": "🤖", "tool": "🔧", "system": "⚙️"}
    
    with st.container(border=True):
        header = f"{icons.get(role, '💬')} **{role.upper()}**"
        if idx is not None:
            header = f"`{idx}` {header}"
        if role == "tool":
            header += f" → `{msg.get('name', 'unknown')}`"
        st.markdown(header)
        
        if content:
            if role == "tool":
                try:
                    st.json(json.loads(content), expanded=False)
                except:
                    st.code(content, language="json")
            else:
                st.markdown(content)
        
        if "tool_calls" in msg:
            st.markdown("**🛠️ Tool Calls:**")
            for tc in msg["tool_calls"]:
                tc = unwrap(tc)
                st.markdown(tc)
                try:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = json.dumps(json.loads(func.get("arguments", "{}")), indent=2)
                    with st.expander(f"`{name}`"):
                        st.code(args, language="json")
                except:
                    st.warning("Failed to parse tool call arguments.")

st.divider()

# --- MAIN: Side-by-side Messages ---
left, right = st.columns(2)

with left:
    st.subheader(f"📥 Input ({len(in_messages)} messages)")
    for i, msg in enumerate(in_messages):
        render_message(msg, i)

with right:
    st.subheader(f"📤 Output ({len(out_messages)} messages)")
    if out_messages:
        for i, msg in enumerate(out_messages[0] if isinstance(out_messages[0], list) else out_messages):
            render_message(msg, i)
    else:
        st.warning("No output messages.")

# --- EXPANDABLE DETAILS ---
with st.expander("📋 Available Tools"):
    tools = inputs.get("tools", []) if isinstance(inputs, dict) else []
    if tools:
        for t in tools:
            func = t.get("function", {})
            st.markdown(f"- **{func.get('name')}**: {func.get('description', '')[:100]}...")
    else:
        st.write("No tools defined.")

with st.expander("🔎 Raw Trace Data"):
    st.json({"trace_id": trace_data.id, "inputs": inputs, "output": output}, expanded=False)