import streamlit as st
import weave
import json

st.set_page_config(layout="wide", page_title="🔍 Memory Strategy Trace Viewer")
st.title("🔍 Memory Strategy Trace Viewer")

# --- HELPERS ---
def unwrap(data):
    """Recursively convert Weave objects to native Python types."""
    if hasattr(data, "data"):
        return unwrap(data.data)
    if isinstance(data, (list, tuple)) or "WeaveList" in type(data).__name__:
        return [unwrap(x) for x in data]
    if isinstance(data, dict) or "WeaveDict" in type(data).__name__:
        return {k: unwrap(v) for k, v in data.items()}
    return data

def get_messages(inputs):
    """Extract messages list from inputs."""
    data = unwrap(inputs)
    return data.get("messages", []) if isinstance(data, dict) else []

def get_strategy_key(call):
    """Extract strategy_key from call inputs."""
    inputs = unwrap(call.inputs)
    return inputs.get("strategy_key", "unknown") if isinstance(inputs, dict) else "unknown"

# --- SIDEBAR: Project & Call ID ---
st.sidebar.header("Load Trace")
org = st.sidebar.text_input("Weave Org", "maxigraf-karlsruhe-institute-of-technology")
project = st.sidebar.text_input("Weave Project", "gpt41mini_memory_test")
call_id = st.sidebar.text_input("Call ID", placeholder="019b2bfb-3ec6-7bf9-bc1b-1c6829cf5d13")

if not call_id:
    st.info("Enter a Call ID in the sidebar to load a trace.")
    st.stop()

# --- FETCH CALL ---
try:
    client = weave.init(f"{org}/{project}")
    call = client.get_call(call_id)
    if not call:
        st.error(f"Call {call_id} not found.")
        st.stop()
except Exception as e:
    st.error(f"Error loading call: {e}")
    st.stop()

# --- EXTRACT DATA ---
inputs = unwrap(call.inputs)
output = unwrap(call.output)
summary = unwrap(call.summary) if call.summary else {}

in_messages = [unwrap(m) for m in get_messages(call.inputs)]
out_messages = [unwrap(m) for m in output] if isinstance(output, list) else []

# --- SIDEBAR: METADATA ---
st.sidebar.header("Trace Metadata")
strategy = get_strategy_key(call)
latency = summary.get("weave", {}).get("latency_ms", 0)
memory_key = inputs.get("strategy_key", "N/A") if isinstance(inputs, dict) else "N/A"
time_str = call.started_at.strftime("%Y-%m-%d %H:%M:%S") if call.started_at else "N/A"

st.sidebar.metric("Strategy", strategy)
st.sidebar.metric("Memory Key", memory_key)
st.sidebar.metric("Messages", f"{len(in_messages)} → {len(out_messages)}")
st.sidebar.metric("Latency", f"{latency:.0f}ms")
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
        for i, msg in enumerate(out_messages[0]):
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
    st.json({"trace_id": call.id, "inputs": inputs, "output": output, "summary": summary}, expanded=False)