# --- MONKEY PATCH TO BYPASS PYARROW/DATASETS CRASH ---
import sys
from types import ModuleType
import importlib.machinery

if 'datasets' not in sys.modules:
    class MockDatasets(ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []
            self.__spec__ = importlib.machinery.ModuleSpec(name, None)

        def __getattr__(self, name):
            if name == '__version__':
                return "3.0.0"
            if name.startswith('__') and name.endswith('__'):
                raise AttributeError(name)
            class Dummy:
                pass
            Dummy.__name__ = name
            return Dummy

    sys.modules['datasets'] = MockDatasets('datasets')

import os
import sys
# Add project root directory to sys.path so memory and conversation modules can be resolved
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -----------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime, timezone, timedelta
from groq import Groq

# Set page layout to wide and title
st.set_page_config(
    page_title="MeshMind Memory Architecture Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for wow factor
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 300;
        font-size: 1.2rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        border-color: rgba(77, 150, 255, 0.3);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4D96FF;
        margin-bottom: 0.2rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #AAAAAA;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        margin-right: 0.5rem;
    }
    
    .badge-blue { background: rgba(77, 150, 255, 0.15); color: #4D96FF; border: 1px solid rgba(77, 150, 255, 0.3); }
    .badge-green { background: rgba(87, 204, 153, 0.15); color: #57CC99; border: 1px solid rgba(87, 204, 153, 0.3); }
    .badge-orange { background: rgba(255, 159, 67, 0.15); color: #FF9F43; border: 1px solid rgba(255, 159, 67, 0.3); }
    .badge-red { background: rgba(255, 107, 107, 0.15); color: #FF6B6B; border: 1px solid rgba(255, 107, 107, 0.3); }
    
    .chat-bubble {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        max-width: 85%;
    }
    
    .chat-user {
        background: rgba(77, 150, 255, 0.1);
        border-left: 4px solid #4D96FF;
        margin-left: auto;
    }
    
    .chat-assistant {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #888888;
    }
    
    .memory-context-box {
        background: rgba(87, 204, 153, 0.05);
        border: 1px dashed rgba(87, 204, 153, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load configuration
@st.cache_data
def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
results_dir = Path(config.get("results_dir", "./results"))

# Load benchmark summary results
def load_benchmark_summary():
    summary_file = results_dir / "benchmark_results.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            return json.load(f)
    return None

# Load condition results
def load_condition_results(condition_name):
    result_file = results_dir / f"{condition_name}_results.json"
    if result_file.exists():
        try:
            with open(result_file, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# Title header
st.markdown('<div class="main-title">MeshMind</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Production-Grade Persistent Memory Architecture for Conversational AI</div>', unsafe_allow_html=True)

# Read results
summary = load_benchmark_summary()
conditions = ["no_memory", "no_forgetting", "recency_decay", "hybrid"]
condition_data = {}

for cond in conditions:
    condition_data[cond] = load_condition_results(cond)

# Check run progress
completed_conditions = [c for c in conditions if condition_data[c] is not None]
is_running_benchmark = len(completed_conditions) < len(conditions)

# Sidebar configurations
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/brain.png", width=80)
    st.markdown("### Architecture Settings")
    st.info(f"**Groq Model:** `{config.get('groq_model')}`\n\n**Judge Model:** `{config.get('judge_model')}`\n\n**Embeddings:** `{config.get('embedding_model')}`")
    
    st.markdown("### Forgetting Rules")
    st.write(f"- **Max Memories/User:** {config.get('max_memories_per_user')}")
    st.write(f"- **Half-life:** {config.get('recency_decay_halflife_days')} days")
    st.write(f"- **Pruning Threshold:** {config.get('relevance_prune_threshold')}")
    st.write(f"- **Min Retrieves to Keep:** {config.get('min_retrieval_count_to_keep')}")

    st.markdown("---")
    st.markdown("### Benchmark Progress")
    for cond in conditions:
        if condition_data[cond] is not None:
            conv_count = len(condition_data[cond].get("conversations", {}))
            status_text = f"✅ {cond} ({conv_count}/20)"
        else:
            status_text = f"⏳ {cond} (Pending)"
        st.write(status_text)
        
    if is_running_benchmark:
        st.warning("Benchmark is currently running or incomplete. Results will update automatically when you refresh.")
        if st.button("Refresh Dashboard"):
            st.rerun()

# ----------------- TABS SETUP -----------------
tab_summary, tab_analytics, tab_explorer, tab_sandbox = st.tabs([
    "📊 Executive Summary", 
    "📈 Detailed Analytics", 
    "🔍 Conversation Explorer", 
    "🎮 Interactive Sandbox"
])

# ----------------- TAB 1: EXECUTIVE SUMMARY -----------------
with tab_summary:
    st.markdown("### 🏆 Memory Condition Performance Comparison")
    st.markdown("This study evaluates conversational agent performance across four memory settings using synthetic test dialogues.")
    
    # Compile comparison dataframe
    rows = []
    for cond in conditions:
        metrics = {}
        # Try getting from summary file first
        if summary and cond in summary:
            metrics = summary[cond]
        elif condition_data[cond] is not None and "aggregated_metrics" in condition_data[cond]:
            metrics = condition_data[cond]["aggregated_metrics"]
        
        # Fallback if metrics are missing but some conversations are evaluated
        if not metrics and condition_data[cond] is not None:
            convs = condition_data[cond].get("conversations", {})
            if convs:
                all_metrics = [c["metrics"] for c in convs.values() if "metrics" in c]
                if all_metrics:
                    for key in all_metrics[0].keys():
                        metrics[key] = np.mean([m[key] for m in all_metrics])
        
        if metrics:
            rows.append({
                "Memory Condition": cond.replace("_", " ").title(),
                "Precision": f"{metrics.get('precision', 0.0):.4f}",
                "Recall": f"{metrics.get('recall', 0.0):.4f}",
                "Hallucination Rate": f"{metrics.get('hallucination_rate', 0.0):.4f}",
                "Personalization Score (1-5)": f"{metrics.get('personalization_score', 1.0):.2f}",
                "Coherence Score (1-5)": f"{metrics.get('coherence_score', 5.0):.2f}",
                "Avg Latency (ms)": f"{metrics.get('avg_latency_ms', 0.0):.1f}"
            })
        else:
            rows.append({
                "Memory Condition": cond.replace("_", " ").title(),
                "Precision": "N/A",
                "Recall": "N/A",
                "Hallucination Rate": "N/A",
                "Personalization Score (1-5)": "N/A",
                "Coherence Score (1-5)": "N/A",
                "Avg Latency (ms)": "N/A"
            })
            
    df_compare = pd.DataFrame(rows)
    st.dataframe(df_compare, use_container_width=True, hide_index=True)
    
    # Metric highlight cards for the selected condition
    st.markdown("---")
    st.markdown("### 🎯 Key Performance Indicator (KPI) Highlights")
    selected_cond_kpi = st.selectbox("Select Memory Condition to highlight:", conditions, format_func=lambda x: x.replace("_", " ").title())
    
    kpi_cols = st.columns(5)
    
    # Extract metrics for selected highlight
    kpi_metrics = {}
    if summary and selected_cond_kpi in summary:
        kpi_metrics = summary[selected_cond_kpi]
    elif condition_data[selected_cond_kpi] is not None and "aggregated_metrics" in condition_data[selected_cond_kpi]:
        kpi_metrics = condition_data[selected_cond_kpi]["aggregated_metrics"]
        
    def display_kpi(col, label, val_format, val, color="#4D96FF"):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {color}">{val_format.format(val) if val is not None else 'N/A'}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
            
    if kpi_metrics:
        display_kpi(kpi_cols[0], "Memory Precision", "{:.2%}", kpi_metrics.get("precision"), color="#4D96FF")
        display_kpi(kpi_cols[1], "Memory Recall", "{:.2%}", kpi_metrics.get("recall"), color="#57CC99")
        display_kpi(kpi_cols[2], "Hallucination Rate", "{:.2%}", kpi_metrics.get("hallucination_rate"), color="#FF6B6B")
        display_kpi(kpi_cols[3], "Personalization", "{:.2f} / 5", kpi_metrics.get("personalization_score"), color="#FF9F43")
        display_kpi(kpi_cols[4], "Coherence Score", "{:.2f} / 5", kpi_metrics.get("coherence_score"), color="#9b5DE5")
    else:
        for c in kpi_cols:
            with c:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value" style="color: #888888">N/A</div>
                    <div class="metric-label">Data Pending</div>
                </div>
                """, unsafe_allow_html=True)

# ----------------- TAB 2: DETAILED ANALYTICS -----------------
with tab_analytics:
    st.markdown("### 📈 Metric Visualizations")
    
    # We will build pandas DataFrames to plot using Streamlit's built-in charts
    chart_data_list = []
    for cond in conditions:
        metrics = None
        if summary and cond in summary:
            metrics = summary[cond]
        elif condition_data[cond] is not None and "aggregated_metrics" in condition_data[cond]:
            metrics = condition_data[cond]["aggregated_metrics"]
            
        if metrics:
            chart_data_list.append({
                "Condition": cond.replace("_", " ").title(),
                "Precision": metrics.get("precision", 0.0),
                "Recall": metrics.get("recall", 0.0),
                "Hallucination Rate": metrics.get("hallucination_rate", 0.0),
                "Personalization Score": metrics.get("personalization_score", 1.0),
                "Coherence Score": metrics.get("coherence_score", 5.0),
                "Latency (ms)": metrics.get("avg_latency_ms", 0.0)
            })
            
    if chart_data_list:
        df_chart = pd.DataFrame(chart_data_list).set_index("Condition")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Retrieval Accuracy (Precision vs Recall)")
            st.bar_chart(df_chart[["Precision", "Recall"]], height=350)
            
            st.markdown("#### Hallucination Rate (Lower is Better)")
            st.bar_chart(df_chart[["Hallucination Rate"]], height=350)
            
        with col2:
            st.markdown("#### LLM Judge Scores (1 - 5)")
            st.bar_chart(df_chart[["Personalization Score", "Coherence Score"]], height=350)
            
            st.markdown("#### Average Latency (ms)")
            st.bar_chart(df_chart[["Latency (ms)"]], height=350)
            
        # Forgetting Curve analysis
        st.markdown("---")
        st.markdown("#### ⏳ Memory Count over Conversation Turns (Forgetting Speed)")
        st.markdown("This chart tracks how memory counts grow and shrink turn-by-turn. Notice how `recency_decay` and `hybrid` strategies manage the vector storage capacity dynamically.")
        
        # Calculate avg memory counts at each turn for each condition
        turn_curves = {}
        for cond in ["no_forgetting", "recency_decay", "hybrid"]:
            if condition_data[cond] is not None:
                convs = condition_data[cond].get("conversations", {})
                if convs:
                    # Collect memories after turn counts
                    counts_matrix = []
                    for conv in convs.values():
                        counts = [t.get("memory_count_after_turn", 0) for t in conv.get("turns", [])]
                        counts_matrix.append(counts)
                    # Handle varying lengths
                    max_len = max(len(c) for c in counts_matrix) if counts_matrix else 0
                    padded_matrix = []
                    for c in counts_matrix:
                        if len(c) < max_len:
                            c = c + [c[-1]] * (max_len - len(c))
                        padded_matrix.append(c)
                    if padded_matrix:
                        avg_counts = np.mean(padded_matrix, axis=0)
                        turn_curves[cond.replace("_", " ").title()] = avg_counts
                        
        if turn_curves:
            df_curves = pd.DataFrame(turn_curves)
            df_curves.index = [f"Turn {i+1}" for i in range(len(df_curves))]
            st.line_chart(df_curves, height=350)
        else:
            st.info("Run the benchmark to generate memory counts curves!")
            
    else:
        st.info("No benchmark results completed yet to build charts.")

# ----------------- TAB 3: CONVERSATION EXPLORER -----------------
with tab_explorer:
    st.markdown("### 🔍 Drill-down Dialogue & Retrieval Inspector")
    
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    
    with col_sel1:
        explorer_cond = st.selectbox("Select Memory Condition:", [c for c in conditions if condition_data[c] is not None])
        
    with col_sel2:
        available_convs = []
        if explorer_cond and condition_data[explorer_cond]:
            available_convs = list(condition_data[explorer_cond].get("conversations", {}).keys())
        explorer_conv_id = st.selectbox("Select Conversation:", available_convs)
        
    if explorer_cond and explorer_conv_id:
        conv_record = condition_data[explorer_cond]["conversations"][explorer_conv_id]
        
        st.markdown(f"**Category:** `{conv_record.get('category')}` | **User ID:** `{conv_record.get('user_id')}`")
        
        # Display aggregated metrics for this specific conversation
        c_metrics = conv_record.get("metrics", {})
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.metric("Precision", f"{c_metrics.get('precision', 0.0):.2%}")
        with col_m2:
            st.metric("Recall", f"{c_metrics.get('recall', 0.0):.2%}")
        with col_m3:
            st.metric("Hallucinations", "Yes" if c_metrics.get("hallucination_rate", 0.0) > 0 else "No")
        with col_m4:
            st.metric("Personalization", f"{c_metrics.get('personalization_score', 1.0):.1f}/5")
        with col_m5:
            st.metric("Coherence", f"{c_metrics.get('coherence_score', 5.0):.1f}/5")
            
        st.markdown("---")
        st.markdown("#### Turn-by-Turn Inspector")
        
        turns = conv_record.get("turns", [])
        for i, turn in enumerate(turns):
            with st.expander(f"Turn {i+1}: {turn.get('user_message')[:50]}...", expanded=(i == 0)):
                col_turn1, col_turn2 = st.columns([2, 1])
                
                with col_turn1:
                    st.markdown("**Dialogue**")
                    st.markdown(f"""
                    <div class="chat-bubble chat-user">
                        <b>User:</b> {turn.get('user_message')}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="chat-bubble chat-assistant">
                        <b>Agent (Ira):</b> {turn.get('response')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_turn2:
                    st.markdown("**Memory Operations**")
                    
                    # Context injected
                    ctx_inj = turn.get("memory_context_injected", "")
                    if ctx_inj:
                        st.markdown(f"""
                        <div class="memory-context-box">
                            <b>Injected Context:</b><br/>
                            {ctx_inj.replace(chr(10), '<br/>')}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.write("❌ *No context injected*")
                        
                    # Memories retrieved
                    m_used = turn.get("memories_used", [])
                    if m_used:
                        st.markdown("🔑 **Retrieved Memories:**")
                        for m in m_used:
                            content = m.get("content", "")
                            st.write(f"- {content}")
                    
                    # Memories stored in this turn
                    m_stored = turn.get("memories_stored", [])
                    if m_stored:
                        st.markdown("💾 **Newly Extracted Memories:**")
                        for m in m_stored:
                            st.write(f"- {m.get('content')}")
                    
                    st.write(f"⏱️ **Latency:** `{turn.get('latency_ms', 0):.0f} ms` | **Count:** `{turn.get('memory_count_after_turn', 0)}`")

# ----------------- TAB 4: INTERACTIVE SANDBOX -----------------
with tab_sandbox:
    st.markdown("### 🎮 Play with MeshMind Live!")
    st.markdown("Interact directly with the memory architecture, test custom inputs, watch memories extract, decay, prune, and retrieve in real-time.")
    
    # Initialize Groq client in Streamlit state
    if "groq_api_key" not in st.session_state:
        from dotenv import load_dotenv
        load_dotenv()
        st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")
        
    api_key_input = st.text_input("Groq API Key (loaded from .env if empty):", value=st.session_state.groq_api_key, type="password")
    
    if api_key_input:
        st.session_state.groq_api_key = api_key_input
        
    if not st.session_state.groq_api_key:
        st.warning("Please enter a Groq API Key or set GROQ_API_KEY in your .env file to run the Sandbox.")
    else:
        # Import core modules
        from memory.memory_manager import MemoryManager
        from conversation.agent import ConversationalAgent
        
        # Setup session state agent & memory manager
        sandbox_user_id = "sandbox_user"
        sandbox_conv_id = "sandbox_conv"
        
        if "sandbox_manager" not in st.session_state:
            # Create a unique sandbox collection
            sandbox_coll = f"meshmind_sandbox_{int(datetime.now().timestamp())}"
            st.session_state.sandbox_manager = MemoryManager(config, collection_name=sandbox_coll)
            st.session_state.sandbox_agent = ConversationalAgent(
                config, 
                st.session_state.sandbox_manager, 
                Groq(api_key=st.session_state.groq_api_key)
            )
            st.session_state.sandbox_history = []
            st.session_state.sandbox_days_offset = 0
            
        mgr = st.session_state.sandbox_manager
        agent = st.session_state.sandbox_agent
        
        col_sb1, col_sb2 = st.columns([2, 1])
        
        with col_sb1:
            st.markdown("#### Chat with Ira (Conversational Agent)")
            
            # Simple custom chat display
            for h in st.session_state.sandbox_history:
                role_class = "chat-user" if h["role"] == "user" else "chat-assistant"
                role_name = "User" if h["role"] == "user" else "Ira"
                st.markdown(f"""
                <div class="chat-bubble {role_class}">
                    <b>{role_name}:</b> {h["content"]}
                </div>
                """, unsafe_allow_html=True)
                
            # Form for user input
            with st.form("chat_form", clear_on_submit=True):
                user_msg = st.text_input("Type something to Ira (e.g. 'I love cooking Italian food on Saturdays. My sister's name is Priya.'):")
                sandbox_condition = st.selectbox("Active Memory Condition:", conditions, index=3) # default hybrid
                submit_button = st.form_submit_button("Send")
                
            if submit_button and user_msg:
                # 1. Add user message to history
                st.session_state.sandbox_history.append({"role": "user", "content": user_msg})
                
                # 2. Get history context
                history_turns = []
                for h in st.session_state.sandbox_history[:-1]:
                    history_turns.append(f"{h['role']}: {h['content']}")
                    
                # 3. Call Chat Agent
                response_pkg = agent.chat(
                    user_message=user_msg,
                    user_id=sandbox_user_id,
                    conversation_id=sandbox_conv_id,
                    memory_condition=sandbox_condition
                )
                
                # 4. Extract new memories (if memory is enabled)
                new_mems = []
                if sandbox_condition != "no_memory":
                    new_mems = mgr.extractor.extract_memories(
                        conversation_turn=user_msg,
                        context="\n".join(history_turns),
                        user_id=sandbox_user_id,
                        conversation_id=sandbox_conv_id
                    )
                    
                    # Enforce limit
                    max_memories = config.get("max_memories_per_user", 100)
                    for m in new_mems:
                        if mgr.store.get_memory_count(sandbox_user_id) >= max_memories:
                            all_m = mgr.store.get_all_memories(sandbox_user_id)
                            if all_m:
                                mgr.store.delete_memory(all_m[-1].memory_id)
                        mgr.store.add_memory(m)
                        
                # Save details of the last operation to session state
                st.session_state.last_op = {
                    "extracted": [m.content for m in new_mems],
                    "retrieved": [m.content for m in response_pkg.memories_used],
                    "context_injected": response_pkg.memory_context_injected
                }
                
                # Add assistant response to history
                st.session_state.sandbox_history.append({"role": "assistant", "content": response_pkg.response})
                st.rerun()
                
            # Clear Chat Button
            if st.button("Reset Chat Session"):
                st.session_state.sandbox_history = []
                st.session_state.last_op = None
                # Clear Chroma DB sandbox collection
                try:
                    mgr.store.collection.delete(where={"user_id": sandbox_user_id})
                except Exception:
                    pass
                st.rerun()
                
        with col_sb2:
            st.markdown("#### 🛠️ Vector Memory DB Inspector")
            
            # Action: Simulate Passage of Time
            st.markdown("**Simulate Time Passing**")
            shift_days = st.number_input("Days to advance forward:", min_value=1, max_value=30, value=4)
            if st.button(f"Advance {shift_days} Days"):
                # Shift all memories in ChromaDB back by shift_days
                mems = mgr.store.get_all_memories(sandbox_user_id)
                for m in mems:
                    dt = datetime.fromisoformat(m.created_at)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    shifted = dt - timedelta(days=shift_days)
                    mgr.store.update_metadata(m.memory_id, {"created_at": shifted.isoformat()})
                
                # Apply Forgetting (decay and/or prune if hybrid selected)
                # We can trigger memory decay/pruning by calling forgetter manually
                if "sandbox_condition" in st.session_state:
                    cond = st.session_state.sandbox_condition
                else:
                    cond = "hybrid"
                mgr.forgetter.forget(user_id=sandbox_user_id, strategy=cond)
                
                st.success(f"Shifted timestamps and executed '{cond}' forgetting logic!")
                st.rerun()
                
            # List current memories in ChromaDB
            mems = mgr.store.get_all_memories(sandbox_user_id)
            st.write(f"**Current Memory Count:** `{len(mems)}`")
            
            if mems:
                st.write("---")
                for m in mems:
                    with st.container():
                        # Calculate current decay weight
                        created_dt = datetime.fromisoformat(m.created_at)
                        if created_dt.tzinfo is None:
                            created_dt = created_dt.replace(tzinfo=timezone.utc)
                        now_dt = datetime.now(timezone.utc)
                        days_passed = (now_dt - created_dt).total_seconds() / 86400.0
                        
                        st.markdown(f"**Memory:** {m.content}")
                        st.markdown(
                            f"<span class='badge badge-blue'>Importance: {m.importance_score:.2f}</span>"
                            f"<span class='badge badge-orange'>Decay Wt: {m.decay_weight:.3f}</span>"
                            f"<span class='badge badge-green'>Retrieved: {m.retrieval_count}x</span>",
                            unsafe_allow_html=True
                        )
                        st.caption(f"Created: {created_dt.strftime('%Y-%m-%d %H:%M:%S')} ({days_passed:.1f} days ago)")
                        st.write("---")
            else:
                st.caption("No memories stored yet. Talk to Ira to create memories!")
                
            # Display last operation details
            if "last_op" in st.session_state and st.session_state.last_op:
                op = st.session_state.last_op
                st.markdown("#### Last Turn Details")
                if op["extracted"]:
                    st.write("**Extracted:**", op["extracted"])
                if op["retrieved"]:
                    st.write("**Retrieved:**", op["retrieved"])
                if op["context_injected"]:
                    st.code(op["context_injected"], language="markdown")
