
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os

st.set_page_config(page_title="LLM Benchmarking Dashboard", page_icon="🚀", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #6b7280; font-size: 1.1rem; margin-top: 0; }
    .metric-card {
        background: linear-gradient(135deg, #f5f3ff, #fdf2f8);
        border-radius: 16px; padding: 20px; text-align: center;
        border: 1px solid #e5e7eb;
    }
    .insight-box {
        background-color: #f0f9ff; border-left: 5px solid #3b82f6;
        padding: 15px 20px; border-radius: 8px; margin: 10px 0;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- Check required files exist before loading ----------
required_files = [
    "model_architecture_comparison.csv",
    "benchmark_baseline_results.csv",
    "quality_scores_summary.csv"
]

missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error("Missing required data files:")
    for f in missing_files:
        st.write(f"- {f}")
    st.write("Files found in this folder:")
    st.write(os.listdir("."))
    st.stop()

# ---------- Load data ----------
df_architecture = pd.read_csv("model_architecture_comparison.csv")
df_speed = pd.read_csv("benchmark_baseline_results.csv")
df_quality = pd.read_csv("quality_scores_summary.csv")

GPU_COST_PER_HOUR = 1.00
df_speed["Cost per 1000 tokens (USD)"] = ((1000 / df_speed["Tokens/sec"]) / 3600 * GPU_COST_PER_HOUR).round(6)

master_df = df_architecture.copy()
master_df = master_df.merge(
    df_speed[["Model", "Tokens/sec", "GPU Memory (MB)", "Generation Time (s)", "Cost per 1000 tokens (USD)"]],
    on="Model", how="left"
)
master_df = master_df.merge(df_quality, on="Model", how="left")
master_df = master_df.rename(columns={
    "Generation Time (s)": "Latency (s)",
    "Max Context Length": "Context Length (tokens)"
})

def plot_metric(metric_col, title, color, ylabel):
    fig, ax = plt.subplots(figsize=(9, 5))
    data = master_df.dropna(subset=[metric_col]).sort_values(metric_col, ascending=False)
    bars = ax.bar(data["Model"], data[metric_col], color=color, edgecolor="white", linewidth=1.5)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:,.1f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig

st.markdown('<p class="main-header">LLM Benchmarking Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A hands-on comparison of open-source LLMs across speed, memory, quality, context length, latency, and cost.</p>', unsafe_allow_html=True)
st.markdown("---")

fastest_row = master_df.loc[master_df["Tokens/sec"].idxmax()]
lightest_row = master_df.loc[master_df["GPU Memory (MB)"].idxmin()]
quality_row = master_df.loc[master_df["Average Quality Score"].idxmax()]
cheapest_row = master_df.loc[master_df["Cost per 1000 tokens (USD)"].idxmin()]

fastest_name = fastest_row["Model"]
fastest_speed = fastest_row["Tokens/sec"]
lightest_name = lightest_row["Model"]
lightest_mem = lightest_row["GPU Memory (MB)"]
quality_name = quality_row["Model"]
quality_score = quality_row["Average Quality Score"]
cheapest_name = cheapest_row["Model"]
cheapest_cost = cheapest_row["Cost per 1000 tokens (USD)"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card">⚡<br><b>Fastest</b><br>{fastest_name}<br><small>{fastest_speed:.1f} tok/s</small></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card">💾<br><b>Lightest</b><br>{lightest_name}<br><small>{lightest_mem:.0f} MB</small></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card">⭐<br><b>Best Quality</b><br>{quality_name}<br><small>{quality_score:.1f}/10</small></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card">💰<br><b>Cheapest</b><br>{cheapest_name}<br><small>${cheapest_cost:.6f}/1K tok</small></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tabs = st.tabs(["Full Table", "Speed", "Memory", "Context Length", "Latency", "Cost", "Quality", "Recommend Me a Model", "About"])

with tabs[0]:
    st.subheader("Full Architecture & Benchmark Comparison")
    st.dataframe(master_df, use_container_width=True)
    st.caption("Data collected via Hugging Face configs and live benchmarking on Colab T4 GPU.")

with tabs[1]:
    st.subheader("Inference Speed (Tokens/sec)")
    st.pyplot(plot_metric("Tokens/sec", "Inference Speed", "#6366f1", "Tokens/sec"))
    st.markdown(f'<div class="insight-box">{fastest_name} generates text the fastest, making it the best pick for real-time or high-throughput applications.</div>', unsafe_allow_html=True)

with tabs[2]:
    st.subheader("GPU Memory Usage")
    st.pyplot(plot_metric("GPU Memory (MB)", "GPU Memory Usage", "#f59e0b", "MB"))
    st.markdown(f'<div class="insight-box">{lightest_name} has the smallest memory footprint, ideal for limited-VRAM environments.</div>', unsafe_allow_html=True)

with tabs[3]:
    st.subheader("Maximum Context Length")
    st.pyplot(plot_metric("Context Length (tokens)", "Max Context Length", "#10b981", "Tokens"))
    st.markdown('<div class="insight-box">A longer context window means the model can process longer documents without losing earlier information.</div>', unsafe_allow_html=True)

with tabs[4]:
    st.subheader("Response Latency")
    st.pyplot(plot_metric("Latency (s)", "Response Latency", "#ef4444", "Seconds"))
    st.markdown('<div class="insight-box">Lower latency means faster response times for end users.</div>', unsafe_allow_html=True)

with tabs[5]:
    st.subheader("Estimated Cost per 1,000 Tokens")
    st.pyplot(plot_metric("Cost per 1000 tokens (USD)", "Cost Estimation", "#eab308", "USD per 1000 tokens"))
    st.caption("Estimated using a typical cloud GPU rental rate of $1.00/hour.")

with tabs[6]:
    st.subheader("Response Quality (LLM-as-Judge Scored)")
    st.pyplot(plot_metric("Average Quality Score", "Response Quality", "#a855f7", "Score (1-10)"))
    st.markdown(f'<div class="insight-box">{quality_name} produced the most accurate and coherent answers, judged automatically across factual, reasoning, coding, and instruction-following prompts.</div>', unsafe_allow_html=True)

with tabs[7]:
    st.subheader("Not sure which model to pick? Let the data decide.")
    priority = st.selectbox("What matters most to you?", ["Fastest", "Lowest Memory", "Best Quality", "Cheapest"])

    if priority == "Fastest":
        st.success(f"Winner: {fastest_name} — {fastest_speed:.1f} tokens/sec. Best for real-time, high-volume use cases.")
    elif priority == "Lowest Memory":
        st.success(f"Winner: {lightest_name} — {lightest_mem:.0f} MB. Best for constrained hardware or edge deployment.")
    elif priority == "Best Quality":
        st.success(f"Winner: {quality_name} — {quality_score:.1f}/10 quality score. Best when accuracy matters most.")
    elif priority == "Cheapest":
        st.success(f"Winner: {cheapest_name} — ${cheapest_cost:.6f} per 1000 tokens. Best for cost-sensitive, large-scale deployments.")

with tabs[8]:
    st.subheader("About this project")
    st.markdown("""
    This dashboard was built as part of a Week 1 deep dive into Transformer architecture and LLM internals, covering:

    - Core concepts: tokenization, embeddings, attention, KV cache, quantization
    - A Transformer decoder block built and trained from scratch in PyTorch
    - Real architecture comparison across Llama, Mistral, Qwen, Gemma, DeepSeek, and Phi
    - Live benchmarking of speed, memory, and quantization tradeoffs
    - Automated quality scoring using an LLM-as-judge approach
    - This interactive dashboard, tying it all together

    Built with: PyTorch, Hugging Face Transformers, BitsAndBytes, Streamlit
    """)
    st.caption("Made as a personal learning project.")

st.markdown("---")
st.markdown(
    "<p style=\"text-align: center; color: #9ca3af; font-size: 0.9rem;\">Made by <b>Zuha Junaid</b></p>",
    unsafe_allow_html=True
)
