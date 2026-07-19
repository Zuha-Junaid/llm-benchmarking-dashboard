# 🚀 LLM Benchmarking Dashboard

An interactive dashboard for comparing open-source Large Language Models (LLMs) across speed, memory usage, response quality, context length, latency, and cost — built as part of a hands-on deep dive into Transformer architecture and LLM internals.

**🔗 Live App:** [Add your Streamlit Cloud link here]

---

## 📖 About This Project

This project was built as a Week 1 exploration of how modern LLMs work internally — starting from core Transformer concepts, building a decoder block from scratch, and ending with a full benchmarking dashboard comparing real open-source models.

The goal was to move beyond just *reading* about LLM internals, and instead *prove* understanding by building, training, benchmarking, and visualizing every concept hands-on.

---

## ✨ What This Dashboard Shows

The dashboard compares the following open-source models:

- **Qwen2.5 1.5B**
- **Phi-2**
- **DeepSeek-Coder 1.3B**
- **Gemma 2 2B**

Across six key metrics:

| Metric | What it measures |
|---|---|
| ⚡ **Inference Speed** | Tokens generated per second |
| 💾 **GPU Memory Usage** | Memory footprint during generation |
| ⭐ **Response Quality** | Automated scoring via an LLM-as-judge approach |
| 📏 **Context Length** | Maximum tokens the model can process at once |
| ⏱️ **Latency** | Time taken to generate a response |
| 💰 **Cost Estimation** | Estimated cost per 1,000 tokens on typical cloud GPU pricing |

It also includes a **"Recommend Me a Model"** tool — pick what matters most to you (speed, memory, quality, or cost), and it suggests the best-fitting model based on the collected data.

---

## 🧠 What Was Built Along the Way

This dashboard is the final piece of a larger learning project. The full journey included:

1. **Core LLM Concepts** — tokenization, embeddings, positional encoding, self-attention, multi-head attention, KV cache, context windows, and quantization, each explained with working code demos.
2. **A Transformer Decoder Block Built From Scratch** — implemented in PyTorch, trained on a small text corpus, with generated text and attention-weight visualizations.
3. **Architecture Comparison** — real configuration data pulled directly from Hugging Face for Llama, Mistral, Qwen, Gemma, DeepSeek, and Phi, comparing attention type (MHA vs. GQA), activation functions, vocabulary size, and context length.
4. **Benchmarking** — actual measured inference speed, GPU memory usage, and the effects of 8-bit and 4-bit quantization using BitsAndBytes.
5. **Memory Profiling & GPU Optimization** — peak memory scaling with input length, batching efficiency tests, and `torch.compile` speedup testing.
6. **LLM-as-Judge Quality Scoring** — using a separate LLM to automatically score each model's answers across factual, reasoning, coding, and instruction-following prompts.
7. **This Dashboard** — tying every result together into one interactive tool.

---

## 🛠️ Tech Stack

- **PyTorch** — building and training the decoder from scratch
- **Hugging Face Transformers** — loading and running open-source models
- **BitsAndBytes** — quantization (8-bit / 4-bit)
- **Streamlit** — the interactive dashboard
- **Pandas / Matplotlib** — data handling and visualization
- **Google Colab** — development and benchmarking environment (free T4 GPU)

---

## 📂 Repository Structure

```
├── app.py                                  # Streamlit dashboard app
├── requirements.txt                        # Python dependencies
├── model_architecture_comparison.csv       # Architecture comparison data
├── benchmark_baseline_results.csv          # Speed & memory benchmark data
├── quality_scores_summary.csv              # LLM-as-judge quality scores
└── README.md                               # This file
```

---

## 🚀 Running Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/zuha-junaid/llm-benchmarking-dashboard.git
   cd llm-benchmarking-dashboard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in your terminal (usually `http://localhost:8501`)

---

## ⚠️ Notes & Limitations

- Cost estimates assume a flat $1.00/hour cloud GPU rate — actual costs vary by provider and GPU type.
- Quality scores were generated using a free, locally-run LLM judge rather than a paid frontier model API, which may be less consistent than a top-tier judge model.
- Benchmarks were run on a free-tier Colab T4 GPU; results may differ on other hardware.
- Models tested were kept to the 1-3B parameter range to fit within free-tier GPU memory limits.

---

## 🙋 About Me

Made by **Zuha Junaid** as a personal learning project exploring how modern LLMs work under the hood.

Feedback and suggestions are welcome!
