# 🚏 LocalSearchBench

Benchmarking agentic local-life search with nine high-density Chinese cities.

## 🌐 Quick Links

- **Demo Website**: [localsearchbench.github.io](https://localsearchbench.github.io)
- **Paper**: Coming soon
- **RAG Server**: See server README for backend deployment
- **MCP Tools**: Use as AI tool in Claude Desktop, Cursor, and more!

## 📝 Abstract
Recent advances in large reasoning models (LRMs) have enabled agentic search systems to perform complex multi-step reasoning across multiple sources. However, most studies focus on general information retrieval and rarely explores vertical domains with unique challenges. In this work, we focus on local life services and introduce **LocalSearchBench**, which encompasses diverse and complex business scenarios. Real-world queries in this domain are often ambiguous and require multi-hop reasoning across merchants and products, remaining challenging and not fully addressed. As the first comprehensive benchmark for agentic search in local life services, LocalSearchBench comprises a database of over **1.3M** merchant entries across **6** service categories and **9** major cities, and **900** multi-hop QA tasks from real user queries that require multi-step reasoning. We also developed **LocalPlayground**, a unified environment integrating retrieval, reranking, and tool-use traces for comprehensive evaluation. Experiments show that even state-of-the-art LRMs struggle on LocalSearchBench: the best model (Deepseek-V3.2) reaches only **32.93%** correctness, highlighting the need for specialized benchmarks and domain-specific agent training in local life services.

## ⭐ Key Features
- **Six core scenarios**: Dining, entertainment, shopping, hotel, travel errands, and mixed lifestyle tasks reflect the platform's dominant traffic mix.
- **Nine major cities**: Shanghai, Beijing, Guangzhou, Shenzhen, Hangzhou, Suzhou, Chengdu, Chongqing, and Wuhan ensure geographic and economic diversity.
- **Multi-hop supervision**: Each QA example bundles a full `Multi-hop search path` trace that records tool calls and intermediate evidence.
- **Difficulty grading (L3–L4)**: First industry-aligned grading system for local-life agentic search, measuring requirement complexity and plan length.
- **Merchant-scale grounding**: 1.3M merchants with anonymization, augmentation, and quality filtering support faithful retrieval.
- **Interactive Demo**: Web-based interface for testing
- **GPU-Accelerated RAG**: VLLM-powered embedding and reranking
- **MCP Integration**: Use as AI tool in Claude Desktop, Cursor, and more!

## 📊 Benchmark At a Glance
- **Merchant database**: 1,354,185 POIs collected via multi-agent crawling and QA.
- **QA tasks**: 900 samples → 100 tasks per city, 3–5 hops each.
- **Models evaluated**: Qwen3 series, GPT-4.1, o3, Gemini-2.5 series, LongCat series, GLM-4.6, Deepseek-V3.2, etc.
- **Metrics**: Correctness, completeness, fluency, faithfulness, safety, avg. tool calls, avg. rounds (matches leaderboard section on the website).

## 🧱 Data Fields
| Field | Type | Description |
| --- | --- | --- |
| `Hop Count` | `int64` | Number of chained search hops used to solve the query (3–5). |
| `Difficulty` | `string` | L3 (standard multi-hop) or L4 (complex constraints). |
| `City` | `string` | One of the nine supported Chinese cities. |
| `Question` | `string` | Natural-language request describing the user's goal. |
| `Multi-hop search path` | `string` | Serialized trace of each hop's query + retrieved evidence. |
| `Answer` | `string` | Final recommendation sequence with addresses, hours, and pricing. |


## 🚀 Sample Usage
```python
from datasets import load_dataset

ds = load_dataset("localsearchbench/localsearchbench")
print(ds["train"][0]["Question"])
print(ds["train"][0]["Multi-hop search path"])
```


## 🏅 Leaderboard Snapshot

### 📊 Answer Quality Metrics
| Model | Avg tool calls | Avg rounds | Correctness | Completeness | Fluency | Faithfulness | Safety |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Deepseek-V3.2 (w/ thinking)** | 3.21 | 4.20 | **32.93** | 77.63 | 71.01 | 39.87 | 81.22 |
| **GLM-4.6 (w/ thinking)** | 3.08 | 4.06 | 32.83 | 76.83 | 70.27 | 37.48 | 81.30 |
| **Gemini-2.5-Pro** | 2.75 | 3.10 | 32.41 | 71.10 | 71.21 | 35.02 | 82.41 |
| Deepseek-V3.2 (w/o thinking) | 3.12 | 4.11 | 32.81 | 77.15 | 70.21 | 36.05 | 81.61 |
| o3(high) | 2.91 | 3.38 | 31.60 | 69.71 | 70.80 | 33.98 | 81.96 |
| GLM-4.6 (w/o thinking) | 2.86 | 3.86 | 28.97 | 76.45 | 70.37 | 35.40 | 81.40 |
| LongCat-Flash-Thinking | 3.04 | 3.20 | 30.68 | 68.83 | 69.07 | 31.47 | 80.10 |
| Qwen3-235B-A22B (w/ thinking) | 2.31 | 3.17 | 30.24 | 71.20 | 71.58 | 26.90 | 81.76 |
| LongCat-Flash-Chat | 2.34 | 3.07 | 25.28 | 52.98 | 69.45 | 27.49 | 83.61 |
| Qwen3-14B (w/ thinking) | 2.57 | 2.12 | 25.17 | 40.98 | 69.32 | 28.40 | 80.44 |
| Qwen3-32B (w/ thinking) | 2.80 | 3.12 | 25.63 | 40.66 | 68.44 | 22.40 | 79.54 |
| Qwen3-14B (w/o thinking) | 2.53 | 2.07 | 24.21 | 40.60 | 69.62 | 27.44 | 80.78 |
| Gemini-2.5-Flash | 1.84 | 2.51 | 21.04 | 58.51 | 68.13 | 35.81 | 79.79 |
| Qwen3-235B-A22B (w/o thinking) | 2.00 | 2.93 | 21.18 | 50.94 | 69.16 | 25.28 | 79.72 |
| Qwen3-32B (w/o thinking) | 2.78 | 3.11 | 19.76 | 40.96 | 68.50 | 21.38 | 80.76 |
| GPT-4.1 | 1.73 | 2.42 | 18.56 | 45.44 | 66.02 | 28.85 | 77.47 |

### 🎯 Trajectory Effectiveness Metrics
| Model | Action Relevance | Evidence Sufficiency | Causal Coherence | Search Efficiency |
| --- | --- | --- | --- | --- |
| Deepseek-V3.2 (w/ thinking) | **81.53** | 49.95 | 53.70 | 51.04 |
| GLM-4.6 (w/ thinking) | 80.88 | 47.54 | 52.83 | 51.09 |
| Gemini-2.5-Pro | 79.47 | 47.68 | 52.52 | 50.22 |
| o3(high) | 78.68 | 46.47 | 51.10 | 47.95 |
| Qwen3-235B-A22B (w/ thinking) | 80.40 | 45.75 | 52.04 | 48.63 |
| LongCat-Flash-Thinking | 75.90 | 43.52 | 48.96 | 46.56 |
| GPT-4.1 | 74.28 | 42.53 | 48.10 | 44.83 |

## 📖 Citation
```bibtex
@article{localsearchbench2025,
  title={LocalSearchBench: A Benchmark for Local Search and Recommendation},
  author={Your Name},
  year={2025}
}
```

## ✅ License
MIT License. Commercial and research use permitted with attribution to LocalSearchBench.
