# 🚏 LocalSearchBench

Benchmarking agentic local-life search with nine high-density Chinese cities.

## 🌐 Quick Links

- **Demo Website**: [localsearchbench.github.io](https://localsearchbench.github.io)
- **RAG Server**: See server README for backend deployment

## 📝 Abstract
Recent advances in large reasoning models (LRMs) have enabled agentic search systems to perform complex multi-step reasoning across multiple sources. However, most studies focus on general information retrieval and rarely explore vertical domains with unique challenges. In this work, we focus on local life services and introduce **LocalSearchBench**, which encompasses diverse and complex business scenarios. Real-world queries in this domain are often ambiguous and require multi-hop reasoning across merchants and products, remaining challenging and not fully addressed. As the first comprehensive benchmark for agentic search in local life services, LocalSearchBench comprises a database of over **1.3M** merchant entries across **6** service categories and **9** major cities, and **900** multi-hop QA tasks from real user queries that require multi-step reasoning. We also developed **LocalPlayground**, a unified environment integrating multiple tools for LRMs interaction. Experiments show that even state-of-the-art LRMs struggle on LocalSearchBench: the best model (DeepSeek-V3.2) achieves only **35.60%** correctness, and most models have issues with completeness (average 60.32%) and faithfulness (average 30.72%). This highlights the need for specialized benchmarks and domain-specific agent training in local life services. Code, Benchmark, and Leaderboard are available at [localsearchbench.github.io](https://localsearchbench.github.io).

## ⭐ Key Features
- **Six core scenarios**: Dining, entertainment, shopping, hotel, travel errands, and mixed lifestyle tasks reflect the platform's dominant traffic mix.
- **Nine major cities**: Shanghai, Beijing, Guangzhou, Shenzhen, Hangzhou, Suzhou, Chengdu, Chongqing, and Wuhan ensure geographic and economic diversity.
- **Multi-hop supervision**: Each QA example bundles a full `Multi-hop search path` trace that records tool calls and intermediate evidence.
- **Difficulty grading (L3–L4)**: First industry-aligned grading system for local-life agentic search, measuring requirement complexity and plan length.
- **Merchant-scale grounding**: 1.3M merchants with anonymization, augmentation, and quality filtering support faithful retrieval.
- **Interactive Demo**: Web-based interface for testing
- **GPU-Accelerated RAG**: VLLM-powered embedding and reranking


## 📊 Benchmark At a Glance
- **Merchant database**: 1,354,185 POIs collected via multi-agent crawling and QA.
- **QA tasks**: 900 samples → 100 tasks per city, 3–5 hops each.
- **Models evaluated**: Qwen3 series, GPT-4.1, o3, Gemini-2.5 series, LongCat series, GLM-4.6, DeepSeek-V3.2, etc.
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
| DeepSeek-V3.2 (w/ thinking) | 3.21 | 4.20 | 35.60 | 77.56 | 70.92 | 39.78 | 81.13 |
| GLM-4.6 (w/ thinking) | 3.08 | 4.06 | 32.83 | 76.83 | 70.27 | 37.48 | 81.30 |
| DeepSeek-V3.2 (w/o thinking) | 3.12 | 4.11 | 32.74 | 77.08 | 70.12 | 35.96 | 81.52 |
| Gemini-2.5-Pro | 2.75 | 3.10 | 32.34 | 71.03 | 71.12 | 34.93 | 82.32 |
| o3(high) | 2.91 | 3.38 | 31.53 | 69.64 | 70.72 | 33.89 | 81.87 |
| LongCat-Flash-Thinking | 3.04 | 3.20 | 30.68 | 68.83 | 69.07 | 31.47 | 80.10 |
| Qwen3-235B-A22B (w/ thinking) | 2.31 | 3.17 | 30.24 | 71.20 | 71.58 | 26.90 | 81.76 |
| GLM-4.6 (w/o thinking) | 2.86 | 3.86 | 28.97 | 76.45 | 70.37 | 35.40 | 81.40 |
| LongCat-Flash-Chat | 2.34 | 3.07 | 25.28 | 52.98 | 69.45 | 27.49 | 83.61 |
| Qwen3-32B (w/ thinking) | 2.80 | 3.12 | 25.63 | 40.66 | 68.44 | 22.40 | 79.54 |
| Qwen3-14B (w/ thinking) | 2.57 | 2.12 | 25.17 | 40.98 | 69.32 | 28.40 | 80.44 |
| Qwen3-14B (w/o thinking) | 2.53 | 2.07 | 24.21 | 40.60 | 69.62 | 27.44 | 80.78 |
| Gemini-2.5-Flash | 1.84 | 2.51 | 20.97 | 58.44 | 68.04 | 35.72 | 79.70 |
| Qwen3-235B-A22B (w/o thinking) | 2.00 | 2.93 | 21.18 | 50.94 | 69.16 | 25.28 | 79.72 |
| Qwen3-32B (w/o thinking) | 2.78 | 3.11 | 19.76 | 40.96 | 68.50 | 21.38 | 80.76 |
| GPT-4.1 | 1.73 | 2.42 | 18.47 | 45.37 | 65.93 | 28.76 | 77.38 |

### 🎯 Trajectory Effectiveness Metrics
| Model | Action Relevance | Evidence Sufficiency | Causal Coherence | Search Efficiency |
| --- | --- | --- | --- | --- |
| Qwen3-14B (w/ thinking) | 81.19 | 47.24 | 52.52 | 51.65 |
| Qwen3-235B-A22B (w/ thinking) | 80.42 | 45.75 | 52.04 | 48.63 |
| Qwen3-14B (w/o thinking) | 80.46 | 46.44 | 50.96 | 50.02 |
| LongCat-Flash-Thinking | 78.50 | 47.37 | 53.18 | 53.27 |
| LongCat-Flash-Chat | 77.78 | 47.33 | 50.86 | 52.29 |
| GLM-4.6 (w/ thinking) | 77.66 | 48.90 | 52.67 | 54.43 |
| Gemini-2.5-Pro | 77.31 | 45.73 | 52.87 | 41.78 |
| DeepSeek-V3.2 (w/ thinking) | 75.58 | 48.86 | 52.62 | 54.83 |
| Qwen3-32B (w/ thinking) | 75.06 | 46.99 | 48.87 | 49.13 |
| o3(high) | 75.93 | 44.71 | 51.78 | 42.96 |
| DeepSeek-V3.2 (w/o thinking) | 75.51 | 48.49 | 52.23 | 54.33 |
| Qwen3-235B-A22B (w/o thinking) | 75.22 | 43.61 | 50.68 | 45.99 |
| Qwen3-32B (w/o thinking) | 74.67 | 46.52 | 48.82 | 49.84 |
| GLM-4.6 (w/o thinking) | 74.28 | 48.44 | 50.79 | 52.76 |
| Gemini-2.5-Flash | 70.58 | 41.61 | 48.94 | 46.91 |
| GPT-4.1 | 68.47 | 38.62 | 45.83 | 42.29 |


## ✅ License
MIT License. Commercial and research use permitted with attribution to LocalSearchBench.
