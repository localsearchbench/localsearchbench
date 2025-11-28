# 🚏 LocalSearchBench

Benchmarking agentic local-life search with nine high-density Chinese cities.


## 📝 Abstract
Recent advances in large reasoning models (LRMs) have enabled agentic search systems to perform complex multi-step reasoning across multiple sources. However, most research focuses on general information retrieval and rarely explores vertical domains with unique challenges. In this work, we focus on local life services and introduce LocalSearchBench, which encompasses diverse and complex business scenarios. Real-world queries in this domain are often ambiguous and require multi-hop reasoning across merchants and products, remaining challenging and not fully addressed. As the first comprehensive benchmark for agentic search in local life services, LocalSearchBench includes over **1,354,185** high-quality merchant entries spanning six business scenarios and nine cities. We construct **900** multi-hop QA tasks (3–5 hops, L3–L4 difficulty) based on real user intents, challenging agents to decompose requirements, retrieve information, and synthesize itineraries. We also developed **LocalPlayground**, a unified evaluation interface integrating retrieval, reranking, and tool-use traces. Experiments show that even state-of-the-art LRMs struggle on LocalSearchBench: the best model (DeepSeek-V3.1) reaches only **34.34% correctness**, with average completeness and faithfulness below 80% and 62%, respectively. This highlights the need for specialized benchmarks and domain-specific agent training in local life services.

## ⭐ Key Features
- **Six core scenarios**: Dining, entertainment, shopping, hotel, travel errands, and mixed lifestyle tasks reflect the platform's dominant traffic mix.
- **Nine major cities**: Shanghai, Beijing, Guangzhou, Shenzhen, Hangzhou, Suzhou, Chengdu, Chongqing, and Wuhan ensure geographic and economic diversity.
- **Multi-hop supervision**: Each QA example bundles a full `Multi-hop search path` trace that records tool calls and intermediate evidence.
- **Difficulty grading (L3–L4)**: First industry-aligned grading system for local-life agentic search, measuring requirement complexity and plan length.
- **Merchant-scale grounding**: 1.35M merchants with anonymization, augmentation, and quality filtering support faithful retrieval.


## 📊 Benchmark At a Glance
- **Merchant database**: 1,354,185 POIs collected via multi-agent crawling and QA.
- **QA tasks**: 900 samples → 100 tasks per city, 3–5 hops each.
- **Models evaluated**: GPT-4.1, Gemini-2.5-Pro, Qwen3 series, GLM-4.5, DeepSeek-V3.1, LongCat-Large, Hunyuan-T1, etc.
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
| Model | Avg tool calls | Avg rounds | Correctness | Completeness | Faithfulness |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-V3.1 | 3.43 | 4.02 | **34.34** | 80.00 | 60.80 |
| Qwen-Plus-Latest | 2.59 | 3.12 | 32.79 | **80.94** | 68.68 |
| LongCat-Large-32K | 2.73 | 3.22 | 33.19 | 80.51 | 60.80 |
| GPT-4.1 | 1.72 | 2.70 | 26.76 | 75.42 | 72.63 |
| Gemini-2.5-Pro | 1.89 | 2.86 | 26.09 | 77.93 | **78.26** |

## ✅ License
MIT License. Commercial and research use permitted with attribution to LocalSearchBench.