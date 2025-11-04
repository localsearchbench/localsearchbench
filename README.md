---
dataset_info:
  features:
  - name: Hop Count
    dtype: int64
  - name: Difficulty
    dtype: string
  - name: Question
    dtype: string
  - name: Multi-hop search path
    dtype: string
  - name: Answer
    dtype: string
  splits:
  - name: train
    num_bytes: 229077
    num_examples: 47
  download_size: 132077
  dataset_size: 229077
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
