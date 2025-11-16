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
    num_bytes: 338979
    num_examples: 100
  download_size: 169686
  dataset_size: 338979
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
