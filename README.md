---
dataset_info:
  features:
  - name: Hop Count
    dtype: int64
  - name: Difficulty
    dtype: string
  - name: City
    dtype: string
  - name: Question
    dtype: string
  - name: Multi-hop search path
    dtype: string
  - name: Answer
    dtype: string
  splits:
  - name: train
    num_bytes: 2126889
    num_examples: 900
  download_size: 789091
  dataset_size: 2126889
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
