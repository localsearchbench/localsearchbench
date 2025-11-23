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
    num_bytes: 4139365
    num_examples: 1751
  download_size: 1427095
  dataset_size: 4139365
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
