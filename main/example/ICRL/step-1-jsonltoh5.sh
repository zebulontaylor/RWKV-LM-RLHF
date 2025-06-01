#!/bin/bash

# Step 1: Convert ICRL JSONL to H5 format
echo "Converting ICRL JSONL to H5 format..."

python3 ../../icrl_generate_h5.py \
  --input_jsonl "input_jsonl/icrl_data.jsonl" \
  --output_h5 "output_h5/icrl_episodes.h5" \
  --tokenizer_path "../../tokenizer/world" \
  --ctx_len 4096 \
  --k_exemplars 3 \
  --n_queries 1

echo "H5 dataset created: output_h5/icrl_episodes.h5" 