#!/bin/bash

# Step 0: Prepare ICRL JSONL Dataset
# This script helps convert your data to the ICRL format

echo "Converting data to ICRL JSONL format..."

python3 ../../icrl_data_converter.py \
  --input_path "input_data.csv" \
  --output_path "input_jsonl/icrl_data.jsonl" \
  --k_exemplars 3 \
  --n_queries 1 \
  --task_type "general"

echo "ICRL JSONL data prepared in input_jsonl/icrl_data.jsonl"
echo "Please verify the format matches the documentation before proceeding to step 1." 