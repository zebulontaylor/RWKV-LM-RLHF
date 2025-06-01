#!/bin/bash

# Step 5: Test-time Adaptive Optimization (Optional Stage 5)
# Per-request adaptation with synthetic rollouts (TAO style)

echo "Starting Test-time Adaptive Optimization..."

python3 ../../icrl_test_time_adapt.py \
  --model_path "output_save/icrl_ppo/rwkv-50.pth" \
  --reward_model_path "output_save/reward_model/rwkv-10.pth" \
  --test_queries "input_jsonl/test_queries.jsonl" \
  --output_dir "output_save/test_time_adapt" \
  --ctx_len 4096 \
  --k_exemplars 3 \
  --adapt_steps 3 \
  --adapt_lr 1e-6 \
  --gen_count 2 \
  --gen_length 256 \
  --gen_temperature 1.0 \
  --gen_topp 0.9 \
  --value_head_only 1

echo "Test-time adaptation completed. Results saved in output_save/test_time_adapt/"
echo "Note: Weights are discarded after each request for zero deployment risk." 