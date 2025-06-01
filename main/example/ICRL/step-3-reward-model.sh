#!/bin/bash

# Step 3: Reward Model Training (Stage 2)
# Train separate reward model on pairwise comparisons

echo "Starting Reward Model Training..."

python3 ../../train.py \
  --load_model "output_save/bootstrap/rwkv-5.pth" \
  --proj_dir "output_save/reward_model" \
  --train_data_file "output_h5/icrl_episodes.h5" \
  --ctx_len 4096 \
  --chunk_ctx 512 \
  --micro_bsz 1 \
  --epoch_steps 200 \
  --epoch_count 10 \
  --epoch_save 2 \
  --lr_init 3e-5 \
  --lr_final 1e-6 \
  --warmup_steps 20 \
  --beta1 0.9 \
  --beta2 0.99 \
  --adam_eps 1e-8 \
  --accelerator gpu \
  --devices 1 \
  --precision fp16 \
  --strategy auto \
  --grad_cp 1 \
  --my_testing x070 \
  --icrl_reward_model 1 \
  --icrl_value_head 1 \
  --icrl_k_exemplars 3 \
  --icrl_n_queries 1

echo "Reward model training completed. Model saved in output_save/reward_model/" 