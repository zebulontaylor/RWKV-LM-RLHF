#!/bin/bash

# Step 4: PPO Fine-tuning with ICRL (Stage 3-4)
# Main ICRL training with PPO and exploration fixes

echo "Starting ICRL PPO Training..."

python3 ../../train.py \
  --load_model "output_save/bootstrap/rwkv-5.pth" \
  --proj_dir "output_save/icrl_ppo" \
  --train_data_file "output_h5/icrl_episodes.h5" \
  --ctx_len 4096 \
  --chunk_ctx 512 \
  --micro_bsz 1 \
  --epoch_steps 1000 \
  --epoch_count 50 \
  --epoch_save 5 \
  --lr_init 1e-5 \
  --lr_final 1e-6 \
  --warmup_steps 50 \
  --beta1 0.9 \
  --beta2 0.99 \
  --adam_eps 1e-8 \
  --accelerator gpu \
  --devices 1 \
  --precision fp16 \
  --strategy auto \
  --grad_cp 1 \
  --my_testing x070 \
  --icrl 1 \
  --icrl_value_head 1 \
  --icrl_k_exemplars 3 \
  --icrl_n_queries 1 \
  --icrl_episode_length 2048 \
  --icrl_ppo_epochs 4 \
  --icrl_kl_beta 0.01 \
  --icrl_entropy_beta 0.01 \
  --icrl_epsilon_mixture 0.1 \
  --icrl_reward_model_path "output_save/reward_model/rwkv-10.pth" \
  --icrl_gen_count 4 \
  --icrl_gen_length 512 \
  --icrl_gen_temperature 1.0 \
  --icrl_gen_topp 0.9

echo "ICRL PPO training completed. Model saved in output_save/icrl_ppo/" 