#!/bin/bash

# Step 2: Supervised Bootstrap Training (Stage 1)
# Short training run on mixed demonstration→query corpus with KL penalty

echo "Starting ICRL Bootstrap Training..."

python3 ../../train.py \
  --load_model "../../myfolder/models/RWKV-x070-7B-20240816-ctx4096.pth" \
  --proj_dir "output_save/bootstrap" \
  --train_data_file "output_h5/icrl_episodes.h5" \
  --ctx_len 4096 \
  --chunk_ctx 512 \
  --micro_bsz 1 \
  --epoch_steps 100 \
  --epoch_count 5 \
  --epoch_save 1 \
  --lr_init 5e-5 \
  --lr_final 1e-5 \
  --warmup_steps 10 \
  --beta1 0.9 \
  --beta2 0.99 \
  --adam_eps 1e-8 \
  --accelerator gpu \
  --devices 1 \
  --precision fp16 \
  --strategy auto \
  --grad_cp 1 \
  --my_testing x070 \
  --sft 1 \
  --sft_kl_mode 1 \
  --sft_kl_alpha 0.1 \
  --sft_kl_temperature 2.0 \
  --icrl_bootstrap 1 \
  --icrl_k_exemplars 3 \
  --icrl_n_queries 1

echo "Bootstrap training completed. Model saved in output_save/bootstrap/" 