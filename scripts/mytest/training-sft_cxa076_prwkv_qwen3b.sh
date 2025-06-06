python train.py --load_model "/home/client/Projects/RWKV-Infer/models/PRWKV7-cxa076-qwen3b-stage2final-ctx2048.pth" \
 --wandb "RWKV-LM-RLHF cxa076 SFT infctx Hiroshima" --proj_dir "myfolder/Outputs/PRWKV7-cxa076-qwen3b-hiroshima" \
 --vocab_size 151936 --ctx_len 4096 \
 --chunk_ctx 2048 \
 --infctx 0 \
 --epoch_steps 100 --epoch_count 200 --epoch_begin 0 --epoch_save 1 \
 --micro_bsz 2 --n_layer 36 --n_embd 2048 --dim_ffn 11008 \
 --gqa_kv_heads 2 \
 --head_size_a 128 \
 --rms_norm_eps 1e-6 \
 --warmup_steps 100 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
 --accelerator gpu --devices 1 --precision 'bf16' \
 --grad_cp 1 --my_testing "cxa076" \
 --strategy deepspeed_stage_2_offload \
 --layer_profile 'layerprofile/36_TEST_lora.csv' \
 --quant 1 \
 --quant_mode 'int8'\
 --gpu_arch 'backstepping_longhead' \
 --limited_lora 0 \
 --sft 1 \
 --sft_jsonmode 1 \
 --sft_jsonmode_tokenizermode 'qwen' \
 --smoothing 0.0001 \
 --random_mode 1 \
 --infctx_dataset_multiplier 4 \
 --optim 'muon' \
 --train_data_file 'myfolder/jsonl_dataset_hiroshima_light' \
 --accumulate_grad_batches 4
