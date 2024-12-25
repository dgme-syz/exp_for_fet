@echo off

set MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
set DATASET_ID = "figer_test"
set OUTPUT_DIR = "./saves/sft"
set TEMPLATE = "qwen"
set FINETUNING = "lora"

llamafactory-cli train ^
    --stage sft ^
    --do_train ^
    --model_name_or_path %MODEL_ID% ^
    --dataset %DATASET_ID% ^
    --dataset_dir "LLaMA-Factory/data" ^
    --template %TEMPLATE% ^
    --finetuning_type %FINETUNING% ^
    --output_dir %OUTPUT_DIR% ^
    --overwrite_cache ^
    --overwrite_output_dir ^
    --cutoff_len 1024 ^
    --per_device_train_batch_size 2 ^
    --gradient_accumulation_steps 8 ^
    --lr_scheduler_type cosine ^
    --logging_steps 50 ^
    --warmup_steps 20 ^
    --save_steps 100 ^
    --eval_steps 50 ^
    --evaluation_strategy steps ^
    --load_best_model_at_end ^
    --learning_rate 5e-5 ^
    --num_train_epochs 5.0 ^
    --max_samples 1000 ^
    --val_size 0.1 ^
    --plot_loss ^
    --save_total_limit 10 ^ 
    --fp16