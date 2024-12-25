@echo off

set DATA_PATH = "DGME/figer"
set CLS_PATH = "modules/datasets_classes/figer.txt"
set SPLIT = "train"
set ORDER = 0
set MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
@REM -1 for all samples
set SAMPLE = 1000 

python main.py ^
    --model_path %MODEL_PATH% ^
    --data_path %DATA_PATH% ^
    --cls_path %CLS_PATH% ^
    --split %SPLIT% ^
    --order %ORDER% ^
    --sample %SAMPLE%
