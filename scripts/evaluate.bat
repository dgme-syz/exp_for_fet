@echo off
@REM "DGME/figer"
set DATA_PATH="DGME/figer"
@REM set DATA_PATH="DGME/ontonotes"
@REM "modules/datasets_classes/figer.txt"
@REM set CLS_PATH="modules/datasets_classes/ontonotes.txt"
set CLS_PATH="modules/datasets_classes/figer.txt"
set SPLIT="test"
set ORDER=0
@REM "Qwen/Qwen2.5-0.5B-Instruct"
set MODEL_PATH="E:\\pretrained_models\\Qwen\\Qwen2___5-0___5B-Instruct" 
@REM -1 for all samples
set SAMPLE=1000 

python main.py ^
    --model_path %MODEL_PATH% ^
    --data_path %DATA_PATH% ^
    --cls_path %CLS_PATH% ^
    --split %SPLIT% ^
    --order %ORDER% ^
    --sample %SAMPLE%
