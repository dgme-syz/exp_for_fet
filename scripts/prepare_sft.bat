@echo off

set DATA_PATH = "DGME/figer"
set CLS_PATH = "modules/datasets_classes/figer.txt"
set SPLIT = "test"
set ORDER = 0


python prepare_sft.py ^
    --data_path %DATA_PATH% ^
    --cls_path %CLS_PATH% ^
    --split %SPLIT% ^
    --order %ORDER%