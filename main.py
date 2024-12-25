from transformers import pipeline, AutoTokenizer
import argparse
from torch.utils.data import DataLoader
from datasets import load_dataset
from modules import AutoCLS, eval
from typing import List
import torch
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(
        description='Test various ways to solve FET problem'
    )
    
    parser.add_argument(
        "--model_path", 
        type=str,
        default="E:/pretrained_models/Qwen/Qwen2___5-0___5B-Instruct",
        help="Path to the model or huggingface model repository"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="DGME/figer",
        help="Path to the data"        
    )
    
    parser.add_argument(
        "--cls_path",
        type=str,
        default="modules/figer.txt",
        help="Path to the class file"
    )
    
    parser.add_argument(
        "--order",
        type=int, 
        default=0,
        help="Order of classes"
    )
    
    return parser.parse_args()
    

            
            
def main(args):
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    cls = AutoCLS.from_file(args.cls_path)
    order = cls[args.order]
    data = load_dataset(args.data_path, trust_remote_code=True, split="train").select(range(1000))
    pipe = pipeline("text-generation", args.model_path, device_map="auto", torch_dtype="auto")
    eval(
        pipe=pipe,
        tokenizer=tokenizer,
        cls=cls,
        data=data,
        order=order
    )
    
    
    
    
if __name__ == "__main__":
    main(get_args())
