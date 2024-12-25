from transformers import pipeline, AutoTokenizer
import argparse
from datasets import load_dataset
from modules import AutoCLS, eval, get_cls_path
import torch
import wandb

def get_args():
    parser = argparse.ArgumentParser(
        description='Test various ways to solve FET problem'
    )
    
    parser.add_argument(
        "--model_path", 
        type=str,
        default=None,
        help="Path to the model or huggingface model repository"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the data"        
    )
    
    parser.add_argument(
        "--cls_path",
        type=str,
        default=None,
        help="Path to the class file"
    )
    
    parser.add_argument(
        "--order",
        type=int, 
        default=None,
        help="Order of classes"
    )
    
    parser.add_argument(
        "--split", 
        type=str,
        default=None,
        help="split of the data"
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        default=-1,
        help="Number of samples to test"
    )
    
    return parser.parse_args()
    

            
            
def main(args):
    torch.manual_seed(0)    
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    
    if args.cls_path is None:
        args.cls_path = get_cls_path(args.data_path)
    
    cls = AutoCLS.from_file(args.cls_path)
    order = cls[args.order]
    data = load_dataset(args.data_path, trust_remote_code=True, split=args.split)
    
    if args.sample != -1:
        data = data.select(range(args.sample))
    
    pipe = pipeline("text-generation", args.model_path, device_map="auto", torch_dtype="auto")
    run = wandb.init(
        project="FET", 
        name=args.model_path.split("/")[-1] + "_" + args.data_path.split("/")[-1] + "_test"
    )
    
    eval(
        pipe=pipe,
        tokenizer=tokenizer,
        cls=cls,
        data=data,
        order=order
    )
    run.finish()
    
    
if __name__ == "__main__":
    main(get_args())
