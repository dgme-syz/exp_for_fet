from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from torch.utils.data import DataLoader
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor

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
    
    return parser.parse_args()
    
def process(
    examples: dict, 
    tokenizer: AutoTokenizer,
) -> dict:
    raise NotImplementedError
    
def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    
    
    data_loader = DataLoader(
        dataset = load_dataset(args.data_path, trust_remote_code=True), split="train"
    )
    
    
    
    
    
    
if __name__ == "__main__":
    main(get_args())