from modules.utils import process
from datasets import load_dataset
import argparse
from modules import AutoCLS
import os


def get_args():
    parser = argparse.ArgumentParser(
        description='convert the data to the format that fit for sft'
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
        "--split", 
        type=str, 
        default=None,
        help="split of the data"
    )
    
    parser.add_argument(
        "--order",
        type=int,
        default=None,
        help="Order of classes"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the data"
    )
    
    
    return parser.parse_args()

def main(args):
    dataset = load_dataset(args.data_path, trust_remote_code=True, split=args.split)
    order = AutoCLS.from_file(args.cls_path)[args.order]
    
    dataset = dataset.map(
        function=lambda x: process(x, order, use_type="alpaca"), 
        batched=True, 
        remove_columns=dataset.column_names, 
    )
    
    # 保留外面的中括号
    dataset.to_json(
        os.path.join(args.output_path, args.data_path.split("/")[-1] + "_" + args.split + ".json")
    )
    
if __name__ == "__main__":
    main(get_args())