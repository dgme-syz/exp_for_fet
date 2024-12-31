from collections import defaultdict
from typing import List
from transformers import pipeline, AutoTokenizer
from .cls import AutoCLS, get_suf
from concurrent.futures import ThreadPoolExecutor
from .template import register_template
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Union
import wandb

SYSTEM_INFO = {"role": "system", "content": "You are a excellent linguist, you can finish the following task well! Also, you need to recognize some entity types are relative."}


class Scores:
    def __init__(self):
        self.u = 0
        self.prediction, self.ground_truth = defaultdict(set), defaultdict(set)
        
    def update(self, preds: List[str], truths: List[str]):
        self.prediction[self.u] = set(preds)
        self.ground_truth[self.u] = set(truths)
        self.u += 1
    
    @property
    def evaluate(self):
        """
        Evaluation matrix.
        :param prediction: a dictionary of labels. e.g {0:[1,0],1:[2],2:[3,4],3:[5,6,7]}
        :param ground_truth: a dictionary of labels
        :return:
        """
        # print("prediction:%d, ground:%d"%(len(self.prediction),len(self.ground_truth)))
        assert len(self.prediction) == len(self.ground_truth)
        count = len(self.prediction)
        # print 'Test', count, 'mentions'
        info = {
            'same': 0, 'macro_precision': 0.0, 'macro_recall': 0.0,
            'micro_n': 0.0, 'micro_precision': 0.0, 'micro_recall': 0.0
        }
        for i in self.ground_truth:
            p, g = self.prediction[i], self.ground_truth[i]
            if p == g:
                info['same'] += 1
            same_count = len(p & g)
            info['macro_precision'] += float(same_count) / float(1e-8 if len(p) == 0 else len(p))
            info['macro_recall'] += float(same_count) / float(len(g))
            info['micro_n'] += same_count
            info['micro_precision'] += len(p)
            info['micro_recall'] += len(g)
        info['accuracy'] = float(info['same']) / float(count)
        info['macro_precision'] /= count
        info['macro_recall'] /= count
        info['macro_f1'] = 2 * info['macro_precision'] * info['macro_recall'] / \
            (info['macro_precision'] + info['macro_recall'] + 1e-8)
        info['micro_precision'] = info['micro_n'] / info['micro_precision']
        info['micro_recall'] = info['micro_n'] / info['micro_recall']
        info['micro_f1'] = 2 * info['micro_precision'] * info['micro_recall'] / \
            (info['micro_precision'] + info['micro_recall'] + 1e-8)
        return info 
    
def process(
    examples: dict, 
    order: List[str],
    use_type: str = "eval", 
) -> dict:
    r"""
        (1). use_type == "eval", prepare the data for evaluation.
        (2.  use_type == "alpaca", prepare the data(alpaca format) for sft.
    """
    left_context, right_context = examples["left_context_token"], examples["right_context_token"]
    mentions = examples["mention_span"]
    text, label, num_samples = [], [], len(mentions)
    
    def access_single(i: int):
        t = register_template(
            sentence=" ".join(left_context[i]) + " " + mentions[i] + " " + " ".join(right_context[i]), 
            mention=mentions[i], cls_ord=order
        )
        l = [get_suf(x) for x in examples["y_str"][i]]
        return t, l
    
    # 多线程
    with ThreadPoolExecutor() as executor:
        res = executor.map(access_single, range(num_samples))
        for t, l in res: text.append(t), label.append(l)
        
    if use_type == "eval":
        return {"text": text, "label": label}
    elif use_type == "alpaca":
        return {
            "instruction": text, 
            "output": [" ".join(x) for x in label], 
            "input": ["" for _ in range(len(text))]
        }
    else:
        raise ValueError(f"Invalid use_type: {use_type}")
    
def eval(
    pipe: pipeline,  
    tokenizer: AutoTokenizer,
    cls: AutoCLS,
    data: Dataset, 
    order: List[str],
    **kwargs
):
    mec = Scores()
    dataset=data.map(
        function=lambda x: process(x, order), batched=True, remove_columns=data.column_names
    )
    
    pbar = tqdm(dataset, desc="Evaluating: ")
    with torch.no_grad():
        for i, x in enumerate(pbar):
            text, label = x["text"], x["label"]
            response = pipe(
                [SYSTEM_INFO, {"role": "user", "content": text}], 
                max_new_tokens=256
            )[0]['generated_text'][-1]['content']
            
            # print(
            #     # f"question: {text}\n"
            #     f"Answer: {response}\n"
            #     f"Ground Truth: {label}"
            # )
            mec.update(
                preds=cls.extract_answer(response), truths=label
            )
            info = mec.evaluate
            
            pbar.set_postfix({
                "acc": info["accuracy"],
            })
            wandb.log({"acc": info["accuracy"]})
    print(
        f"Accuracy: {mec.evaluate['accuracy']:.4f}\n"
        f"Macro F1: {mec.evaluate['macro_f1']:.4f}\n"
        f"Micro F1: {mec.evaluate['micro_f1']:.4f}"
    )