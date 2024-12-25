from collections import defaultdict
from typing import List
from transformers import pipeline, AutoTokenizer
from .cls import AutoCLS, get_suf
from datasets import Dataset, load_dataset
from concurrent.futures import ThreadPoolExecutor
from .template import register_template
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
) -> dict:
    mentions, tokens = examples["mentions"], examples["tokens"]
    text, label, num_samples = [], [], len(mentions)
    
    def access_single(i: int):
        t, l = [], []
        mention, token = mentions[i], tokens[i] # there may be multiple mentions
        num_entity = len(mention["start"])
        for j in range(num_entity):
            start, end, labels = mention["start"][j], mention["end"][j], mention["labels"][j]
            sentence = token
            entity = sentence[start:end]
            t.append(register_template(
                sentence=" ".join(sentence), mention=" ".join(entity), cls_ord=order
            ))
            l.append([get_suf(x) for x in labels])
        return t, l
    
    # 多线程
    with ThreadPoolExecutor() as executor:
        res = executor.map(access_single, range(num_samples))
        for t, l in res: text.extend(t), label.extend(l)
        
    return {"text": text, "label": label}
    
def eval(
    pipe: pipeline,  
    tokenizer: AutoTokenizer,
    cls: AutoCLS,
    data: Dataset, 
    order: List[str],
    **kwargs
):
    mec = Scores()
    data_loader = DataLoader(
        dataset=data.map(
            function=lambda x: process(x, order), batched=True, remove_columns=data.column_names, num_proc=2
        ), 
        batch_size=1, 
        num_workers=2,
        pin_memory=torch.cuda.is_available(), 
        shuffle=True
    )
    
    pbar = tqdm(data_loader, desc="Evaluating: ")
    with torch.no_grad():
        for i, x in enumerate(pbar):
            text, label = x["text"][0], x["label"][0]
            
            response = pipe(
                [SYSTEM_INFO, {"role": "user", "content": text}], 
                max_new_tokens=256
            )[0]['generated_text'][-1]['content']
            
            mec.update(
                preds=cls.extract_answer(response), truths=label
            )
            info = mec.evaluate
            
            pbar.set_postfix({
                "acc": info["accuracy"],
            })
    print(
        f"Accuracy: {mec.evaluate['accuracy']:.4f}\n"
        f"Macro F1: {mec.evaluate['macro_f1']:.4f}\n"
        f"Micro F1: {mec.evaluate['micro_f1']:.4f}"
    )