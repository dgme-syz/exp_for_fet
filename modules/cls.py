import random
from typing import List, Union
import ast
from collections import OrderedDict
from functools import lru_cache
from copy import deepcopy
import re

NAME_CLS_MAPPING = OrderedDict({
    ("figer", "modules/datasets_classes/figer.txt")
})


def get_suf(x: str):
    # /person/arist -> arist
    if x == "/":
        return x
    return x[x.rfind('/') + 1:]
    

class AutoCLS:
    sort_module = None
    tree_enabled = False
    def __init__(self, init_cls: List[str], **kwargs):
        self.int2str = init_cls
        self.str2int = {v: k for k, v in enumerate(init_cls)}
        self.num_classes = len(init_cls)
        
        if "/" in self.int2str[0]:
            self._from_tree_get_cls_ord()
            self.tree_enabled = True
            self.int2str = [
                get_suf(x) for x in self.int2str
            ]
            self.str2int = {
                get_suf(x): i for i, x in enumerate(self.int2str)
            }
            
    @classmethod
    def from_file(cls, file_path: str, **kwargs):
        return cls(ast.literal_eval(open(file_path, "r").read()), **kwargs)    
    
    def __getitem__(self, x:Union[List[int], int]) -> Union[List[str], List[List[str]]]:
        r"""
            select the order of classes by x, return the corresponding class names.
        """

        if self.sort_module is None:
            self.sort_module = {
                0: self.get_tree_order_pos, 1: self.get_tree_order_neg, 
                2: self.get_dict_ord_pos, 3: self.get_dict_ord_neg, 4: self.get_shuffle_ord
            }
        
        if isinstance(x, int):
            return self.sort_module[x]()
        return [self.sort_module[i]() for i in x]
        
    def get_tree_order_pos(self) -> List[str]:
        if not self.tree_enabled:
            raise ValueError("Tree structure is not enabled, you must ensure class name contains '/', '/people/actor' for example.")
        return self._remove_wrapper(deepcopy(self.tree_ord)) # 0
    
    def get_tree_order_neg(self) -> List[str]:
        if not self.tree_enabled:
            raise ValueError("Tree structure is not enabled, you must ensure class name contains '/', '/people/actor' for example.")
        return self._remove_wrapper(deepcopy(self.tree_ord[::-1])) # 1
    
    def get_dict_ord_pos(self) -> List[str]:
        return self._remove_wrapper(deepcopy(self.int2str)) # 2
    
    def get_dict_ord_neg(self) -> List[str]:
        return self._remove_wrapper(deepcopy(self.int2str[::-1])) # 3
    
    def get_shuffle_ord(self) -> List[str]:
        return self._remove_wrapper(random.sample(self.int2str, self.num_classes)) # 4
    
    def _remove_wrapper(self, order: List[str]) -> List[str]:
        if self.tree_enabled:
            return [get_suf(x) for x in order]
        else:
            return order
    
    def _dynamic_add(self, cls: str):
        if cls not in self.str2int:
            self.str2int[cls] = self.num_classes
            self.int2str.append(cls)
            self.num_classes += 1
     
    def _from_tree_get_cls_ord(self):
        r"""
            if we want to keep parent-child relationship, we can use this to keep their order.
        """
        self._dynamic_add("/")
        self.parent = [0] * self.num_classes
        adj = [[] for _ in range(self.num_classes)] # 邻接表
        # construct the tree
        for i, x in enumerate(self.int2str):
            if x == "/": continue
            last = x.rfind('/')
            par = self.str2int[x[:last] if last != 0 else "/"]
            adj[par].append(i)
            self.parent[i] = par
            
        dfn, idx = [0] * self.num_classes, 0
        @lru_cache
        def dfs(u):
            nonlocal idx
            dfn[u] = idx
            idx += 1
            for v in adj[u]:
                dfs(v)
        dfs(self._end_token) # 递归
        ord: List[str] = [' '] * self.num_classes
        for i in range(self.num_classes): 
            ord[dfn[i]] = self.int2str[i]
            
        self.tree_ord = ord
    
    def __len__(self):
        return self.num_classes
    
    def find(self, x: str) -> int:
        x = get_suf(x)
        return self.str2int.get(x, -1)
    
    @property
    def _end_token(self):
        return self.str2int["/"]
    
    def extract_answer(self, text: str) -> set[str]:
        # 按照空格换行标点符号分割
        text = re.split(r"\s+|,|\.|!|\?|;", text)
        answers = set()
        
        if self.tree_enabled:
            for x in text:
                x_id = self.find(x)
                if x_id == -1: continue
                
                while x_id != self._end_token:
                    answers.add(self.int2str[x_id])
                    x_id = self.parent[x_id]
        else:
            for x in text:
                x_id = self.find(x)
                if x_id != -1:
                    answers.add(self.int2str[x_id])
        return answers