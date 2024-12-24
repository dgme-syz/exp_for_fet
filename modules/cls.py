from itertools import lru_cache
import random
from typing import List, Union
import ast

def get_suf(x: str):
    # /person/arist -> arist
    return x[x.rfind('/') + 1:]
    

class AutoCLS:
    sort_module: None
    tree_enabled: bool = False
    def __init__(self, init_cls: List[str], **kwargs):
        self.int2str = init_cls
        self.str2int = {v: k for k, v in enumerate(init_cls)}
        self.num_classes = len(init_cls)
        
        if "/" in self.int2str[0]:
            self._from_tree_get_cls_ord()
            self.tree_enabled = True
            
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
        return self.tree_ord # 0
    
    def get_tree_order_neg(self) -> List[str]:
        if not self.tree_enabled:
            raise ValueError("Tree structure is not enabled, you must ensure class name contains '/', '/people/actor' for example.")
        return self.tree_ord[::-1] # 1
    
    def get_dict_ord_pos(self) -> List[str]:
        return self.int2str # 2
    
    def get_dict_ord_neg(self) -> List[str]:
        return self.int2str[::-1] # 3
    
    def get_shuffle_ord(self) -> List[str]:
        return random.sample(self.int2str, self.num_classes) # 4
        
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
            last = x.rfind('/')
            par = self.int2str(x[:last] if last != self.str2int["/"] else "/")
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
        dfs(0) # 递归
        ord: List[str] = [' '] * self.num_classes
        for i in self.num_classes: ord[dfn[i]] = self.int2str[i]
            
        self.tree_ord = ord
    
    def __len__(self):
        return self.num_classes
    
    @property
    def _end_token(self):
        return self.str2int["/"]
    
    def extract_answer(self, text: str) -> List[int]:
        text = text.strip().split()
        answers = set()
        
        if self.tree_enabled:
            for x in text:
                if x in self.int2str:
                    x_id = self.str2int[x]
                    while x_id != self._end_token:
                        answers.add(x_id)
                        x_id = self.parent[x_id]
        else:
            for x in text:
                if x in self.int2str:
                    answers.add(self.str2int[x])
        return list(answers)