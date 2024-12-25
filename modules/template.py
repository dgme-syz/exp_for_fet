from collections import OrderedDict
from typing import List
from transformers import Qwen2Tokenizer

EXTRA_LIMIT = "[Warning]: Just output nothing except entity types above, separate them by one space, there may be more than one answer"




def register_template(
    sentence: str,
    mention: str, 
    cls_ord: List[str],
    extra_limit: bool = True, 
    **kwargs
) -> str:
    r"""
        cls_ord: List[str]
            List of entity types, sorted before the function is called.
    """
    union_cls = " ".join(cls_ord)
    Q = f"[Task]: Fine-grained entity classification\n[sentence]: {sentence}\n[entity]: {mention}\n[entity types]: {union_cls}\n[Classification Result]: \n"
    if extra_limit:
        Q += EXTRA_LIMIT
    return Q



