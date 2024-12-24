from collections import OrderedDict
from typing import List

EXTRA_LIMIT = "[Warning]: Just output nothing except entity types above, separate them by one space, there may be more than one answer"

NAME_CLS_MAPPING = OrderedDict({
    ("figer", "data_module/figer.txt")
})



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
    union_cls = " ".join(ord)
    Q = f"[Task]: Fine-grained entity classification\n[sentence]: {sentence}\n[entity]: {mention}\n[entity types]: {union_cls}\n[Classification Result]: \n"
    if extra_limit:
        Q += EXTRA_LIMIT
    return Q




