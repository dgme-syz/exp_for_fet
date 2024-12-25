from .template import register_template
from .cls import AutoCLS, NAME_CLS_MAPPING
from .utils import Scores, eval, process


def get_cls_path(name: str) -> str:
    return NAME_CLS_MAPPING[name]


__all__ = [
    "register_template", 
    "AutoCLS", 
    "eval",
    "process", 
    "get_cls_path"
]