from typing import Literal, NamedTuple, Callable, Optional

ModelType = Literal["gpt2", "bert", "albert", "roberta", "electra"]
MetricType = Literal["squad", "glue"]


class TrainerHook(NamedTuple):
    pretrain: Optional[Callable] = None
    prestep: Optional[Callable] = None
    preoptimize: Optional[Callable] = None
    postoptimize: Optional[Callable] = None
    posttrain: Optional[Callable] = None
