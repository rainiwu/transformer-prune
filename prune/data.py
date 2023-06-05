from typing import Dict, Literal, Tuple, Any

from transformers import (
    GPT2Tokenizer,
    GPT2Model,
    BertTokenizer,
    BertModel,
    AlbertTokenizer,
    AlbertModel,
    RobertaTokenizer,
    RobertaModel,
    ElectraTokenizer,
    ElectraModel,
)

ModelType = Literal["gpt2", "bert", "albert", "roberta", "electra"]

pretrained: Dict[ModelType, Tuple[Any, Any]] = {
    "gpt2": (GPT2Tokenizer, GPT2Model),
    "bert": (BertTokenizer, BertModel),
    "albert": (AlbertTokenizer, AlbertModel),
    "roberta": (RobertaTokenizer, RobertaModel),
    "electra": (ElectraTokenizer, ElectraModel),
}
