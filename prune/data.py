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
    "gpt2": (
        GPT2Tokenizer.from_pretrained("gpt2"),
        GPT2Model.from_pretrained("gpt2"),
    ),
    "bert": (
        BertTokenizer.from_pretrained("bert-base-uncased"),
        BertModel.from_pretrained("bert-base-uncased"),
    ),
    "albert": (
        AlbertTokenizer.from_pretrained("albert-base-v2"),
        AlbertModel.from_pretrained("albert-base-v2"),
    ),
    "roberta": (
        RobertaTokenizer.from_pretrained("roberta-base"),
        RobertaModel.from_pretrained("roberta-base"),
    ),
    "electra": (
        ElectraTokenizer.from_pretrained("google/electra-base-discriminator"),
        ElectraModel.from_pretrained("google/electra-base-discriminator"),
    ),
}
