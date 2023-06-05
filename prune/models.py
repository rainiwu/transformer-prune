from typing import Dict, Tuple, Any

from transformers import (
    GPT2Tokenizer,
    GPT2ForQuestionAnswering,
    GPT2ForSequenceClassification,
    BertTokenizer,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    AlbertTokenizer,
    AlbertForQuestionAnswering,
    AlbertForSequenceClassification,
    RobertaTokenizer,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    ElectraTokenizer,
    ElectraForQuestionAnswering,
    ElectraForSequenceClassification,
)

from prune.types import MetricType, ModelType


pretrained: Dict[MetricType, Dict[ModelType, Tuple[Any, Any]]] = {
    "squad": {
        "gpt2": (
            GPT2Tokenizer.from_pretrained("gpt2"),
            GPT2ForQuestionAnswering.from_pretrained("gpt2"),
        ),
        "bert": (
            BertTokenizer.from_pretrained("bert-base-uncased"),
            BertForQuestionAnswering.from_pretrained("bert-base-uncased"),
        ),
        "albert": (
            AlbertTokenizer.from_pretrained("albert-base-v2"),
            AlbertForQuestionAnswering.from_pretrained("albert-base-v2"),
        ),
        "roberta": (
            RobertaTokenizer.from_pretrained("roberta-base"),
            RobertaForQuestionAnswering.from_pretrained("roberta-base"),
        ),
        "electra": (
            ElectraTokenizer.from_pretrained("google/electra-base-discriminator"),
            ElectraForQuestionAnswering.from_pretrained(
                "google/electra-base-discriminator"
            ),
        ),
    },
    "glue": {
        "gpt2": (
            GPT2Tokenizer.from_pretrained("gpt2"),
            GPT2ForSequenceClassification.from_pretrained("gpt2"),
        ),
        "bert": (
            BertTokenizer.from_pretrained("bert-base-uncased"),
            BertForSequenceClassification.from_pretrained("bert-base-uncased"),
        ),
        "albert": (
            AlbertTokenizer.from_pretrained("albert-base-v2"),
            AlbertForSequenceClassification.from_pretrained("albert-base-v2"),
        ),
        "roberta": (
            RobertaTokenizer.from_pretrained("roberta-base"),
            RobertaForSequenceClassification.from_pretrained("roberta-base"),
        ),
        "electra": (
            ElectraTokenizer.from_pretrained("google/electra-base-discriminator"),
            ElectraForSequenceClassification.from_pretrained(
                "google/electra-base-discriminator"
            ),
        ),
    },
}
