from typing import Dict, Tuple, Any

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
)

from prune.types import MetricType, ModelType


PRETRAINED: Dict[MetricType, Dict[ModelType, Tuple[Any, Any]]] = {
    "squad": {
        "gpt2": (
            AutoTokenizer.from_pretrained("gpt2"),
            AutoModelForQuestionAnswering.from_pretrained("gpt2"),
        ),
        "bert": (
            AutoTokenizer.from_pretrained("bert-base-uncased"),
            AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased"),
        ),
        "albert": (
            AutoTokenizer.from_pretrained("albert-base-v2"),
            AutoModelForQuestionAnswering.from_pretrained("albert-base-v2"),
        ),
        "roberta": (
            AutoTokenizer.from_pretrained("roberta-base"),
            AutoModelForQuestionAnswering.from_pretrained("roberta-base"),
        ),
        "electra": (
            AutoTokenizer.from_pretrained("google/electra-base-discriminator"),
            AutoModelForQuestionAnswering.from_pretrained(
                "google/electra-base-discriminator"
            ),
        ),
    },
    "glue": {
        "gpt2": (
            AutoTokenizer.from_pretrained("gpt2"),
            AutoModelForSequenceClassification.from_pretrained("gpt2"),
        ),
        "bert": (
            AutoTokenizer.from_pretrained("bert-base-uncased"),
            AutoModelForSequenceClassification.from_pretrained("bert-base-uncased"),
        ),
        "albert": (
            AutoTokenizer.from_pretrained("albert-base-v2"),
            AutoModelForSequenceClassification.from_pretrained("albert-base-v2"),
        ),
        "roberta": (
            AutoTokenizer.from_pretrained("roberta-base"),
            AutoModelForSequenceClassification.from_pretrained("roberta-base"),
        ),
        "electra": (
            AutoTokenizer.from_pretrained("google/electra-base-discriminator"),
            AutoModelForSequenceClassification.from_pretrained(
                "google/electra-base-discriminator"
            ),
        ),
    },
}
