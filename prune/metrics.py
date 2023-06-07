"""
submodule defining operations on metric datasets

significant portions of this submodule are adapated from
https://huggingface.co/learn/nlp-course/
with modifications for integration purposes

the huggingface nlp course is licensed under the Apache v2.0 license
you may obtain a copy of the license here http://www.apache.org/licenses/LICENSE-2.0
"""

from typing import Callable, Tuple, Any
import collections
import functools
import copy

import numpy as np
import datasets
import evaluate
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_scheduler

SQUAD_DATA = datasets.load_dataset("squad")
SQUAD_METRIC = evaluate.load("squad")


def preprocess_training_examples(examples, tokenizer: Callable):
    """
    training set preprocessing adapted from huggingface course:
    https://huggingface.co/learn/nlp-course/chapter7/7
    """
    max_length = 384
    stride = 128

    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(examples, tokenizer: Callable):
    """
    validation set preprocessing adapted from huggingface course:
    https://huggingface.co/learn/nlp-course/chapter7/7
    """
    max_length = 384
    stride = 128

    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def compute_metrics(start_logits, end_logits, features, examples):
    """
    metric computation adapted from huggingface course
    """
    n_best = 20
    max_answer_length = 30

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]
    return SQUAD_METRIC.compute(
        predictions=predicted_answers, references=theoretical_answers
    )


def generate_squad_dataloaders(
    tokenizer: Callable, dataset_percent: int = 50
) -> Tuple[DataLoader, DataLoader, Any]:
    """
    create the appropriate dataloaders for training and evaluation dataset
    """
    training_preprocessing: Callable = functools.partial(
        preprocess_training_examples, tokenizer=tokenizer
    )
    validation_preprocess: Callable = functools.partial(
        preprocess_validation_examples, tokenizer=tokenizer
    )

    reduced_training = datasets.load_dataset(
        "squad", split=f"train[:{dataset_percent}%]"
    )
    reduced_validate = datasets.load_dataset(
        "squad", split=f"validation[:{dataset_percent}%]"
    )

    train_dataset = reduced_training.map(  # type: ignore
        training_preprocessing,
        batched=True,
        remove_columns=SQUAD_DATA["train"].column_names,  # type: ignore
    )

    validation_dataset = reduced_validate.map(  # type: ignore
        validation_preprocess,
        batched=True,
        remove_columns=SQUAD_DATA["validation"].column_names,  # type: ignore
    )

    train_dataset.set_format("torch")  # type: ignore
    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")  # type: ignore

    train_dataloader = DataLoader(
        train_dataset,  # type: ignore
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=8,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        validation_set, collate_fn=default_data_collator, batch_size=8, pin_memory=True  # type: ignore
    )

    return train_dataloader, eval_dataloader, validation_dataset


def finetune_squad(
    tokenizer: Callable, model: torch.nn.Module, dataset_percent: int = 100
) -> torch.nn.Module:
    """
    finetune a given model on the squad dataset
    """
    model = copy.deepcopy(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    train_dataloader, eval_dataloader, validation_dataset = generate_squad_dataloaders(
        tokenizer, dataset_percent
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    num_train_epochs = 3
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for _, batch in enumerate(train_dataloader):
            gpu_batch = {}
            for key, value in batch.items():
                gpu_batch[key] = value.to(device)

            optimizer.zero_grad()

            outputs = model(**gpu_batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        start_logits = []
        end_logits = []
        for batch in tqdm(eval_dataloader):
            gpu_batch = {}
            for key, value in batch.items():
                gpu_batch[key] = value.to(device)

            with torch.no_grad():
                outputs = model(**gpu_batch)

            start_logits.append(outputs.start_logits.to("cpu"))
            end_logits.append(outputs.end_logits.to("cpu"))

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(validation_dataset)]  # type: ignore
        end_logits = end_logits[: len(validation_dataset)]  # type: ignore

        metrics = compute_metrics(
            start_logits,
            end_logits,
            validation_dataset,
            SQUAD_DATA["validation"],  # type: ignore
        )
        print(f"epoch {epoch}:", metrics)

    return model.to("cpu")
