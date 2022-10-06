from typing import List, Dict
from itertools import chain

from datasets import load_dataset, Dataset, load_metric
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments, AutoConfig, EarlyStoppingCallback, pipeline
import torch
import numpy as np

from dataset import generate_rules
from lemma_rules import apply_lemma_rule


def add_rule_labels(dataset: Dataset, rule_map: Dict[str, int]):
    lemma_rules: List[List[str]] = dataset["lemma_rules"]
    rule_labels: List[List[int]] = []
    for sent in lemma_rules:
        rule_labels.append(
            list(map(lambda rule: rule_map.get(rule, rule_map["unk"]), sent))
        )
    return {"rule_labels": rule_labels}


def tokenize_and_align_labels(examples: Dataset, label_all_tokens: bool = True):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["rule_labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2rule[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2rule[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    total = 0
    correct = 0
    for preds, gold_labels in zip(true_predictions, true_labels):
        for pred, label in zip(preds, gold_labels):
            total += 1
            if pred == label:
                correct += 1
    value = correct / total

    return {
        "accuracy": value
    }


MODEL_NAME = "tartuNLP/EstBERT"
DATASET_NAME = "et_edt"
ALLOW_COPY = True
MAX_LENGTH = 256
BATCH_SIZE = 96
TRAIN_EPOCHS = 1  # 100
EVAL_PER_EPOCH = 1  # 3
EARLY_STOPPING_PATIENCE = 6
LABEL_ALL_TOKENS = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding="longest",
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

data = load_dataset("universal_dependencies", DATASET_NAME)
data = data.map(generate_rules, batched=True, fn_kwargs={"allow_lr_copy": ALLOW_COPY})
data = data.remove_columns(set(data.column_names["train"]) - {"idx", "tokens", "lemma_rules", "lemmas"})


rule2id = {key: i for i, key in enumerate(set(chain(*data["train"]["lemma_rules"])))}
rule2id["unk"] = len(rule2id)
id2rule = {value: key for key, value in rule2id.items()}
data = data.map(add_rule_labels, batched=True, fn_kwargs={"rule_map": rule2id})

tokenized = data.map(tokenize_and_align_labels, batched=True, fn_kwargs={"label_all_tokens": LABEL_ALL_TOKENS})
tokenized = tokenized.remove_columns(
    set(tokenized.column_names["train"]) - {"input_ids", "token_type_ids", "attention_mask", "labels"}
)

config = AutoConfig.from_pretrained(MODEL_NAME, label2id=rule2id, id2label=id2rule)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config)

batch_size = BATCH_SIZE
eval_steps = len(tokenized["train"]) // batch_size // EVAL_PER_EPOCH
early_stopping = EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
args = TrainingArguments(
    "seqlem_model",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=TRAIN_EPOCHS,
    weight_decay=0.01,
    eval_accumulation_steps=5,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    group_by_length=True,
    eval_steps=eval_steps,
    save_steps=eval_steps,
    resume_from_checkpoint="./seqlem_model/checkpoint-11730",
)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

trainer.train()

predictions, labels, metrics = trainer.predict(tokenized["test"], metric_key_prefix="predict")
predictions = np.argmax(predictions, axis=2)

true_predictions: List[List[str]] = [
    [id2rule[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

true_tokens: List[List[str]] = tokenized["test"]["tokens"]

lemmatized: List[List[str]] = []

for (token_list, pred_list) in zip(true_tokens, true_predictions):
    assert len(token_list) == len(pred_list)
    cur_lemmas = []
    for (token, predicted_rule) in zip(token_list, pred_list):
        cur_lemmas.append(apply_lemma_rule(token, predicted_rule))
    lemmatized.append(cur_lemmas)

with open("./test_preds.txt", "w") as writer:
    for prediction in true_predictions:
        writer.write(" ".join(prediction) + "\n")

# def evaluate_dataset(df):
#     predictions, labels, _ = trainer.predict(df)
#     predictions = np.argmax(predictions, axis=2)
#
#     # Remove ignored index (special tokens)
#     true_predictions = [
#         [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     true_labels = [
#         [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#
#     results = metric.compute(predictions=true_predictions, references=true_labels)
#
#     return results
