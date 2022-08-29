from typing import List, Dict
from itertools import chain

from datasets import load_dataset, Dataset, load_metric
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments
import torch

from dataset import generate_rules


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
    # predictions, labels = p
    # predictions = np.argmax(predictions, axis=2)

    # # Remove ignored index (special tokens)
    # true_predictions = [
    #     [id2rule[p] for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]
    # true_labels = [
    #     [id2rule[l] for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]

    # print(true_predictions)

    return {
        "accuracy": 0
    }


MODEL_NAME = "tartuNLP/EstBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding="longest",
    max_length=256,
    return_tensors="pt",
)

data = load_dataset("universal_dependencies", "et_edt")
data = data.map(generate_rules, batched=True, fn_kwargs={"allow_lr_copy": True})
data = data.remove_columns(set(data.column_names["train"]) - {"idx", "tokens", "lemma_rules", "lemmas"})


rule2id = {key: i for i, key in enumerate(set(chain(*data["train"]["lemma_rules"])))}
rule2id["unk"] = len(rule2id)
id2rule = {value: key for key, value in rule2id.items()}
data = data.map(add_rule_labels, batched=True, fn_kwargs={"rule_map": rule2id})

tokenized = data.map(tokenize_and_align_labels, batched=True)
tokenized = tokenized.remove_columns(set(tokenized.column_names["train"]) - {"input_ids", "token_type_ids", "attention_mask", "labels"})

batch_size = 96
args = TrainingArguments(
    "lemmatization",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_accumulation_steps=5,
)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized["validation"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()