from typing import Dict, List, Optional, Union
from itertools import chain
import json

import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset

from lemma_rules import gen_lemma_rule


class RuleMap:

    def __init__(self, rule2id: Dict[str, int], unk_rule_token: str = "<UNK>"):
        self.rule2id = rule2id
        self.id2rule = None
        self.unk_rule_token = unk_rule_token
        self._init_helper()
        self.num_labels = len(self.rule2id)

    def _init_helper(self):
        assert self.unk_rule_token in self.rule2id
        self.id2rule = {value: key for key, value in self.rule2id.items()}

    @classmethod
    def from_dataset(
            cls,
            dataset: Dataset,
            lemma_rule_column_name: str,
            unk_rule_token: str = "<UNK>",
    ) -> "RuleMap":
        unique_rules = set(chain(*dataset[lemma_rule_column_name]))
        rule2id = {rule: i for i, rule in enumerate(unique_rules)}
        rule2id[unk_rule_token] = len(rule2id)
        return cls(rule2id=rule2id, unk_rule_token=unk_rule_token)

    @classmethod
    def from_json(cls, path: str, unk_rule_token: str = "<UNK>") -> "RuleMap":
        with open(path) as file:
            rule2id = json.loads(file.read())
        return cls(rule2id=rule2id, unk_rule_token=unk_rule_token)

    def serialize(self, path: str) -> None:
        with open(path, "w") as file:
            file.write(json.dumps(self.rule2id, ensure_ascii=False))

    def encode_rules(self, rules: List[str]) -> List[int]:
        return [
            self.rule2id.get(rule, self.rule2id[self.unk_rule_token]) for rule in rules
        ]

    def decode_rules(self, encoded_rules: List[int]) -> List[str]:
        return [
            self.id2rule.get(encoded_rule, self.unk_rule_token) for encoded_rule in encoded_rules
        ]


def generate_rules(dataset: Dataset, allow_lr_copy: bool, lemma_rule_column_name: str = "lemma_rules"):
    tokens: List[List[str]] = dataset["tokens"]
    lemmas: List[List[str]] = dataset["lemmas"]
    lemma_rules: List[List[str]] = []
    for token_list, lemma_list in zip(tokens, lemmas):
        tmp_list: List[str] = []
        for token, lemma in zip(token_list, lemma_list):
            tmp_list.append(gen_lemma_rule(token, lemma, allow_lr_copy))
        lemma_rules.append(tmp_list)
    return {lemma_rule_column_name: lemma_rules}


def label_rules(
        dataset: Dataset,
        rule_map: RuleMap,
        lemma_rule_column_name: str = "lemma_rules",
        rule_label_column_name: str = "lemma_rule_labels",
):
    lemma_rules: List[List[str]] = dataset[lemma_rule_column_name]
    lemma_rule_labels: List[List[int]] = []
    for sent in lemma_rules:
        lemma_rule_labels.append(
            rule_map.encode_rules(sent)
        )
    return {rule_label_column_name: lemma_rule_labels}


class LemmaRulePreprocessor:

    def __init__(
            self,
            allow_lr_copy: bool = True,
            unk_rule_token: str = "<UNK>",
            ignore_index: int = -100,
            lemma_rule_column_name: str = "lemma_rules",
            rule_label_column_name: str = "lemma_rule_labels",
            rule_map: Optional[RuleMap] = None,
            rule_map_path: Optional[str] = None,
            rule_label_split: str = "train",
            # batch_size: Optional[int] = None,
    ):
        self.allow_lr_copy = allow_lr_copy
        self.unk_rule_token = unk_rule_token
        self.ignore_index = ignore_index
        self.lemma_rule_column_name = lemma_rule_column_name
        self.rule_label_column_name = rule_label_column_name
        self.rule_map = rule_map
        self.rule_map_path = rule_map_path
        self.rule_label_split = rule_label_split

    def __call__(
            self,
            dataset: Union[Dataset, DatasetDict],
            override_rule_map: bool = False,
    ) -> Union[Dataset, DatasetDict]:
        dataset = dataset.map(
            generate_rules, batched=True, fn_kwargs={
                "allow_lr_copy": self.allow_lr_copy, "lemma_rule_column_name": self.lemma_rule_column_name
            }
        )

        if not self.rule_map and not self.rule_map_path or override_rule_map:
            if isinstance(dataset, DatasetDict):
                rule_dataset = dataset[self.rule_label_split]
            else:
                rule_dataset = dataset
            self.rule_map = RuleMap.from_dataset(
                rule_dataset, lemma_rule_column_name=self.lemma_rule_column_name, unk_rule_token=self.unk_rule_token
            )
        elif self.rule_map_path and not self.rule_map:
            # check that json exists
            self.rule_map = RuleMap.from_json(self.rule_map_path, unk_rule_token=self.unk_rule_token)

        dataset = dataset.map(
            label_rules, batched=True, fn_kwargs={
                "rule_map": self.rule_map,
                "lemma_rule_column_name": self.lemma_rule_column_name,
                "rule_label_column_name": self.rule_label_column_name,
            }
        )
        return dataset

    def save_rule_map(self, path: Optional[str] = None):
        if not self.rule_map:
            raise ValueError("Rule map was not created")
        if not path and not self.rule_map_path:
            raise ValueError("No save path specified")
        elif not path and self.rule_map_path:
            self.rule_map.serialize(path)
        else:
            self.rule_map.serialize(self.rule_map_path)


def tokenize_and_align_labels(
        examples: Dataset,
        tokenizer,
        max_length: int,
        label_all_tokens: bool = True,
        rule_label_column_name: str = "lemma_rule_labels",
        ignore_index: int = -100,

):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples[rule_label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(ignore_index)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else ignore_index)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


class LemmaRuleDataset(TorchDataset):

    return_columns = (
        "input_ids",
        "labels",
    )

    def __init__(
            self,
            dataset: Dataset,
            tokenizer,
            device,
            max_length: int,
            label_all_tokens: bool = True,
            rule_label_column_name: str = "lemma_rule_labels",
            ignore_index: int = -100,

    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        self.label_all_tokens = label_all_tokens
        self.rule_label_column_name = rule_label_column_name
        self.ignore_index = ignore_index

        self._preprocess()

    def _preprocess(self) -> None:
        self.dataset = self.dataset.map(
            tokenize_and_align_labels,
            batched=True,
            fn_kwargs={
                "label_all_tokens": self.label_all_tokens,
                "rule_label_column_name": self.rule_label_column_name,
                "ignore_index": self.ignore_index,
                "max_length": self.max_length,
                "tokenizer": self.tokenizer,
            }

        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        item = {
            key: torch.tensor(value).to(self.device) for key, value in item.items() if key in self.return_columns
        }
        return item

# add collator


