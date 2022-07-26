# https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner_no_trainer.py
from typing import Dict
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification

from dataset import LemmaRulePreprocessor, LemmaRuleDataset

parser = ArgumentParser()
parser.add_argument("--pretrained_model_name", type=str, default="tartuNLP/EstBERT")
parser.add_argument("--dataset_name", type=str, default="et_edt")
parser.add_argument("--max_length", type=int, default=224)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()


epochs = args["epochs"]
name = args["pretrained_model_name"]
max_lengths = args["max_length"]
device = args["device"]
batch_size = args["batch_size"]
learning_rate = args["learning_rate"]
dataset_name = args["dataset_name"]


lp = LemmaRulePreprocessor()

dataset = load_dataset("universal_dependencies", dataset_name)
dataset = lp(dataset["test"])

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForTokenClassification.from_pretrained(name, num_labels=lp.rule_map.num_labels)

lr_dataset = LemmaRuleDataset(
    dataset=dataset,
    tokenizer=tokenizer,
    device=device,
    max_length=max_lengths,
)

lr_loader = DataLoader(lr_dataset, batch_size=batch_size)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)


def batch_accuracy(batch_predictions: torch.tensor, batch_labels: torch.tensor) -> float:
    if not batch_predictions.shape == batch_labels.shape:
        raise ValueError(
            f"Shape mismatch between predictions and gold labels: {batch_predictions.shape} vs. {batch_labels.shape}"
        )
    true_predictions = [
        [p for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(batch_predictions, batch_labels)
    ]
    true_labels = [
        [l for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(batch_predictions, batch_labels)
    ]

    total = 0
    correct = 0
    for preds, gold_labels in zip(true_predictions, true_labels):
        for pred, label in zip(preds, gold_labels):
            total += 1
            if pred == label:
                correct += 1
    value = correct / total
    return value


for epoch in range(epochs):
    losses = []
    model.train()
    for batch in tqdm(lr_loader):
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    print(f"Train loss on epoch {epoch}: {np.mean(losses)}")
    batch_metrics = []
    model.eval()
    for batch in tqdm(lr_loader):
        with torch.no_grad():
            logits = model(**batch).logits
        preds = torch.argmax(logits, dim=-1).detach()
        labels = batch["labels"].detach()
        batch_metrics.append(batch_accuracy(preds, labels))
    print(f"Accuracy on epoch {epoch}: {np.mean(batch_metrics)}")

