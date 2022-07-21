# https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner_no_trainer.py
from typing import Dict

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification

from dataset import LemmaRulePreprocessor, LemmaRuleDataset

EPOCHS = 1
NAME = "tartuNLP/EstBERT"
MAX_LENGTH = 224
DEVICE = torch.device("cpu")
BATCH_SIZE = 4
LEARNING_RATE = 1e-5


lp = LemmaRulePreprocessor()

dataset = load_dataset("universal_dependencies", "et_edt")
dataset = lp(dataset["test"])

tokenizer = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForTokenClassification.from_pretrained(NAME, num_labels=lp.rule_map.num_labels)

lr_dataset = LemmaRuleDataset(
    dataset=dataset,
    tokenizer=tokenizer,
    device=DEVICE,
    max_length=MAX_LENGTH,
)

lr_loader = DataLoader(lr_dataset, batch_size=BATCH_SIZE)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)


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


for epoch in range(EPOCHS):
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

