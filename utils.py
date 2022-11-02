from typing import Tuple

import torch


def batch_accuracy(
        batch_predictions: torch.tensor, batch_labels: torch.tensor, ignore_index: int = -100,
) -> float:
    if not batch_predictions.shape == batch_labels.shape:
        raise ValueError(
            f"Shape mismatch between predictions and gold labels: {batch_predictions.shape} vs. {batch_labels.shape}"
        )
    true_predictions = [
        [p for (p, l) in zip(pred, gold_label) if l != ignore_index]
        for pred, gold_label in zip(batch_predictions, batch_labels)
    ]
    true_labels = [
        [l for (p, l) in zip(pred, gold_label) if l != ignore_index]
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


def remove_symbols_helper(form: str, lemma: str, symbols: Tuple[str, ...]) -> str:
    processed_lemma = lemma
    for symbol in symbols:
        if symbol in lemma and symbol not in form and len(lemma) > 1:
            processed_lemma = processed_lemma.replace(symbol, "")
    return processed_lemma
