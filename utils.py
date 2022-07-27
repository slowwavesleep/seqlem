import torch


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
