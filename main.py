# https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner_no_trainer.py
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig

from dataset import LemmaRulePreprocessor, LemmaRuleDataset
from utils import batch_accuracy

parser = ArgumentParser()
parser.add_argument("--pretrained_model_name", type=str, default="tartuNLP/EstBERT")
parser.add_argument("--dataset_name", type=str, default="et_edt")
parser.add_argument("--max_length", type=int, default=224)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--label_all_tokens", default=False, action="store_true")
parser.add_argument("--ignore_index", type=int, default=-100)
parser.add_argument("--allow_lr_copy", default=False, action="store_true")
parser.add_argument("--model_save_path", type=str, default="./model")
# parser.add_argument("--lemma_rule_column_name", type=str, default="lemma_rules")
args = parser.parse_args()


epochs = args.epochs
name = args.pretrained_model_name
max_length = args.max_length
device = args.device
batch_size = args.batch_size
learning_rate = args.learning_rate
dataset_name = args.dataset_name
label_all_tokens = args.label_all_tokens
ignore_index = args.ignore_index
allow_lr_copy = args.allow_lr_copy
model_save_path = args.model_save_path

if __name__ == "__main__":
    lp = LemmaRulePreprocessor(
        allow_lr_copy=allow_lr_copy,
        ignore_index=ignore_index,
    )

    dataset = load_dataset("universal_dependencies", dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(name)

    train_ds = LemmaRuleDataset(
        dataset=lp(dataset["train"]),
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        label_all_tokens=label_all_tokens,
        ignore_index=ignore_index,
    )

    validation_ds = LemmaRuleDataset(
        dataset=lp(dataset["validation"]),
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        label_all_tokens=label_all_tokens,
        ignore_index=ignore_index,
    )

    # initialize hf config with previously generated mappings
    # this is needed to be able to save label names along with the model itself in a non-cumbersome way
    config = AutoConfig.from_pretrained(
        name, num_labels=lp.rule_map.num_labels, label2id=lp.rule_map.rule2id, id2label=lp.rule_map.id2rule
    )
    # reinitializes classification head weights from scratch
    model = AutoModelForTokenClassification.from_pretrained(name, config=config)
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    validation_loader = DataLoader(validation_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    for epoch in range(epochs):
        losses = []
        model.train()
        for batch in tqdm(train_loader):
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        print(f"Train loss on epoch {epoch + 1}: {np.mean(losses)}")
        batch_metrics = []
        model.eval()
        for batch in tqdm(validation_loader):
            with torch.no_grad():
                logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).detach()
            labels = batch["labels"].detach()
            batch_metrics.append(batch_accuracy(preds, labels, ignore_index=ignore_index))
        print(f"Accuracy on epoch {epoch + 1}: {np.mean(batch_metrics)}")

    model.save_pretrained(model_save_path)
