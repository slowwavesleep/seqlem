from argparse import ArgumentParser

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset

parser = ArgumentParser()
parser.add_argument("--pretrained_model_name_or_path", type=str)
args = parser.parse_args()

dataset = load_dataset("universal_dependencies", "et_edt")

model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

lemma_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="token-classification",
    aggregation_strategy="first",
)

lemma_pipeline(dataset["test"]["texts"][0])


