from itertools import chain

from datasets import load_dataset

from lemma_rules import gen_lemma_rule, apply_lemma_rule

DATASET_NAME = "et_edt"
ALLOW_COPY = True

data = load_dataset("universal_dependencies", DATASET_NAME)

data = data["train"]
tokens = list(chain(*data["tokens"]))
lemmas = list(chain(*data["lemmas"]))

for token, lemma in zip(tokens, lemmas):
    lemma_rule = gen_lemma_rule(token, lemma, ALLOW_COPY)
    assert lemma == apply_lemma_rule(token, lemma_rule)
