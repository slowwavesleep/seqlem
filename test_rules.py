from itertools import chain

from datasets import load_dataset
from tqdm.auto import tqdm

from lemma_rules import gen_lemma_rule, apply_lemma_rule

DATASET_NAME = "et_edt"
ALLOW_COPY = True

data = load_dataset("universal_dependencies", DATASET_NAME)

data = data["train"]
tokens = list(chain(*data["tokens"]))
lemmas = list(chain(*data["lemmas"]))
pairs = list(
    set(
        zip(tokens, lemmas)
    )
)

cache = {}

for token, lemma in tqdm(pairs, desc="Testing lemma rules..."):
    if (token, lemma) not in cache:
        lemma_rule = gen_lemma_rule(token, lemma, ALLOW_COPY)
        cache[(token, lemma)] = lemma_rule
    else:
        lemma_rule = cache[(token, lemma)]
    assert lemma == apply_lemma_rule(token, lemma_rule)
