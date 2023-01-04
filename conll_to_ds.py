import requests
import datasets

import conllu

et_edt = {
    "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Estonian-EDT/r2.10/et_edt-ud-train.conllu",
    "validation": "https://raw.githubusercontent.com/UniversalDependencies/UD_Estonian-EDT/r2.10/et_edt-ud-dev.conllu",
    "test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Estonian-EDT/r2.10/et_edt-ud-test.conllu",
}

def _generate_examples(filepath):
    data_file = requests.get(filepath).text
    tokenlist = list(conllu.parse(data_file))
    for sent in tokenlist:
        if "sent_id" in sent.metadata:
            idx = sent.metadata["sent_id"]
        else:
            idx = id

        tokens = [token["form"] for token in sent]

        if "text" in sent.metadata:
            txt = sent.metadata["text"]
        else:
            txt = " ".join(tokens)

        yield {
            "idx": str(idx),
            "text": txt,
            "tokens": [token["form"] for token in sent],
            "lemmas": [token["lemma"] for token in sent],
            "upos": [token["upos"] for token in sent],
            "xpos": [token["xpos"] for token in sent],
            "feats": [str(token["feats"]) for token in sent],
            "head": [str(token["head"]) for token in sent],
            "deprel": [str(token["deprel"]) for token in sent],
            "deps": [str(token["deps"]) for token in sent],
            "misc": [str(token["misc"]) for token in sent],
        }

features = datasets.Features(
                {
                    "idx": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "lemmas": datasets.Sequence(datasets.Value("string")),
                    "upos": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "NOUN",
                                "PUNCT",
                                "ADP",
                                "NUM",
                                "SYM",
                                "SCONJ",
                                "ADJ",
                                "PART",
                                "DET",
                                "CCONJ",
                                "PROPN",
                                "PRON",
                                "X",
                                "_",
                                "ADV",
                                "INTJ",
                                "VERB",
                                "AUX",
                            ]
                        )
                    ),
                    "xpos": datasets.Sequence(datasets.Value("string")),
                    "feats": datasets.Sequence(datasets.Value("string")),
                    "head": datasets.Sequence(datasets.Value("string")),
                    "deprel": datasets.Sequence(datasets.Value("string")),
                    "deps": datasets.Sequence(datasets.Value("string")),
                    "misc": datasets.Sequence(datasets.Value("string")),
                }
)

tmp = {}
for key, value in et_edt.items():
    data = [el for el in _generate_examples(value)]
    tmp[key] = datasets.Dataset.from_list(data)

dataset = datasets.DatasetDict(tmp)

dataset.save_to_disk("./data")