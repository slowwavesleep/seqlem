from collections import Counter

from conllu import parse

from utils import remove_symbols_helper

gold_file_path = "et_edt-ud-dev.conllu"
pred_file_path = "predicted.conllu"

REMOVE_SYMBOLS = True
ALL_LOWER_CASE = True

with open(gold_file_path) as file:
    gold_conll = parse(file.read())

with open(pred_file_path) as file:
    pred_conll = parse(file.read())

total = 0
correct = 0

errors = []

for pred_token_list, gold_token_list in zip(pred_conll, gold_conll):
    for pred_token, gold_token in zip(pred_token_list, gold_token_list):
        true_lemma = gold_token["lemma"]
        # true_upos = gold_token["upos"]
        true_form = gold_token["form"]
        # if REMOVE_SYMBOLS and true_upos != "PUNCT" and len(true_lemma) > 1:
        if REMOVE_SYMBOLS:
            true_lemma = remove_symbols_helper(true_form, true_lemma, symbols=("_", "="))
        if ALL_LOWER_CASE:
            if pred_token["lemma"].lower() == true_lemma.lower():
                correct += 1
            else:
                errors.append((pred_token["form"], pred_token["lemma"].lower(), true_lemma.lower()))
                # print(
                #     f'original form: {pred_token["form"]}, predicted lemma: {pred_token["lemma"].lower()}, '
                #     f'true lemma: {true_lemma.lower()} '
                # )
        else:
            if pred_token["lemma"] == true_lemma:
                correct += 1
            else:
                errors.append((pred_token["form"], pred_token["lemma"], true_lemma))
                # print(
                #     f'original form: {pred_token["form"]}, predicted lemma: {pred_token["lemma"]}, '
                #     f'true lemma: {true_lemma}'
                # )
        total += 1

for el in Counter(errors).most_common():
    strings, count = el
    form, predicted, actual = strings
    print(
        f'original form: {form}, predicted lemma: {predicted}, '
        f'true lemma: {actual}, count: {count}'
    )
print(correct / total)
