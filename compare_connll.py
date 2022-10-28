from conllu import parse

gold_file_path = "et_edt-ud-dev.conllu"
pred_file_path = "predicted.conllu"

REMOVE_SYMBOLS = True
ALL_LOWER_CASE = False

with open(gold_file_path) as file:
    gold_conll = parse(file.read())

with open(pred_file_path) as file:
    pred_conll = parse(file.read())

total = 0
correct = 0

for pred_token_list, gold_token_list in zip(pred_conll, gold_conll):
    for pred_token, gold_token in zip(pred_token_list, gold_token_list):
        true_lemma = gold_token["lemma"]
        true_upos = gold_token["upos"]
        if REMOVE_SYMBOLS and true_upos != "PUNCT" and len(true_lemma) > 1:
            true_lemma = true_lemma.replace("_", "").replace("=", "")
        if ALL_LOWER_CASE:
            if pred_token["lemma"].lower() == true_lemma.lower():
                correct += 1
            else:
                print(
                    f'original form: {pred_token["form"]}, predicted lemma: {pred_token["lemma"].lower()}, '
                    f'true lemma: {true_lemma.lower()} '
                )
        else:
            if pred_token["lemma"] == true_lemma:
                correct += 1
            else:
                print(
                    f'original form: {pred_token["form"]}, predicted lemma: {pred_token["lemma"]}, '
                    f'true lemma: {true_lemma}'
                )
        total += 1

print(correct / total)
