from conllu import parse

gold_file_path = "et_edt-ud-dev.conllu"
pred_file_path = "predicted.conllu"

with open(gold_file_path) as file:
    gold_conll = parse(file.read())

with open(pred_file_path) as file:
    pred_conll = parse(file.read())

total = 0
correct = 0

for pred_token_list, gold_token_list in zip(pred_conll, gold_conll):
    for pred_token, gold_token in zip(pred_token_list, gold_token_list):
        if pred_token["lemma"] == gold_token["lemma"]:
            correct += 1
        else:
            print(f'{pred_token["lemma"]} != {gold_token["lemma"]}')
        total += 1

print(correct / total)