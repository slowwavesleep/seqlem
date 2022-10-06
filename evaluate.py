from itertools import zip_longest

import conllu

with open("et_edt-ud-test.conllu") as file:
    data = file.read()
    parsed = conllu.parse(data)

with open("test_preds.txt") as file:
    preds = []
    for line in file:
        preds.append(line.strip("\n").split(" "))


for (true_sent, pred_sent) in zip(parsed, preds):

    if len(true_sent) != len(pred_sent):
        for (a, b) in zip_longest(true_sent, pred_sent):
            print(a, b)

    for i, pred in enumerate(pred_sent):
        true_sent[i]["lemma"] = pred

with open("predicted.conllu", "w") as file:
    file.writelines([sentence.serialize() + "\n" for sentence in parsed])