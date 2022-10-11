from itertools import zip_longest

import conllu

import conll18_ud_eval as ud_eval

with open("et_edt-ud-dev.conllu") as file:
    data = file.read()
    parsed = conllu.parse(data)

with open("test_preds.txt") as file:
    preds = []
    for line in file:
        preds.append(line.strip("\n").split(" "))

for i, (true_sent, pred_sent) in enumerate(zip(parsed, preds)):

    if len(true_sent) != len(pred_sent):
        print(i)
        for (a, b) in zip_longest(true_sent, pred_sent):
            print(a, b)

    for j, pred in enumerate(pred_sent):
        true_sent[j]["lemma"] = pred

with open("predicted.conllu", "w") as file:
    file.writelines([sentence.serialize() + "\n" for sentence in parsed])


def score(system_conllu_file, gold_conllu_file):
    """ Wrapper for lemma scorer. """
    gold_ud = ud_eval.load_conllu_file(gold_conllu_file)
    system_ud = ud_eval.load_conllu_file(system_conllu_file)
    evaluation = ud_eval.evaluate(gold_ud, system_ud)
    el = evaluation["Lemmas"]
    p, r, f = el.precision, el.recall, el.f1
    return p, r, f


_, _, f = score("predicted.conllu", "et_edt-ud-dev.conllu")

print(f)
