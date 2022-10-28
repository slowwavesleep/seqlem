from conllu import parse

TRUE_LABELS_PATH = "true_labels.txt"
PRED_LABELS_PATH = "predicted_labels.txt"
PRED_LEMMAS_PATH = "test_preds.txt"
TRUE_LEMMAS_PATH = "et_edt-ud-dev.conllu"

true_labels = []
pred_labels = []
pred_lemmas = []

with open(TRUE_LEMMAS_PATH) as file:
    gold_conll = parse(file.read())

true_lemmas = []

for sent in true_lemmas:
    tmp = []
    for token in sent:
        tmp.append(token["lemma"])
    true_lemmas.append(tmp)

with open(TRUE_LABELS_PATH) as file:
    for line in file:
        labels = line.strip("\n").split(" ")
        true_labels.append([int(label) for label in labels])

with open(PRED_LABELS_PATH) as file:
    for line in file:
        labels = line.strip("\n").split(" ")
        pred_labels.append([int(label) for label in labels])

with open(PRED_LEMMAS_PATH) as file:
    for line in file:
        lemmas = line.strip("\n").split(" ")
        pred_lemmas.append(lemmas)

equal_lengths = []

for true_lemma_sent, pred_lemma_sent, label_sent, true_sent in zip(true_lemmas, pred_lemmas, pred_labels, true_labels):
    true_preds = []
    for p, t in zip(label_sent, true_sent):
        if t != -100:
            true_preds.append(p)
    equal_lengths.append(len(true_lemma_sent) == len(true_preds) == len(pred_lemma_sent))

print(all(equal_lengths))

total = 0
correct = 0

for true_sent, pred_sent in zip(true_labels, pred_labels):
    for true_label, pred_label in zip(true_sent, pred_sent):
        if true_label != -100:
            total += 1
            if true_label == pred_label:
                correct += 1

print(correct / total)
