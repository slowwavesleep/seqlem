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

actual_pred_labels = []
actual_true_labels = []

for pred_sent, true_sent in zip(pred_labels, true_labels):
    tmp_pred = []
    tmp_true = []
    for p, t in zip(pred_sent, true_sent):
        if t != -100:
            tmp_pred.append(p)
            tmp_true.append(t)
    actual_pred_labels.append(tmp_pred)
    actual_true_labels.append(tmp_true)

total = 0
lemma_correct = 0
label_correct = 0

print(len(true_lemmas), len(pred_lemmas), len(actual_true_labels), len(actual_pred_labels))

for true_lemma_sent, pred_lemma_sent, label_sent, true_sent in zip(
        true_lemmas, pred_lemmas, actual_pred_labels, actual_true_labels
):
    print(total)
    # assert len(true_lemma_sent) == len(pred_lemma_sent) == len(label_sent) == len(true_sent)
    for true_lemma_token, pred_lemma_token, pred_label, true_label in zip(
            true_lemma_sent, pred_lemma_sent, label_sent, true_sent
    ):

        total += 1
        if true_lemma_token == pred_lemma_token:
            lemma_correct += 1
        if pred_label == true_label:
            label_correct += 1

print(total)
# print(lemma_correct / total)
# print(label_correct / total)

# equal_lengths = []

# for true_lemma_sent, pred_lemma_sent, label_sent, true_sent in zip(true_lemmas, pred_lemmas, pred_labels, true_labels):
#     true_preds = []
#     for p, t in zip(label_sent, true_sent):
#         if t != -100:
#             true_preds.append(p)
#     equal_lengths.append(len(true_lemma_sent) == len(true_preds) == len(pred_lemma_sent))



# print(all(equal_lengths))
#
# total = 0
# correct = 0
#
# for true_sent, pred_sent in zip(true_labels, pred_labels):
#     for true_label, pred_label in zip(true_sent, pred_sent):
#         if true_label != -100:
#             total += 1
#             if true_label == pred_label:
#                 correct += 1
#
