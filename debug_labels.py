TRUE_LABELS_PATH = "true_labels.txt"
PRED_LABELS_PATH = "predicted_labels.txt"
PRED_LEMMAS_PATH = "test_preds.txt"

true_labels = []
pred_labels = []
pred_lemmas = []

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

for lemma_sent, label_sent in zip(pred_lemmas, pred_labels):
    true_label_length = len([label for label in label_sent if label != -100])
    print(len(label_sent), true_label_length, len(lemma_sent))


total = 0
correct = 0

for true_sent, pred_sent in zip(true_labels, pred_labels):
    for true_label, pred_label in zip(true_sent, pred_sent):
        if true_label != -100:
            total += 1
            if true_label == pred_label:
                correct += 1

print(correct / total)