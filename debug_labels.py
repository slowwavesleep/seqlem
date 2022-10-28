TRUE_LABELS_PATH = "true_labels.txt"
PRED_LABELS_PATH = "predicted_labels.txt"

true_labels = []
pred_labels = []

with open(TRUE_LABELS_PATH) as file:
    for line in file:
        labels = line.strip("\n").split(" ")
        true_labels.append([int(label) for label in labels])

with open(PRED_LABELS_PATH) as file:
    for line in file:
        labels = line.strip("\n").split(" ")
        pred_labels.append([int(label) for label in labels])

total = 0
correct = 0

for true_sent, pred_sent in zip(true_labels, pred_labels):
    for true_label, pred_label in zip(true_sent, pred_sent):
        if true_label != -100:
            total += 1
            if true_label == pred_label:
                correct += 1

print(correct / total)