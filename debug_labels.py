import json

from conllu import parse

from lemma_rules import apply_lemma_rule

TRUE_LABELS_PATH = "true_labels.txt"
PRED_LABELS_PATH = "predicted_labels.txt"
PRED_LEMMAS_PATH = "test_preds.txt"
TRUE_LEMMAS_PATH = "et_edt-ud-dev.conllu"

CONFIG_PATH = "seqlem_model/checkpoint-255/config.json"

true_labels = []
pred_labels = []
pred_lemmas = []


with open(CONFIG_PATH) as file:
    config = json.loads(file.read())

# print(config["id2label"])


with open(TRUE_LEMMAS_PATH) as file:
    gold_conll = parse(file.read())

forms = []
true_lemmas = []

for sent in gold_conll:
    tmp_lemma = []
    tmp_form = []
    for token in sent:
        tmp_lemma.append(token["lemma"].replace("_", "").replace("=", ""))
        tmp_form.append(token["form"])
    true_lemmas.append(tmp_lemma)
    forms.append(tmp_form)

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

for form_sent, true_lemma_sent, pred_lemma_sent, label_sent, true_sent in zip(
        forms, true_lemmas, pred_lemmas, actual_pred_labels, actual_true_labels
):
    assert len(form_sent) == len(true_lemma_sent) == len(pred_lemma_sent) == len(label_sent) == len(true_sent)
    for form_token, true_lemma_token, pred_lemma_token, pred_label, true_label in zip(
            form_sent, true_lemma_sent, pred_lemma_sent, label_sent, true_sent
    ):

        total += 1
        corr_lemma = False
        corr_label = False
        if true_lemma_token == pred_lemma_token:
            lemma_correct += 1
            corr_lemma = True
        if pred_label == true_label:
            label_correct += 1
            corr_label = True

        if corr_lemma != corr_label:
            print(f"Correct lemma: {corr_lemma}, correct label {corr_label}")
            print(
                form_token,
                true_lemma_token,
                pred_lemma_token,
                config["id2label"][str(pred_label)],
                config["id2label"][str(true_label)],
            )
            applied_predicted = apply_lemma_rule(form_token, config["id2label"][str(pred_label)])
            applied_true = apply_lemma_rule(form_token, config["id2label"][str(true_label)])
            print(applied_predicted)
            print(applied_true)
            print(f"Applying predicted and true rule has the same result: {applied_predicted == applied_true}")
            print()

print(lemma_correct / total)
print(label_correct / total)

