import json
import sys

with open(sys.argv[1], "r") as f:
    data = json.load(f)
    tp = 0
    fp = 0
    fn = 0
    assert len(data.keys()) == 1

    language = list(data.keys())[0]

    for elem in data[language]:
        if elem["marked_correct"] == 1:
            if elem["before_correction"] == elem["after__correction"]:
                tp += 1
            else:
                fn += 1
        elif elem["before_correction"] == elem["after__correction"]:
            fp += 1

    precision = 1.0 * tp / (tp + fp)
    recall = 1.0 * tp / (tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall)

    print(f"Precision:\t{precision}\nRecall:\t{recall}\nF1:\t{f1}")
