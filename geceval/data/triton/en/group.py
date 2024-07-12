import json
import os

import errant

annotator = errant.load("en")


def compare(orig, cor):
    orig = annotator.parse(orig)
    cor = annotator.parse(cor)
    edits = annotator.annotate(orig, cor)
    result = []
    for e in edits:
        result.append(
            {
                "orignal": f"{e.o_str} ({e.o_start}/{e.o_end})",
                "corrected": f"{e.c_str} ({e.c_start}/{e.c_end})",
                "type": e.type,
            }
        )
    return result


result = {"correct": [], "incorrect": []}
correct_tp = 0
incorrect_tp = 0
correct_errors = 0
incorrect_errors = 0


with open("output.json", "w") as out:
    for i in range(2500):
        if not os.path.exists(f"correct_{i}.json") and not os.path.exists(
            f"incorrect_{i}.json"
        ):
            continue

        path = ""
        if os.path.exists(f"correct_{i}.json"):
            path = f"correct_{i}.json"
            category = "correct"
        if os.path.exists(f"incorrect_{i}.json"):
            path = f"incorrect_{i}.json"
            category = "incorrect"

        with open(path, "r") as f:
            data_before_fix = json.load(f)["inputs"][0]["data"][0]

        with open(f"correct_{i}_inference.json", "r") as f:
            data_after_fix = json.load(f)["outputs"][0]["data"][0]
        result[category].append(
            {
                "original": data_before_fix.strip(),
                "fixed": data_after_fix.strip(),
                "eidts": compare(data_before_fix.strip(), data_after_fix.strip()),
            }
        )
    out.write(json.dumps(result, indent=4))
