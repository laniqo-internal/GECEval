import json
import os

result = {"correct": [], "incorrect": []}
correct_tp = 0
incorrect_tp = 0
correct_errors = 0
incorrect_errors = 0


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
        # print(data['inputs'][0]['data'][0])

    with open(f"{path[:-5]}_inference.json", "r") as f:
        data_after_fix = json.load(f)["outputs"][0]["data"][0]
        # print(data['outputs'][0]['data'][0])

    result[category].append(
        {"before_fix": data_before_fix, "after_fix": data_after_fix}
    )

for d in result["correct"]:
    if d["before_fix"] == d["after_fix"]:
        correct_tp += 1
    else:
        correct_errors += 1
        print(f"Expected: {d['before_fix']}")
        print(f"Produced: {d['after_fix']}")
        print("\n")

for d in result["incorrect"]:
    if d["before_fix"] == d["after_fix"]:
        incorrect_errors += 1
    else:
        incorrect_tp += 1
        # print(d['before_fix'])
        # print(d['after_fix'])
        # print("\n")

print(correct_tp, correct_errors, incorrect_tp, incorrect_errors)
