import os

min_id = 906
files = os.listdir(".")
for i in range(5000):
    current = min_id + i
    wrong_name = f"correct_{i}_inference.json"
    correct_name = f"incorrect_{i}_inference.json"
    if f"incorrect_{i}.json" in files and os.path.exists(wrong_name):
        os.rename(wrong_name, correct_name)

# print(os.listdir('.'))
