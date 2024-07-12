import json
import os


def read_triton_request(path: str, inputs=True):
    key = "inputs" if inputs else "outputs"
    with open(path, "r") as f:
        return json.load(f)[key][0]["data"][0]


for idx in range(10000):
    for language in ["cs", "de", "en", "sv", "it"]:
        for type in ["correct", "incorrect"]:
            path = f"../data/triton/{language}/{type}_{idx}_inference.json"
            if os.path.exists(path):
                text = read_triton_request(path, inputs=False)
                output_path = f"../data/raw/{language}/{type}_{idx}_inference.txt"
                with open(output_path, "w") as out:
                    out.write(text)
