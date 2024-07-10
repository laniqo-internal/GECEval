import json


def read_triton_request(path: str):
    with open(path, "r") as f:
        return json.load(f)["inputs"][0]["data"][0]


def read_triton_output(path: str):
    with open(f"{path[:-5]}_inference.json", "r") as f:
        return json.load(f)["outputs"][0]["data"][0]
