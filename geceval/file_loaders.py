import json
from pathlib import Path
from typing import Dict


def load_multi_llm_json_outputs(dir: str) -> Dict:
    result = {}
    pathlist = Path(dir).glob("**/*.json")
    for path in pathlist:
        raw_filename = path.stem
        model_id = raw_filename.split("_")[0]
        prompt_id = raw_filename.split("_")[-1]
        with open(str(path)) as f:
            data = json.loads(f.read())
            for lang in data:
                if lang not in result:
                    result[lang] = {}
                for elem in data[lang]:
                    id = elem["id"]
                    if id not in result[lang]:
                        result[lang][id] = {
                            "marked_correct": elem["label"],
                            "text": elem["content"],
                            "corrections": [],
                        }
                    result[lang][id]["corrections"].append(
                        {
                            "prompt_id": int(prompt_id),
                            "content": elem["processed"],
                            "model_name": model_id,
                        }
                    )
        with open("merged_multillm.json", "w") as out:
            out.write(json.dumps(result, indent=4, ensure_ascii=False))
    return result


def read_input(path: str):
    if path.endswith("json"):
        return read_triton_request(path)
    else:
        return read_raw_file(path)


def read_output(path: str):
    if path.endswith("json"):
        return read_triton_output(path)
    else:
        return read_raw_file(path)


def read_triton_request(path: str):
    with open(path, "r") as f:
        return json.load(f)["inputs"][0]["data"][0]


def read_triton_output(path: str):
    with open(path, "r") as f:
        return json.load(f)["outputs"][0]["data"][0]


def read_raw_file(path: str):
    with open(path, "r") as f:
        return f.read().strip()
