import json
import os
from pathlib import Path
from typing import Dict, List


def get_file_path(base_path, idx, use_triton=True, input=True, language="en"):
    prefix = "triton" if use_triton else "raw"
    extension = "json" if use_triton else "txt"
    base_path = os.path.join(base_path, prefix, language)

    inference = "_inference" if not input else ""

    correct_path = os.path.join(base_path, f"correct_{idx}{inference}.{extension}")
    incorrect_path = os.path.join(base_path, f"incorrect_{idx}{inference}.{extension}")

    if os.path.exists(correct_path):
        return True, correct_path
    elif os.path.exists(incorrect_path):
        return False, incorrect_path

    return False, ""


def load_files_to_list_of_dicts(
    path, languages=["en"], use_triton_input=True, use_triton_output=True
):
    result = {}
    for language in languages:
        result[language] = []
        for idx in range(10000):
            is_correct, input_path = get_file_path(
                path, idx, use_triton=use_triton_input, input=True, language=language
            )
            _, output_path = get_file_path(
                path, idx, use_triton=use_triton_output, input=False, language=language
            )

            if len(output_path) == 0 and len(input_path) == 0:
                return result

            result[language].append(
                {
                    "idx": idx,
                    "marked_correct": 1 if is_correct else 0,
                    "before_correction": read_input(input_path),
                    "after__correction": read_output(output_path),
                }
            )

        return result


def load_files_to_dict_of_lists(
    path, languages=["en"], use_triton_input=True, use_triton_output=True
):
    result = {}
    for language in languages:
        result[language] = {
            "idx": [],
            "marked_correct": [],
            "before_correction": [],
            "after_correction": [],
        }

        for idx in range(10000):
            is_correct, input_path = get_file_path(
                path, idx, use_triton=use_triton_input, input=True, language=language
            )
            _, output_path = get_file_path(
                path, idx, use_triton=use_triton_output, input=False, language=language
            )

            if len(output_path) == 0 and len(input_path) == 0:
                return result

            result[language]["idx"].append(idx)
            result[language]["marked_correct"].append(1 if is_correct else 0)
            result[language]["before_correction"].append(read_input(input_path))
            result[language]["after_correction"].append(read_output(output_path))

    return result

def load_multi_llm_json_outputs(dir: str, prompt_list: List[str]) -> Dict:
    result = {}
    pathlist = Path(dir).glob('**/*.json')
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
                    id = elem['id']
                    if id not in result[lang]:
                        result[lang][id] = {
                            "marked_correct": elem['label'],
                            "text": elem['content'],
                            "corrections": []
                        }
                    result[lang][id]['corrections'].append({
                        "prompt_id": int(prompt_id),
                        "content": elem['processed'],
                        "model_name": model_id,
                    })
        with open("merged_multillm.json", 'w') as out:
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
