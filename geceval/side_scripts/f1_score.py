import json
import lzma
import sys
from dataclasses import dataclass
from operator import itemgetter

import numpy as np
from nltk.tokenize import word_tokenize


def load_data(data_path):
    data = {}
    with lzma.open(data_path, "r") as f:
        json_bytes = f.read()
        utf_data = json_bytes.decode("utf-8")
        data = json.loads(utf_data)
    return data


path = sys.argv[1]
data = load_data(path)


@dataclass
class Correction:
    model_name: str
    prompt_id: int
    was_correct: bool
    corrected: bool


@dataclass
class Measure:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    _precision: float = 0.0
    _recall: float = 0.0
    _f1: float = 0.0

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2.0 * self.precision * self.recall / (self.precision + self.recall)


models = set()
prompts = set()

total_lengths_tokens = []
for language in data:  # language
    if language == 'cs':
        continue
    print(f"\nProcessing: {language}")
    results = []
    summary = {"TOTAL": 0, "CORRECT": 0}
    lengths = []
    lengths_tokens = []

    for label in data[language]:  # filename
        summary["TOTAL"] += 1
        current_el = data[language][label]
        is_correct = True if current_el["marked_correct"] == "correct" else False
        if is_correct:
            summary["CORRECT"] += 1

        before_correction = current_el["text"].strip().lower()
        lengths.append(len(before_correction))
        lengths_tokens.append(len(word_tokenize(before_correction)))
        total_lengths_tokens.append(len(word_tokenize(before_correction)))

        for elem in current_el["corrections"]:
            content = elem["content"].strip().lower()
            llm_corrected = True if before_correction != content else False
            results.append(
                Correction(
                    elem["model_name"], elem["prompt_id"], is_correct, llm_corrected
                )
            )
            models.add(elem["model_name"])
            prompts.add(elem["prompt_id"])

    print(summary)
    print(np.mean(lengths), np.std(lengths))
    print(np.mean(lengths_tokens), np.std(lengths_tokens))

    per_prompt_scores = {id: Measure() for id in prompts}
    per_model_scores = {id: Measure() for id in models}
    per_model_scores["always_zero"] = Measure()
    per_model_scores["always_one"] = Measure()

    for elem in results:
        prompt_id = elem.prompt_id
        model_name = elem.model_name

        if elem.was_correct and not elem.corrected:
            per_model_scores[model_name].tp += 1
            per_prompt_scores[prompt_id].tp += 1
        elif elem.was_correct and elem.corrected:
            per_model_scores[model_name].fp += 1
            per_prompt_scores[prompt_id].fp += 1
        elif not elem.was_correct and not elem.corrected:
            per_model_scores[model_name].fn += 1
            per_prompt_scores[prompt_id].fn += 1

        if model_name == "karen":
            if elem.was_correct:
                per_model_scores["always_one"].tp += 1
                per_model_scores["always_zero"].fn += 1
            else:
                per_model_scores["always_one"].fp += 1
                per_model_scores["always_zero"].tp += 1

    per_model_scores_sorted = []
    for k, v in per_model_scores.items():
        per_model_scores_sorted.append((k, v, v.f1))
    per_model_scores_sorted.sort(key=itemgetter(2), reverse=True)

    per_prompt_scores_sorted = []
    for k, v in per_prompt_scores.items():
        per_prompt_scores_sorted.append((k, v, v.f1))

    per_prompt_scores_sorted.sort(key=itemgetter(2), reverse=True)

    for model, _, _ in per_model_scores_sorted:
        print(
            f"Model: {model},\t\tp: {per_model_scores[model].precision},\tr: {per_model_scores[model].recall},\tf1: {per_model_scores[model].f1}"
        )

    for prompt, _, _ in per_prompt_scores_sorted:
        print(
            f"Prompt: {prompt},\t\tp: {per_prompt_scores[prompt].precision},\tr: {per_prompt_scores[prompt].recall},\tf1: {per_prompt_scores[prompt].f1}"
        )

print(np.mean(total_lengths_tokens), np.std(total_lengths_tokens))
