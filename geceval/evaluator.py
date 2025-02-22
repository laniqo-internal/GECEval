import argparse
import json
import logging
import lzma
from collections import defaultdict
from enum import Enum
from typing import Dict

import numpy as np

from geceval.modules.bertscore_module import BERTScoreModule
from geceval.modules.bleurt_module import BleuRTModule
from geceval.modules.gleu import GleuModule
from geceval.modules.jaccard_distance import JaccardDistanceModule
from geceval.modules.language_switch_module import LanguageSwitchModule
from geceval.modules.language_tool_module import LanguageToolModule
from geceval.modules.levenshtein_module import LevenshteinModule
from geceval.modules.punctuation_seeker import PunctuationSeekerModule
from geceval.modules.sentence_bert_module import SentenceBertModule
from geceval.modules.spell_checker_module import SpellcheckerModule
from geceval.modules.token_count_distance import TokenCountDistanceModule

logging.basicConfig(
    filename="log.output.txt",
    filemode="a",
    format="%(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger()


class GECModules(Enum):
    LANGUAGE_TOOL = 1
    SPELLCHECKING = 2
    PUNCTUATION_SEEKER = 3
    BERTSCORE = 4
    LEVENSHTEIN = 5
    JACCARD = 6
    TOKEN_COUNT_DISTANCE = 7
    LANGUAGE_SWITCH = 8
    SENTENCE_BERT = 9
    BLEURT = 10
    GLEU = 11


def log_screen_file(text):
    print(text)
    logger.log(logging.INFO, text)


class Evaluator:
    def __init__(self):
        self.supported_languages = ["en", "cs", "sv", "de", "it"]

        used_modules = {
            GECModules.LANGUAGE_TOOL,
            GECModules.PUNCTUATION_SEEKER,
            GECModules.SPELLCHECKING,
            GECModules.LANGUAGE_SWITCH,
            GECModules.LEVENSHTEIN,
            GECModules.TOKEN_COUNT_DISTANCE,
            GECModules.JACCARD,
            GECModules.BERTSCORE,
            GECModules.SENTENCE_BERT,
            GECModules.BLEURT,
            GECModules.GLEU
        }

        self.per_language_modules = {
            lang: used_modules for lang in self.supported_languages
        }

        self._remove_unsupported_tools()
        self.evaluators = self._construct_evaluators()

    def _remove_unsupported_tools(self):
        if "cs" in self.supported_languages:
            self.per_language_modules["cs"] = self.per_language_modules["cs"] - {
                GECModules.LANGUAGE_TOOL,
                GECModules.SPELLCHECKING,
            }
        if "sv" in self.supported_languages:
            self.per_language_modules["sv"] = self.per_language_modules["sv"] - {
                GECModules.SPELLCHECKING
            }

    def _construct_evaluators(self):
        evaluators = {}

        construction_map = {
            GECModules.SPELLCHECKING: SpellcheckerModule,
            GECModules.LANGUAGE_TOOL: LanguageToolModule,
            GECModules.PUNCTUATION_SEEKER: PunctuationSeekerModule,
            GECModules.LANGUAGE_SWITCH: LanguageSwitchModule,
            GECModules.LEVENSHTEIN: LevenshteinModule,
            GECModules.TOKEN_COUNT_DISTANCE: TokenCountDistanceModule,
            GECModules.JACCARD: JaccardDistanceModule,
            GECModules.BERTSCORE: BERTScoreModule,
            GECModules.SENTENCE_BERT: SentenceBertModule,
            GECModules.BLEURT: BleuRTModule,
            GECModules.GLEU: GleuModule
        }

        for language in self.supported_languages:
            print(f"Constructing evaluators for {language}...")
            evaluators[language] = {}

            for module, constructor in construction_map.items():
                if module in self.per_language_modules[language]:
                    evaluators[language][module] = constructor(language)

        return evaluators

    def _collect_original_texts(self, lang_data):
        return [v["text"] for _, v in lang_data.items()]

    def _collect_corrected_texts(self, lang_data, prompt_id, model_name):
        result = []

        for _, entry in lang_data.items():
            for correction in entry["corrections"]:
                if (
                    correction["prompt_id"] == prompt_id
                    and correction["model_name"] == model_name
                ):
                    result.append(correction["content"])
        return result

    def _get_prompt_ids(self, data):
        prompt_ids = set()

        for _, lang_obj in data.items():
            for _, entry in lang_obj.items():
                for correction in entry["corrections"]:
                    prompt_ids.add(correction["prompt_id"])
        return prompt_ids

    def _get_model_names(self, data):
        model_names = set()

        for _, lang_obj in data.items():
            for _, entry in lang_obj.items():
                for correction in entry["corrections"]:
                    model_names.add(correction["model_name"])
        return model_names

    def load_dataset(self, data_path: str) -> Dict:
        data = {}
        with lzma.open(data_path, "r") as f:
            json_bytes = f.read()
            utf_data = json_bytes.decode("utf-8")
            data = json.loads(utf_data)
        return data

    def _aggregate_prompts(
        self,
        corrected_scores,
        prompt_ids,
        model_names,
        language,
        module,
        original_avg_score,
        use_comparative_metrics,
    ):
        for prompt_id in prompt_ids:
            avg_prompt_score = np.mean(
                [corrected_scores[prompt_id][model] for model in model_names]
            )

            log_text = ""
            log_text += f"Aggregate over models Language: {language}"
            log_text += f"\t prompt: {prompt_id}"
            log_text += f"\t metric: {self.evaluators[language][module].get_name()}"
            if not use_comparative_metrics:
                log_text += f"\t score: {original_avg_score}->{avg_prompt_score}"
            else:
                log_text += f"\t score: {avg_prompt_score}"

            log_screen_file(log_text)

    def _aggregate_models(
        self,
        corrected_scores,
        prompt_ids,
        model_names,
        language,
        module,
        original_avg_score,
        use_comparative_metrics,
    ):
        for model_name in model_names:
            avg_model_score = np.mean(
                [corrected_scores[prompt_id][model_name] for prompt_id in prompt_ids]
            )

            log_text = ""
            log_text += f"Aggregate over prompts Language: {language}"
            log_text += f"\t model_name: {model_name}"
            log_text += f"\t metric: {self.evaluators[language][module].get_name()}"
            if not use_comparative_metrics:
                log_text += f"\t score: {original_avg_score}->{avg_model_score}"
            else:
                log_text += f"\t score: {avg_model_score}"
            log_screen_file(log_text)

    def _requirements_check_failed(self, use_comparative_metrics, evaluator):
        if use_comparative_metrics and not evaluator.supports_references:
            return True
        if not use_comparative_metrics and not evaluator.supports_single_texts:
            return True
        return False

    def evaluate(
        self,
        json_path,
        use_comparative_metrics=False,
        prompt_ids=None,
        model_names=None,
        languages=None,
    ):
        data = self.load_dataset(json_path)

        if not prompt_ids:
            prompt_ids = self._get_prompt_ids(data)
        if not model_names:
            model_names = self._get_model_names(data)
        languages = languages if languages else self.supported_languages

        for language in languages:
            original_texts = self._collect_original_texts(data[language])

            for module in self.per_language_modules[language]:
                evaluator = self.evaluators[language][module]

                if self._requirements_check_failed(use_comparative_metrics, evaluator):
                    continue

                log_screen_file("\n" + "-" * 80)

                if not use_comparative_metrics and evaluator.supports_single_texts:
                    original_avg_score = evaluator.get_average_score(original_texts)
                else:
                    original_avg_score = 0.0

                corrected_scores = defaultdict(dict)

                for prompt_id in prompt_ids:
                    for model_name in model_names:
                        corrected_texts = self._collect_corrected_texts(
                            data[language], prompt_id=prompt_id, model_name=model_name
                        )
                        if (
                            not use_comparative_metrics
                            and evaluator.supports_single_texts
                        ):
                            corrected_avg, scores = evaluator.get_average_score(
                                corrected_texts
                            )
                        else:
                            corrected_avg, scores = evaluator.get_average_pair_score(
                                original_texts, corrected_texts
                            )

                        corrected_scores[prompt_id][model_name] = corrected_avg
                        log_screen_file(
                            f"Language: {language}\t Model: {model_name}\t prompt: {prompt_id}\t metric: {self.evaluators[language][module].get_name()}\t score: {corrected_avg}"
                        )
                self._aggregate_prompts(
                    corrected_scores,
                    prompt_ids,
                    model_names,
                    language,
                    module,
                    original_avg_score,
                    use_comparative_metrics,
                )
                self._aggregate_models(
                    corrected_scores,
                    prompt_ids,
                    model_names,
                    language,
                    module,
                    original_avg_score,
                    use_comparative_metrics,
                )

    def close(self):
        for language in self.supported_languages:
            print(f"Closing evaluators for {language}...")
            for module in self.evaluators[language].keys():
                self.evaluators[language][module].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment_output_path",
        help="JSON data.",
        default="./data/merged_multillm.json.xz",
    )

    parser.add_argument(
        "-m",
        "--models",
        help="Models to evaluate, comma-separated",
        default="aya,bloom,gemma,gemma2B,karen,karencreative,llama31,mistral,openchat,phi,qwen,smol,tower7B1,yi,xglm"
    )

    parser.add_argument(
        "-l",
        "--languages",
        help="Languages to evaluate, comma-separated",
        default="en,de,it,sv"
    )

    parser.add_argument(
        "-p",
        "--prompt_ids",
        help="Prompt ids to use, comma-separated",
        default="2"
    )

    args = parser.parse_args()
    experiment_path = args.experiment_output_path
    model_names = args.models.split(",")
    languages = args.languages.split(",")
    prompt_ids = [int(p) for p in args.prompt_ids.split(",")]

    evaluator = Evaluator()
    evaluator.evaluate(
        experiment_path,
        use_comparative_metrics=True,
        prompt_ids=prompt_ids,
        languages=languages,
        model_names=model_names
    )
    evaluator.close()
