import argparse
import json
import logging
import lzma
from enum import Enum
from typing import Dict

import numpy as np

from geceval.modules.bertscore_module import BERTScoreModule
from geceval.modules.jaccard_distance import JaccardDistanceModule
from geceval.modules.language_identification_module import LanguageIdentificationModule
from geceval.modules.language_tool_module import LanguageToolModule
from geceval.modules.levenshtein_module import LevenshteinModule
from geceval.modules.punctuation_seeker import PunctuationSeekerModule
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
    PERPLEXITY = 2
    ODDBALLNESS = 3
    AFTER_HOURS = 4
    CLASSIFIER = 5
    COLA = 6
    SPELLCHECKING = 7
    EXPECTED_CORRECTIONS = 8
    PUNCTUATION_SEEKER = 9
    LANGUAGE_PRESERVATION = 10
    LEVENSHTEIN = 11
    TOKEN_COUNT_DISTANCE = 12
    JACCARD = 13
    BERTSCORE = 14


class Evaluator:
    def __init__(self):
        self.supported_languages = ["en", "cs", "sv", "de", "it"]

        used_modules = {
            # GECModules.LANGUAGE_TOOL,
            # GECModules.PUNCTUATION_SEEKER,
            # GECModules.SPELLCHECKING,
            # GECModules.LANGUAGE_PRESERVATION,
            # GECModules.LEVENSHTEIN,
            # GECModules.TOKEN_COUNT_DISTANCE,
            # GECModules.JACCARD,
            GECModules.BERTSCORE
        }

        self.per_language_modules = {
            lang: used_modules for lang in self.supported_languages
        }

        if "cs" in self.supported_languages:
            self.per_language_modules["cs"] = self.per_language_modules["cs"] - {
                GECModules.LANGUAGE_TOOL,
                GECModules.SPELLCHECKING,
            }
        if "sv" in self.supported_languages:
            self.per_language_modules["sv"] = self.per_language_modules["sv"] - {
                GECModules.SPELLCHECKING
            }

        self.evaluators = self._construct_evaluators()

    def _construct_evaluators(self):
        evaluators = {}

        for language in self.supported_languages:
            print(f"Constructing evaluators for {language}...")
            evaluators[language] = {}

            if GECModules.SPELLCHECKING in self.per_language_modules[language]:
                evaluators[language][GECModules.SPELLCHECKING] = SpellcheckerModule(
                    language
                )
            if GECModules.LANGUAGE_TOOL in self.per_language_modules[language]:
                evaluators[language][GECModules.LANGUAGE_TOOL] = LanguageToolModule(
                    language
                )
            if GECModules.PUNCTUATION_SEEKER in self.per_language_modules[language]:
                evaluators[language][
                    GECModules.PUNCTUATION_SEEKER
                ] = PunctuationSeekerModule(language)
            if GECModules.LANGUAGE_PRESERVATION in self.per_language_modules[language]:
                evaluators[language][
                    GECModules.LANGUAGE_PRESERVATION
                ] = LanguageIdentificationModule(language)

            if GECModules.LEVENSHTEIN in self.per_language_modules[language]:
                evaluators[language][GECModules.LEVENSHTEIN] = LevenshteinModule(
                    language
                )

            if GECModules.TOKEN_COUNT_DISTANCE in self.per_language_modules[language]:
                evaluators[language][
                    GECModules.TOKEN_COUNT_DISTANCE
                ] = TokenCountDistanceModule(language=language)

            if GECModules.JACCARD in self.per_language_modules[language]:
                evaluators[language][GECModules.JACCARD] = JaccardDistanceModule(
                    language=language
                )

            if GECModules.BERTSCORE in self.per_language_modules[language]:
                evaluators[language][GECModules.BERTSCORE] = BERTScoreModule(
                    language=language
                )

        return evaluators

    def collect_original_texts(self, lang_data):
        return [v["text"] for _, v in lang_data.items()]

    def collect_corrected_texts(self, lang_data, prompt_id, model_name):
        result = []

        for _, entry in lang_data.items():
            for correction in entry["corrections"]:
                if (
                    correction["prompt_id"] == prompt_id
                    and correction["model_name"] == model_name
                ):
                    result.append(correction["content"])
        return result

    def get_prompt_ids(self, data):
        prompt_ids = set()

        for _, lang_obj in data.items():
            for _, entry in lang_obj.items():
                for correction in entry["corrections"]:
                    prompt_ids.add(correction["prompt_id"])
        return prompt_ids

    def get_model_names(self, data):
        model_names = set()

        for _, lang_obj in data.items():
            for _, entry in lang_obj.items():
                for correction in entry["corrections"]:
                    model_names.add(correction["model_name"])
        return model_names

    def evaluate(
        self,
        data_path: str,
        use_referenceless_metrics=True,
        use_metrics_with_references=True,
    ):
        # if use_referenceless_metrics:
        #   self.evaluate_referenceless(data_path)

        if use_metrics_with_references:
            self.evaluate_with_references(data_path)

    def load_dataset(self, data_path: str) -> Dict:
        data = {}
        with lzma.open(data_path, "r") as f:
            json_bytes = f.read()
            utf_data = json_bytes.decode("utf-8")
            data = json.loads(utf_data)
        return data

    def evaluate_referenceless(self, data_path: str):
        data = self.load_dataset(data_path)
        self.prompt_ids = self.get_prompt_ids(data)
        self.model_names = self.get_model_names(data)

        for language in self.supported_languages:
            original_texts = self.collect_original_texts(data[language])

            for module in self.per_language_modules[language]:
                if not self.evaluators[language][module].supports_single_texts:
                    continue

                original_ave = self.evaluators[language][module].get_average_score(
                    original_texts
                )

                print("")
                print("-" * 80)
                logger.log(logging.INFO, "")
                logger.log(logging.INFO, "-" * 80)

                corrected_scores = {}

                for prompt_id in self.prompt_ids:
                    for model_name in self.model_names:
                        corrected_texts = self.collect_corrected_texts(
                            data[language], prompt_id=prompt_id, model_name=model_name
                        )
                        corrected_ave = self.evaluators[language][
                            module
                        ].get_average_score(corrected_texts)
                        if prompt_id not in corrected_scores:
                            corrected_scores[prompt_id] = {}
                        if model_name not in corrected_scores[prompt_id]:
                            corrected_scores[prompt_id][model_name] = corrected_ave

                        print(
                            f"Language: {language}\t Model: {model_name}\t prompt: {prompt_id}\t metric: {self.evaluators[language][module].get_name()}\t score: {original_ave}->{corrected_ave}"
                        )

                        logger.log(
                            logging.INFO,
                            f"Language: {language}\t Model: {model_name}\t prompt: {prompt_id}\t metric: {self.evaluators[language][module].get_name()}\t score: {original_ave}->{corrected_ave}",
                        )

                for prompt_id in self.prompt_ids:
                    ave_prompt = np.mean(
                        [
                            corrected_scores[prompt_id][model]
                            for model in self.model_names
                        ]
                    )
                    print(
                        f"Aggregate over models Language: {language}\t prompt: {prompt_id}\t metric: {self.evaluators[language][module].get_name()}\t score: {original_ave}->{ave_prompt}"
                    )
                    logger.log(
                        logging.INFO,
                        f"Aggregate over models Language: {language}\t prompt: {prompt_id}\t metric: {self.evaluators[language][module].get_name()}\t score: {original_ave}->{ave_prompt}",
                    )
                for model_name in self.model_names:
                    ave_model = np.mean(
                        [
                            corrected_scores[prompt_id][model_name]
                            for prompt_id in self.prompt_ids
                        ]
                    )
                    print(
                        f"Aggregate over prompts Language: {language}\t model_name: {model_name}\t metric: {self.evaluators[language][module].get_name()}\t score: {original_ave}->{ave_model}"
                    )
                    logger.log(
                        logging.INFO,
                        f"Aggregate over prompts Language: {language}\t model_name: {model_name}\t metric: {self.evaluators[language][module].get_name()}\t score: {original_ave}->{ave_model}",
                    )

    def evaluate_with_references(self, data_path: str):
        # BERTScore
        # chcemy ranking, nie wyniki (!)
        #
        print("REFERENCES")
        data = self.load_dataset(data_path)
        self.prompt_ids = self.get_prompt_ids(data)
        self.model_names = self.get_model_names(data)

        for language in self.supported_languages:
            original_texts = self.collect_original_texts(data[language])

            for module in self.per_language_modules[language]:
                if not self.evaluators[language][module].supports_references:
                    continue

                print("")
                print("-" * 80)
                logger.log(logging.INFO, "")
                logger.log(logging.INFO, "-" * 80)

                corrected_scores = {}

                self.prompt_ids = [2]  #!!!!!

                for prompt_id in self.prompt_ids:
                    for model_name in self.model_names:
                        corrected_texts = self.collect_corrected_texts(
                            data[language], prompt_id=prompt_id, model_name=model_name
                        )
                        corrected_ave = self.evaluators[language][
                            module
                        ].get_average_pair_score(original_texts, corrected_texts)
                        if prompt_id not in corrected_scores:
                            corrected_scores[prompt_id] = {}
                        if model_name not in corrected_scores[prompt_id]:
                            corrected_scores[prompt_id][model_name] = corrected_ave

                        print(
                            f"Language: {language}\t Model: {model_name}\t prompt: {prompt_id}\t metric: {self.evaluators[language][module].get_name()}\t score: {corrected_ave}"
                        )

                        logger.log(
                            logging.INFO,
                            f"Language: {language}\t Model: {model_name}\t prompt: {prompt_id}\t metric: {self.evaluators[language][module].get_name()}\t score: {corrected_ave}",
                        )

                for prompt_id in self.prompt_ids:
                    ave_prompt = np.mean(
                        [
                            corrected_scores[prompt_id][model]
                            for model in self.model_names
                        ]
                    )
                    print(
                        f"Aggregate over models Language: {language}\t prompt: {prompt_id}\t metric: {self.evaluators[language][module].get_name()}\t score: {ave_prompt}"
                    )
                    logger.log(
                        logging.INFO,
                        f"Aggregate over models Language: {language}\t prompt: {prompt_id}\t metric: {self.evaluators[language][module].get_name()}\t score: {ave_prompt}",
                    )
                for model_name in self.model_names:
                    ave_model = np.mean(
                        [
                            corrected_scores[prompt_id][model_name]
                            for prompt_id in self.prompt_ids
                        ]
                    )
                    print(
                        f"Aggregate over prompts Language: {language}\t model_name: {model_name}\t metric: {self.evaluators[language][module].get_name()}\t score: {ave_model}"
                    )
                    logger.log(
                        logging.INFO,
                        f"Aggregate over prompts Language: {language}\t model_name: {model_name}\t metric: {self.evaluators[language][module].get_name()}\t score: {ave_model}",
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
    args = parser.parse_args()
    experiment_path = args.experiment_output_path

    evaluator = Evaluator()
    # evaluator.check_test_set(args.test_path, use_triton=args.use_triton_data)
    evaluator.evaluate(experiment_path)
    evaluator.close()
