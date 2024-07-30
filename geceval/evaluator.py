import argparse
import json
from enum import Enum

from geceval.modules.language_tool_module import LanguageToolModule
from geceval.modules.punctuation_seeker import PunctuationSeekerModule
from geceval.modules.spell_checker_module import SpellcheckerModule


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


class Evaluator:
    def __init__(self):
        self.supported_languages = ["en", "cs", "sv", "de", "it"]

        referenceless_modules = {
            GECModules.LANGUAGE_TOOL,
            GECModules.PUNCTUATION_SEEKER,
            GECModules.SPELLCHECKING,
        }

        self.per_language_modules = {
            lang: referenceless_modules for lang in self.supported_languages
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

    def evaluate(self, data_path: str):
        data = {}
        with open(data_path, "r") as f:
            data = json.loads(f.read())

        self.prompt_ids = self.get_prompt_ids(data)
        self.model_names = self.get_model_names(data)

        for language in self.supported_languages:
            original_texts = self.collect_original_texts(data[language])

            for module in self.per_language_modules[language]:
                original_ave = self.evaluators[language][module].get_average_score(
                    original_texts
                )

            print("-" * 80)
            for prompt_id in self.prompt_ids:
                for model_name in self.model_names:
                    corrected_texts = self.collect_corrected_texts(
                        data[language], prompt_id=prompt_id, model_name=model_name
                    )
                    corrected_ave = self.evaluators[language][module].get_average_score(
                        corrected_texts
                    )
                    print(
                        f"Language: {language}\t Model: {model_name}\t prompt: {prompt_id}\t metric: {self.evaluators[language][module].get_name()}\t score: {original_ave}->{corrected_ave}"
                    )

    def evaluate_referenceless(self, data_path: str):
        pass

    def evaluate_with_references(self):
        pass

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
        default="./data/merged_multillm.json",
    )
    args = parser.parse_args()
    experiment_path = args.experiment_output_path

    evaluator = Evaluator()
    # evaluator.check_test_set(args.test_path, use_triton=args.use_triton_data)
    evaluator.evaluate(experiment_path)
    evaluator.close()
