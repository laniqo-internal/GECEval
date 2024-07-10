import os
from enum import Enum
from typing import List, Optional

from geceval.modules.language_tool_module import LanguageToolModule
from geceval.modules.spell_checker_module import SpellcheckerModule
from geceval.triton_communication import (read_triton_output,
                                          read_triton_request)


class GECModules(Enum):
    LANGUAGE_TOOL = 1
    PERPLEXITY = 2
    ODDBALLNESS = 3
    AFTER_HOURS = 4
    CLASSIFIER = 5
    COLA = 6
    SPELLCHECKING = 7
    EXPECTED_CORRECTIONS = 8


class Scorer:
    def __init__(
        self, supported_modules: Optional[List[GECModules]], languages: List[str]
    ):
        self.supported_modules = supported_modules
        self.languages = languages
        self.evaluators = self.construct_evaluators(supported_modules, languages)

    def construct_evaluators(self, supported_modules, languages):
        evaluators = {}
        supported_modules = set(supported_modules)
        language = languages[0]

        if GECModules.SPELLCHECKING in supported_modules:
            evaluators[GECModules.SPELLCHECKING] = SpellcheckerModule(language)
        if GECModules.LANGUAGE_TOOL in supported_modules:
            evaluators[GECModules.LANGUAGE_TOOL] = LanguageToolModule(language)
        return evaluators

    def load_data_to_list_of_dicts(self, path: str):
        result = {}
        for language in self.languages:
            result[language] = []
            language_path = os.path.join(path, language)

            for idx in range(10000):
                correct_path = os.path.join(language_path, f"correct_{idx}.json")
                incorrect_path = os.path.join(language_path, f"incorrect_{idx}.json")
                if os.path.exists(correct_path):
                    result[language].append(
                        {
                            "idx": idx,
                            "marked_correct": 1,
                            "before_correction": read_triton_request(correct_path),
                            "after_correction": read_triton_output(correct_path),
                        }
                    )
                elif os.path.exists(incorrect_path):
                    result[language].append(
                        {
                            "idx": idx,
                            "marked_correct": 0,
                            "before_correction": read_triton_request(incorrect_path),
                            "after_correction": read_triton_output(incorrect_path),
                        }
                    )
                else:
                    continue
        return result

    def load_data_to_dict_of_lists(self, path: str):
        result = {}
        for language in self.languages:
            result[language] = {
                "idx": [],
                "marked_correct": [],
                "before_correction": [],
                "after_correction": [],
            }
            language_path = os.path.join(path, language)

            for idx in range(10000):
                correct_path = os.path.join(language_path, f"correct_{idx}.json")
                incorrect_path = os.path.join(language_path, f"incorrect_{idx}.json")
                if os.path.exists(correct_path):
                    result[language]["idx"].append(idx)
                    result[language]["marked_correct"].append(1)
                    result[language]["before_correction"].append(
                        read_triton_request(correct_path)
                    )
                    result[language]["after_correction"].append(
                        read_triton_output(correct_path)
                    )

                elif os.path.exists(incorrect_path):
                    result[language]["idx"].append(idx)
                    result[language]["marked_correct"].append(0)
                    result[language]["before_correction"].append(
                        read_triton_request(incorrect_path)
                    )
                    result[language]["after_correction"].append(
                        read_triton_output(incorrect_path)
                    )
                else:
                    continue
        return result

    def evaluate_test_set(self, path: str):
        data = self.load_data_to_dict_of_lists(path)
        for language in self.languages:
            for module in self.supported_modules:
                before_ave = self.evaluators[module].get_average_score(
                    data[language]["before_correction"]
                )
                after_ave = self.evaluators[module].get_average_score(
                    data[language]["after_correction"]
                )
                print(self.evaluators[module].get_name(), before_ave, after_ave)
                # for text in data[language]['after_correction']:
                #    has_error, text = self.evaluators[module].explain_errors(text)
                #    if has_error:
                #        print(text)

    def close(self):
        for module in self.supported_modules:
            self.evaluators[module].close()


if __name__ == "__main__":
    test_path = "/home/dwisniewski/test_gec/"

    modules = [
        GECModules.SPELLCHECKING,
        GECModules.LANGUAGE_TOOL,
    ]

    scorer = Scorer(supported_modules=modules, languages=["it"])
    scorer.evaluate_test_set(test_path)
    scorer.close()
