import argparse
import json
from enum import Enum
from typing import List, Optional

from geceval.file_loaders import (
    load_files_to_dict_of_lists,
    load_files_to_list_of_dicts,
)
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
    def __init__(
        self, supported_modules: Optional[List[GECModules]], languages: List[str]
    ):
        assert (
            len(languages) == 1
        ), "Currently only one language at once can be processed"
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
        if GECModules.PUNCTUATION_SEEKER in supported_modules:
            evaluators[GECModules.PUNCTUATION_SEEKER] = PunctuationSeekerModule(
                language
            )
        return evaluators

    def check_test_set(self, path: str, use_triton: bool = True):
        data = load_files_to_dict_of_lists(
            path,
            use_triton_input=use_triton,
            use_triton_output=use_triton,
            languages=self.languages,
        )
        data_list_of_dicts = load_files_to_list_of_dicts(
            path,
            use_triton_input=use_triton,
            use_triton_output=use_triton,
            languages=self.languages,
        )
        language = self.languages[0]

        for language in self.languages:
            for module in self.supported_modules:
                before_ave = self.evaluators[module].get_average_score(
                    data[language]["before_correction"]
                )
                after_ave = self.evaluators[module].get_average_score(
                    data[language]["after_correction"]
                )
                print(self.evaluators[module].get_name(), before_ave, after_ave)

        for language_data in data_list_of_dicts:
            for elem in data_list_of_dicts[language_data]:
                elem["feedback"] = {}
                for module in self.supported_modules:
                    label, explanation = self.evaluators[module].explain_errors(
                        elem["after__correction"]
                    )
                    if label is False:
                        explanation = "no errors found"
                    elem["feedback"][self.evaluators[module].get_name()] = explanation
                # for text in data[language]['after_correction']:
                #    has_error, text = self.evaluators[module].explain_errors(text)
                #    if has_error:
                #        print(text)

        with open(f"{language}_raw_output.json", "w") as out:
            out.write(json.dumps(data_list_of_dicts, indent=4, ensure_ascii=False))

    def close(self):
        for module in self.supported_modules:
            self.evaluators[module].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--test-path", help="Base folder for data.", default="./data"
    )
    parser.add_argument("-l", "--language", help="Language to process.", default="en")
    parser.add_argument(
        "-t",
        "--use-triton-data",
        help="Use output serialized in Triton format.",
        action="store_true",
    )
    args = parser.parse_args()

    test_path = args.test_path
    language = args.language
    use_triton = args.use_triton_data

    modules = [
        GECModules.PUNCTUATION_SEEKER,
    ]
    if args.language != "cs":
        modules.append(GECModules.LANGUAGE_TOOL)

    if args.language not in ["cs", "sv"]:
        modules.append(GECModules.SPELLCHECKING)

    evaluator = Evaluator(supported_modules=modules, languages=[args.language])
    evaluator.check_test_set(args.test_path, use_triton=args.use_triton_data)
    evaluator.close()
