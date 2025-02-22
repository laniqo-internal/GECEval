from typing import List

from spellchecker import SpellChecker

from geceval.modules.gec_module import GECModule


class SpellcheckerModule(GECModule):
    def __init__(self, language="en"):
        self.set_language(language)
        self.spellchecker = SpellChecker(language=self.language, case_sensitive=True)
        self.supports_single_texts = True
        self.supports_references = False

    def score(self, text: str) -> float:
        tokens = self.spellchecker.split_words(text)
        misspelled = self.spellchecker.unknown(tokens)
        return 1.0 / (1.0 + len(misspelled))

    def score_pair(self, texts: List[str], references: List[str]):
        return 0.0

    def explain_errors(self, text: str):
        tokens = self.spellchecker.split_words(text)
        misspelled = self.spellchecker.unknown(tokens)
        label = True if len(misspelled) > 0 else False
        return label, ", ".join(
            [f"{error}->{self.spellchecker.correction(error)}" for error in misspelled]
        )

    def get_name(self):
        return "Spell checker"
