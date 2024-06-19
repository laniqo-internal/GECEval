from geceval.modules.gec_module import GECModule
from spellchecker import SpellChecker
from typing import List
import numpy as np


class SpellcheckerModule(GECModule):
    def __init__(self, language='en'):
        self.set_language(language)
        self._init_spellchecker()
    
    def _init_spellchecker(self):
        self.spellchecker = SpellChecker(language=self.language, case_sensitive=True)

    def score(self, text: str) -> float:
        tokens = self.spellchecker.split_words(text)
        misspelled = self.spellchecker.unknown(tokens)
        return 1.0 / (1.0 + len(misspelled))
    
    def get_average_score(self, texts: List[str]):
        scores = [self.score(text) for text in texts]
        return np.mean(scores)

    def explain_errors(self, text: str):
        tokens = self.spellchecker.split_words(text)
        misspelled = self.spellchecker.unknown(tokens)
        label = True if len(misspelled) > 0 else False
        return label, "Spelling mistakes found: " + ", ".join([f"{error}->{self.spellchecker.correction(error)}" for error in misspelled])

    def set_language(self, language: str):
        self.language = language
    
    def close(self):
        pass
    
    def get_name(self):
        return "Spell checker"