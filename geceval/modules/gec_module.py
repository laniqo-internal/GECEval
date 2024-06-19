from abc import ABC, abstractmethod
from typing import List


class GECModule(ABC):
    def set_language(self, language: str):
        self.language = language

    def compare_scores(self, text_before_correction: str, text_after_correction: str):
        before_score = self.score(text_before_correction)
        after_score = self.score(text_after_correction)
        return after_score - before_score

    @abstractmethod
    def score(self, text: str):
        pass

    @abstractmethod
    def explain_errors(self, text: str):
        pass