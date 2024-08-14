from abc import ABC, abstractmethod
from typing import List

import numpy as np


class GECModule(ABC):
    """
    Abstract class for unsupervised grammatical correction evaluation
    """

    def set_language(self, language: str):
        """Set language so that e.g., spellchecker knows how to operate"""
        self.language = language

    def compare_scores(self, text_before_correction: str, text_after_correction: str):
        """Compare two text to see if metrics changed after correction"""
        before_score = self.score(text_before_correction)
        after_score = self.score(text_after_correction)
        return after_score - before_score

    def get_average_score(self, texts: List[str]):
        """Score a given set of texts based on a given GEC metric"""
        scores = [self.score(text) for text in texts]
        return np.mean(scores), scores

    def get_average_pair_score(self, texts: List[str], references: List[str]):
        """Score a given set of texts based on a given GEC metric"""
        scores = [
            self.score_pair(text, reference)
            for text, reference in zip(texts, references)
        ]
        return np.mean(scores), scores

    def close(self):
        pass

    @abstractmethod
    def score(self, text: str):
        """Score a given text based on a given GEC metric"""
        pass

    @abstractmethod
    def score_pair(self, text: str, reference: str):
        """Score a given text in relation to reference based on a given GEC metric"""
        pass

    @abstractmethod
    def explain_errors(self, text: str):
        """Explain grammatical errors"""
        pass

    @abstractmethod
    def get_name(self):
        """Get name of the module"""
        pass
