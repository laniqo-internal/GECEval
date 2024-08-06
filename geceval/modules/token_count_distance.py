import math

import nltk
from nltk.tokenize import word_tokenize

from geceval.modules.gec_module import GECModule


class TokenCountDistanceModule(GECModule):
    def __init__(self, language="en"):
        self.language = language
        self.supports_single_texts = False
        self.supports_references = True
        nltk.download("punkt")

    def score(self, text: str) -> float:
        pass

    def score_pair(self, text: str, reference: str):
        text_tokens = len(word_tokenize(text))
        reference_tokens = len(word_tokenize(reference))
        max_len = text_tokens if text_tokens > reference_tokens else reference_tokens

        return 1 - (math.fabs(text_tokens - reference_tokens) / max_len)

    def explain_errors(self, text: str):
        pass

    def close(self):
        pass

    def get_name(self):
        return "Token count distance"
