import nltk
from nltk.tokenize import word_tokenize

from geceval.modules.gec_module import GECModule


class JaccardDistanceModule(GECModule):
    def __init__(self, language="en"):
        nltk.download("punkt")
        self.language = language
        self.supports_single_texts = False
        self.supports_references = True

    def score(self, text: str) -> float:
        pass

    def score_pair(self, text: str, reference: str):
        text_tokens = set(word_tokenize(text))
        reference_tokens = set(word_tokenize(reference))

        return (
            1.0
            * len(text_tokens & reference_tokens)
            / len(text_tokens | reference_tokens)
        )

    def explain_errors(self, text: str):
        pass

    def close(self):
        pass

    def get_name(self):
        return "Jaccard distance"
