from Levenshtein import distance

from geceval.modules.gec_module import GECModule


class LevenshteinModule(GECModule):
    def __init__(self, language="en"):
        self.language = language
        self.supports_single_texts = False
        self.supports_references = True

    def score(self, text: str) -> float:
        pass

    def score_pair(self, text: str, reference: str):
        texts_distance = distance(text, reference)
        return 1.0 / (1.0 + texts_distance)

    def explain_errors(self, text: str):
        pass

    def close(self):
        pass

    def get_name(self):
        return "Levensthein distance"
