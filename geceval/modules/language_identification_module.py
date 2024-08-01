import math
import re

import fasttext
from huggingface_hub import hf_hub_download

from geceval.modules.gec_module import GECModule


class LanguageIdentificationModule(GECModule):
    def __init__(self, language="en"):
        self.language = language
        self.label_to_lang = {
            "__label__eng_Latn": "en",
            "__label__deu_Latn": "de",
            "__label__ita_Latn": "it",
            "__label__swe_Latn": "sv",
            "__label__ces_Latn": "cs",
        }
        self.lang_to_label = {v: k for k, v in self.label_to_lang.items()}
        self.language_label = self.lang_to_label[self.language]

        self.model_path = hf_hub_download(
            repo_id="facebook/fasttext-language-identification", filename="model.bin"
        )
        self.model = fasttext.load_model(self.model_path)
        self.supports_single_texts = True
        self.supports_references = True

    def score(self, text: str) -> float:
        text = re.sub(r"\n", " ", text)
        prediction = self.model.predict(text, k=300)
        language_idx = 0
        language_labels = prediction[0]
        language_probabilities = prediction[1]

        for idx in range(len(language_labels)):
            if language_labels[idx] == self.language_label:
                language_idx = idx
                break
        return language_probabilities[language_idx]

    def score_pair(self, text: str, reference: str):
        text_score = self.score(text)
        reference_score = self.score(reference)

        return 1 - math.fabs(text_score - reference_score)

    def explain_errors(self, text: str):
        suggestions = self.lt.check(text)
        label = True if len(suggestions) > 0 else False
        return label, ", ".join([str(x) for x in suggestions])

    def close(self):
        pass

    def get_name(self):
        return "Language identification"
