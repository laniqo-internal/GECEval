from geceval.modules.gec_module import GECModule
import language_tool_python
from typing import List
import numpy as np


class LanguageToolModule(GECModule):
    def __init__(self, language='en'):
        self.language_map = {
            "en": "en-US",
            "de": "de",
            "it": "it"
        }
        self.set_language(language)
        self._init_language_tool()
    
    def _init_language_tool(self):
        self.lt = language_tool_python.LanguageTool(
            self.language_map[self.language])

    def score(self, text: str) -> float:
        suggestions = len(self.lt.check(text))
        return 1.0 / (1.0 + suggestions)
    
    def get_average_score(self, texts: List[str]):
        scores = [self.score(text) for text in texts]
        return np.mean(scores)

    def explain_errors(self, text: str):
        suggestions = self.lt.check(text)
        label = True if len(suggestions) > 0 else False
        return label, "Languagetool mistakes found: " + ", ".join([str(x) for x in suggestions])

    def set_language(self, language: str):
        self.language = language
    
    def close(self):
        self.lt.close()

    def get_name(self):
        return "Language tool"