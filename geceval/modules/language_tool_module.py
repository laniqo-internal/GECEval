import language_tool_python

from geceval.modules.gec_module import GECModule


class LanguageToolModule(GECModule):
    def __init__(self, language="en"):
        self.language_map = {"en": "en-US", "de": "de", "it": "it", "sv": "sv"}
        self.set_language(language)
        self.lt = language_tool_python.LanguageTool(self.language_map[self.language])

    def score(self, text: str) -> float:
        suggestions = len(self.lt.check(text))
        return 1.0 / (1.0 + suggestions)

    def explain_errors(self, text: str):
        suggestions = self.lt.check(text)
        label = True if len(suggestions) > 0 else False
        return label, ", ".join([str(x) for x in suggestions])

    def close(self):
        self.lt.close()

    def get_name(self):
        return "Language tool"
