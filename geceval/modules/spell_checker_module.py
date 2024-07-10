from spellchecker import SpellChecker

from geceval.modules.gec_module import GECModule


class SpellcheckerModule(GECModule):
    def __init__(self, language="en"):
        self.set_language(language)
        self.spellchecker = SpellChecker(language=self.language, case_sensitive=True)

    def score(self, text: str) -> float:
        tokens = self.spellchecker.split_words(text)
        misspelled = self.spellchecker.unknown(tokens)
        return 1.0 / (1.0 + len(misspelled))

    def explain_errors(self, text: str):
        tokens = self.spellchecker.split_words(text)
        misspelled = self.spellchecker.unknown(tokens)
        label = True if len(misspelled) > 0 else False
        return label, "Spelling mistakes found: " + ", ".join(
            [f"{error}->{self.spellchecker.correction(error)}" for error in misspelled]
        )

    def get_name(self):
        return "Spell checker"
