from geceval.modules.gec_module import GECModule


class PunctuationSeekerModule(GECModule):
    def __init__(self, language="en"):
        self.set_language(language)
        self.major_punctuation_marks = ".,!?"
        self.minor_punctuation_marks = "`'\"-;"
        self.supports_single_texts = True
        self.supports_references = False

    def score(self, text: str) -> float:
        for mark in self.major_punctuation_marks:
            if mark in text:
                return 1.0
        for mark in self.minor_punctuation_marks:
            if mark in text:
                return 0.5
        return 0.0

    def explain_errors(self, text: str):
        if self.score(text) == 0:
            return (
                True,
                "Punctuation mistakes found. Not found punctuation marks from set "
                + ", ".join(
                    [
                        str(x)
                        for x in self.major_punctuation_marks
                        + self.minor_punctuation_marks
                    ]
                ),
            )
        else:
            return False, "Punctuation marks present"

    def get_name(self):
        return "Punctuation seeker"
