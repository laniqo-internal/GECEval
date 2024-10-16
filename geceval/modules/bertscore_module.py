from bert_score import BERTScorer

from geceval.modules.gec_module import GECModule


class BERTScoreModule(GECModule):
    def __init__(self, language="en", multilingual_model_for_en=True):
        self.language = language

        if language == "en" and not multilingual_model_for_en:
            self.scorer = BERTScorer(model_type="bert-base-uncased")
        else:
            self.scorer = BERTScorer(model_type="bert-base-multilingual-cased")

        self.supports_single_texts = False
        self.supports_references = True

    def score(self, text: str) -> float:
        pass

    def score_pair(self, text: str, reference: str):
        _, _, f1 = self.scorer.score([text], [reference])
        return f1.item()

    def explain_errors(self, text: str):
        pass

    def close(self):
        pass

    def get_name(self):
        return "BERT Score"
