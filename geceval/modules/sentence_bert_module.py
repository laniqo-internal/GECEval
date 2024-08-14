import torch
from sentence_transformers import SentenceTransformer, util

from geceval.modules.gec_module import GECModule


class SentenceBertModule(GECModule):
    def __init__(
            self,
            language: str,
            model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.supports_single_texts = False
        self.supports_references = True
        self.language = language
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)

    def score(self, text: str) -> float:
        pass

    def score_pair(self, text: str, reference: str):
        e1, e2 = self.model.encode([text, reference], show_progress_bar = False)
        cos_sim = util.cos_sim(e1, e2)
        return cos_sim.item()

    def explain_errors(self, text: str):
        pass

    def close(self):
        del self.model
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def get_name(self):
        return "Sentence Bert"

if __name__ == '__main__':
    d = SentenceBertModule("en")
    print(d.score_pair(
        "IN CASE YOU DECIDE TO CHANGE THE PROGRAM, WE SUGGEST ON TUSDAY GO TO THE SHOW AND ON WENSDAY INSTED FREE TIME, VISIT THE SCIENCE MUSEUM.",
        "IN CASE YOU DECIDE TO CHANGE THE PROGRAM, WE SUGGEST ON TUESDAY GO TO THE SHOW AND ON WEDNESDAY INSTEAD OF FREE TIME, VISIT THE SCIENCE MUSEUM."
    ))
