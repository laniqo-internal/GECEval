import evaluate
import torch

from geceval.modules.gec_module import GECModule


class BleuRTModule(GECModule):
    def __init__(self, language: str, model_name: str = "BLEURT-20-D12"):
        self.model_name = model_name
        self.supports_single_texts = False
        self.supports_references = True
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = SentenceTransformer(model_name, device=self.device)
        self.model = evaluate.load("bleurt", self.model_name, module_type="metric")

    def score(self, text: str) -> float:
        pass

    def score_pair(self, text: str, reference: str):
        bleurt_scores = self.model.compute(predictions=[text], references=[reference])
        return bleurt_scores["scores"][0]

    def explain_errors(self, text: str):
        pass

    def close(self):
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def get_name(self):
        return "BleuRT"


if __name__ == "__main__":
    d = BleuRTModule("en")
    print(
        d.score_pair(
            "IN CASE YOU DECIDE TO CHANGE THE PROGRAM, WE SUGGEST ON TUSDAY GO TO THE SHOW AND ON WENSDAY INSTED FREE TIME, VISIT THE SCIENCE MUSEUM.",
            "IN CASE YOU DECIDE TO CHANGE THE PROGRAM, WE SUGGEST ON TUESDAY GO TO THE SHOW AND ON WEDNESDAY INSTEAD OF FREE TIME, VISIT THE SCIENCE MUSEUM.",
        )
    )
