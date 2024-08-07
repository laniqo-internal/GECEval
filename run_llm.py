import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS_SET = [
    {
        "en": "Edit the following text for spelling and grammar mistakes:",
        "de": "Korrigieren Sie den folgenden Text auf Rechtschreib- und Grammatikfehler:",
        "it": "Modifica il seguente testo per errori di ortografia e grammatica:",
        "sv": "Redigera följande text för stavnings- och grammatikfel:",
        "cs": "Upravte v následujícím textu pravopisné a gramatické chyby:"
    },
    {
        "en": "Edit the following text for spelling and grammar mistakes, return only the corrected text:",
        "de": "Korrigieren Sie den folgenden Text auf Rechtschreib- und Grammatikfehler und geben Sie nur den korrigierten Text zurück:",
        "it": "Modifica il seguente testo per errori di ortografia e grammatica, restituisci solo il testo corretto:",
        "sv": "Redigera följande text för stavnings- och grammatikfel, returnera endast den korrigerade texten:",
        "cs": "Upravte v následujícím textu pravopisné a gramatické chyby, vraťte pouze opravený text:"
    },
    {
        "en": "Edit the following text for spelling and grammar mistakes, make minimal changes, and return only the corrected text. If the text is already correct, return it without any explanations:", # FIXED , return
        "de": "Korrigieren Sie den folgenden Text auf Rechtschreib- und Grammatikfehler, nehmen Sie nur minimale Änderungen vor und senden Sie nur den korrigierten Text zurück. Wenn der Text bereits korrekt ist, senden Sie ihn ohne Erklärungen zurück:",
        "it": "Modifica il testo seguente per errori di ortografia e grammatica, apporta modifiche minime e restituisci solo il testo corretto. Se il testo è già corretto, restituiscilo senza spiegazioni:", # FIXED, bigger diff
        "sv": "Redigera följande text för stavnings- och grammatikfel, gör minimala ändringar och returnera endast den korrigerade texten. Om texten redan är korrekt, returnera den utan några förklaringar:", # FIXED , returnera
        "cs": "Upravte v následujícím textu pravopisné a gramatické chyby, proveďte minimální změny a vraťte pouze opravený text. Pokud je text již správný, vraťte jej bez vysvětlení:"
    }
]

MODELS = {
    "karen_strict": "FPHam/Karen_TheEditor_V2_CREATIVE_Mistral_7B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama31": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "openchat36": "openchat/openchat-3.6-8b-20240522",
    "tower7B": "Unbabel/TowerInstruct-7B-v0.2",
    "tower13B": "Unbabel/TowerInstruct-13B-v0.1",
}


def prepare_prompt(prompt_idx, text, language, tokenizer):
    prompt = PROMPTS_SET[prompt_idx][language]
    chat = [
        {"role": "user", "content": f"{prompt} {text}"}  # texts is a list of sentences to correct
    ]
    filled_template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return filled_template


def prepare_prompts(prompt_idx, batch, language, tokenizer):
    return [prepare_prompt(prompt_idx, elem, language, tokenizer) for elem in batch]


def batchify(all_texts, batch_size=64, only_texts=False):
    if not only_texts:
        data = all_texts
    else:
        data = [e['content'] for e in all_texts]
    return (data[pos:pos + batch_size] for pos in range(0, len(data), batch_size))


def sort_data_lengths(data):
    data_sorted = {}
    for lang in data:
        data_sorted[lang] = sorted(data[lang], key=lambda x: len(x['content']))

    return data_sorted


def process_batch(prompt_idx, batch, language, model, tokenizer, device):
    filled_templates = prepare_prompts(prompt_idx, batch, language, tokenizer)
    tokenized_texts = tokenizer(filled_templates, return_tensors='pt', padding=True)
    prompt_length = tokenized_texts['input_ids'].shape[1]

    model_op = model.generate(
        input_ids=tokenized_texts['input_ids'].to(device),
        attention_mask=tokenized_texts['attention_mask'].to(device),
        renormalize_logits=False, do_sample=True,
        use_cache=True, max_new_tokens=256,
        repetition_penalty=1.18,
        top_k=40,
        top_p=0.1,
        # temperature=0.7
    )
    output = tokenizer.batch_decode([o[prompt_length:] for o in model_op], skip_special_tokens=True)
    return output


def process_all_prompts(data, device, model, model_id, tokenizer, iteration):
    for prompt_idx in range(len(PROMPTS_SET)):
        print("Processing prompt:", prompt_idx)

        outputs = {}

        for language in data:
            idx = 0
            batch_id = 0
            if language not in outputs:
                outputs[language] = []

            for batch in batchify(data[language], only_texts=True):
                print(f"{language}: processing batch: {batch_id}")
                batch_id += 1

                processed_texts = process_batch(prompt_idx, batch, language, model, tokenizer, device)
                for text in processed_texts:
                    e = data[language][idx]
                    e['processed'] = text
                    outputs[language].append(e)
                    idx += 1

        with open(f'{model_id}_output_prompt_{prompt_idx}_{iteration}.json', 'w') as f:
            f.write(json.dumps(outputs, ensure_ascii=False, indent=4))


def process_all_models(models, device, iterations):
    with open('data_joined.json', 'r') as f:
        data = json.load(f)
    data = sort_data_lengths(data)

    for model_id, name in models.items():
        if device == 'cuda':
            torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, trust_remote_code=True).to(device)

        print(f"Start processing model {model_id}: {name}")
        for iteration in range(iterations):
            print(f"Iteration: {iteration + 1}/{iterations}")
            process_all_prompts(data, device, model, model_id, tokenizer, iteration)
            print("-" * 50)
        print(f"Done processing model {model_id}: {name}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="*")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iterations", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    models_to_process = MODELS if not args.model else {
        model_id: name for model_id, name in MODELS.items() if model_id in args.model
    }

    process_all_models(
        models=models_to_process,
        device=args.device,
        iterations=args.iterations,
    )


if __name__ == '__main__':
    main()
