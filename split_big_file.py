import json
import lzma

MODELS = {
    'aya',
    'bloom',
    'gemma',
    'gemma2B',
    'karen',
    'karencreative',
    'llama',
    'llama31',
    'mistral',
    'openchat',
    'openchat36',
    'phi',
    'qwen',
    'smol',
    'tower7B1',
    'tower7B2',
    'tower7B3',
    'tower7B4',
    'tower7B5',
    'xglm',
    'yi',
}


def get_all_data():
    with lzma.open('merged_multillm.json.xz', "r") as f:
        return json.loads(f.read().decode("utf-8"))


def save(model_name, prompt, d):
    with open(f"model_outputs/{prompt}/{model_name}.json", "w") as f:
        json.dump(d, f, indent=4)


def process_single(model_name, prompt_id, data):
    full = {}
    for lang in data:
        full[lang] = []
        for idx, d in data[lang].items():
            label = d['marked_correct']
            for corr in d['corrections']:
                if corr['model_name'] == model_name and corr['prompt_id'] == prompt_id:
                    full[lang].append({
                        'id': idx,
                        'label': label,
                        'content': corr['content'],
                    })
    return full


if __name__ == '__main__':
    data = get_all_data()
    for model in MODELS:
        for prompt_id in (0, 1, 2):
            processed = process_single(model, prompt_id, data)
            save(model, prompt_id, processed)
            print(f"Processed {model} for prompt {prompt_id}")
