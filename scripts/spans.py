from scripts.ar import *


source, target = "de", "en"
dataset = "iwslt17"
ne_counter_path = "mt/res/iwslt17/de-en_ne_counter.json"
models = [
    "retnet",
    "mamba",
    "llama",
    "transformer_encdec",
    "mamba_mha",
    "mamba_mistral",
    "mamba_encdec",
]

ds = build_dataset(dataset, source, target, is_encoder_decoder=False)
models = [build_model(task="mt", name=model) for model in models]
outputs = [
    get_gen_outputs(
        f"mt/res/{dataset}/{dataset}-{source}-{target}-{model.model_name}.json"
    )
    for model in models
]


model_path = download_model("Unbabel/XCOMET-XL", "data/hf-cache")
xcomet = load_from_checkpoint(model_path)


for i, model in enumerate(models):

    get_error_spans(outputs[i], model.model_name, xcomet, batch_size=64)
