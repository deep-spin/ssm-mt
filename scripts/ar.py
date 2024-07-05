from mt.ds import build_dataset
from models import build_model
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import to_hex

import nltk
import math
import json
from nltk import ne_chunk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from comet import download_model, load_from_checkpoint


import argparse

from tqdm.auto import tqdm

nltk.download("punkt")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("averaged_perceptron_tagger")

parser = argparse.ArgumentParser(description="AR script.")

parser.add_argument(
    "--models",
    type=str,
    nargs="+",
    help="List of model names to generate plots for.",
)

parser.add_argument(
    "--dataset",
    type=str,
    default="iwslt17",
    help="Dataset to use for training.",
)

parser.add_argument(
    "--language_pair",
    nargs=2,
    type=str,
    help="Language pair to use for training.",
)


parser.add_argument(
    "--ne_counter_path",
    type=str,
    help="Path to the NE counter.",
)


from nltk import Tree


def extract_named_entities(tree):
    named_entities = []

    for subtree in tree:
        if type(subtree) == Tree and subtree.label() in [
            "PERSON",
            "ORGANIZATION",
            "GPE",
        ]:
            entity = " ".join(word for word, tag in subtree.leaves())
            named_entities.append(entity)
        elif type(subtree) == tuple and subtree[1] == "NNP":
            named_entities.append(subtree[0])

    return named_entities


pipe = lambda x: extract_named_entities(ne_chunk(pos_tag(word_tokenize(x))))


def get_ne_counter(path, ds):

    if path is not None:
        with open(path, "r") as f:
            return json.load(f)

    ne_counter = {}

    for i, sample in enumerate(tqdm(ds.dataset["train"])):
        source_text = (
            sample["translation"]["de"] if "translation" in sample else sample["de"]
        )
        target_text = (
            sample["translation"]["en"] if "translation" in sample else sample["en"]
        )

        target_ne = pipe(target_text)
        # sample freq
        used = set(target_ne)
        for ne in used:
            if ne not in source_text:
                continue
            if ne not in ne_counter:
                ne_counter[ne] = 0
            ne_counter[ne] += 1

    with open(
        f"mt/res/{ds.name}/{ds.name}-{ds.source_lang}-{ds.target_lang}-ne_counter.json",
        "w",
    ) as f:
        json.dump(ne_counter, f)

    return ne_counter


def get_test_ne(ds, source, target):
    test_named_entities = []
    for i, sample in enumerate(tqdm(ds.dataset["test"])):
        source_text = sample["translation"][source] if "translation" in sample else sample[source]
        target_text = sample["translation"][target] if "translation" in sample else sample[target]

        target_ne = pipe(target_text)
        target_ne = [ne for ne in target_ne if ne in source_text]
        # sample freq
        used = set(target_ne)
        test_named_entities.append((i, tuple(ne for ne in used)))

    return test_named_entities


def get_gen_outputs(path):

    outputs = json.load(open(path, "r"))
    save = []
    for entry in outputs:
        x = [sample for sample in zip(*entry)]
        save += x

    return [(i, sample) for i, sample in enumerate(save)]


def get_ne_freq_accuracy(ne_counter, test_named_entities, gen_outputs):

    acc = {}

    for entry in test_named_entities:

        sample_idx = entry[0]

        # src = ds.dataset["test"][sample_idx]["translation"][source]
        # ref = ds.dataset["test"][sample_idx]["translation"][target]

        # retnet_bleu = gen_outputs[sample_idx][1][3]
        # comet = gen_outputs[sample_idx][1][4]
        pred = gen_outputs[sample_idx][1][1]

        if len(entry[1]) > 0:

            for ne in entry[1]:
                # Preparing updates based on current predictions
                current_acc = ne in pred
                if ne not in acc:
                    acc[ne] = (1, current_acc)  # Initial count and accuracy tuple
                else:
                    prev_count, prev_acc = acc[ne]
                    # Updating the count
                    updated_count = prev_count + 1
                    # Updating accuracy values: Calculating new mean accuracy
                    updated_acc = (prev_acc * prev_count + current_acc) / updated_count
                    acc[ne] = (updated_count, updated_acc)

    res = []
    for k, v in acc.items():

        count = ne_counter.get(k, 0)
        res.append((v[1], count))

    return res


def get_ne_len_accuracy(test_named_entities, gen_outputs, ds, source, target):

    acc = {}
    tokenizer = ds.get_tokenizer()

    for entry in test_named_entities:
        sample_idx = entry[0]
        src = ds.dataset["test"][sample_idx]["translation"][source]
        ref = ds.dataset["test"][sample_idx]["translation"][target]
        # retnet_bleu = gen_outputs[sample_idx][1][3]
        # comet = gen_outputs[sample_idx][1][4]
        pred = gen_outputs[sample_idx][1][1]

        if len(entry[1]) > 0:

            n_tokens = len(tokenizer(src)["input_ids"])

            for ne in entry[1]:
                # Preparing updates based on current predictions
                current_acc = ne in pred

                if n_tokens not in acc:
                    acc[n_tokens] = (1, current_acc)  # Initial count and accuracy tuple
                else:
                    prev_count, prev_acc = acc[n_tokens]
                    # Updating the count
                    updated_count = prev_count + 1

                    # Updating accuracy values: Calculating new mean accuracy
                    updated_acc = (prev_acc * prev_count + current_acc) / updated_count

                    acc[n_tokens] = (updated_count, updated_acc)

    res = []
    for k, v in acc.items():
        res.append((v[1], k))

    return res


def get_bleu_scores_by_length(gen_outputs, ds, source, target):
    # Dictionary to store the accumulated BLEU scores and counts for each token length
    bleu_scores = {}
    tokenizer = ds.get_tokenizer()

    for sample_idx in range(len(ds.dataset["test"])):

        src = (
            ds.dataset["test"][sample_idx]["translation"][source]
            if "translation" in ds.dataset["test"][sample_idx]
            else ds.dataset["test"][sample_idx][source]
        )
        # ref = ds.dataset["test"][sample_idx]["translation"][target]
        bleu = gen_outputs[sample_idx][1][
            3
        ]  # Assuming this index stores the pre-calculated BLEU

        # Tokenize the source sentence to get the number of tokens
        n_tokens = len(tokenizer(src)["input_ids"])

        # Accumulate the BLEU scores for each token length
        if n_tokens not in bleu_scores:
            bleu_scores[n_tokens] = [
                bleu
            ]  # Start a list of BLEU scores for this token length
        else:
            bleu_scores[n_tokens].append(bleu)

    # Calculate average BLEU scores per token count
    average_bleu_by_length = []
    for n_tokens, scores in bleu_scores.items():
        avg_bleu = sum(scores) / len(scores)
        average_bleu_by_length.append((avg_bleu, n_tokens))
    return average_bleu_by_length


def get_comet_scores_by_length(gen_outputs, ds, source, target):
    # Dictionary to store the accumulated COMET scores and counts for each token length
    comet_scores = {}
    tokenizer = ds.get_tokenizer()

    for sample_idx in range(len(ds.dataset["test"])):
        src = (
            ds.dataset["test"][sample_idx]["translation"][source]
            if "translation" in ds.dataset["test"][sample_idx]
            else ds.dataset["test"][sample_idx][source]
        )
        # ref = ds.dataset["test"][sample_idx]["translation"][target]
        comet = gen_outputs[sample_idx][1][4]

        # Tokenize the source sentence to get the number of tokens
        n_tokens = len(tokenizer(src)["input_ids"])

        # Accumulate the BLEU scores for each token length
        if n_tokens not in comet_scores:
            comet_scores[n_tokens] = [
                comet
            ]  # Start a list of BLEU scores for this token length
        else:
            comet_scores[n_tokens].append(comet)

    # Calculate average BLEU scores per token count
    average_comet_by_length = []
    for n_tokens, scores in comet_scores.items():
        avg_comet = sum(scores) / len(scores)
        average_comet_by_length.append((avg_comet, n_tokens))
    return average_comet_by_length


def get_error_spans(data, model_name, xcomet, batch_size=64):

    def map_format(sample):
        return {
            "src": sample[0],  # Assuming sample[0] is source text
            "mt": sample[1],  # Assuming sample[1] is machine translation output
            "ref": sample[2],  # Assuming sample[2] is reference translation
        }

    ds_size = len(data)

    mapped_data = [map_format(sample[1]) for sample in data]
    spans = xcomet.predict(mapped_data, batch_size=batch_size, gpus=1)

    # for start in range(0, ds_size, batch_size):

    #     if start in (1024, 2048, 4096, 6144):
    #         print(f"model_name: {model_name}, idx: {start}")

    #     end = min(start + batch_size, ds_size)
    #     batch = data[start:end]

    #     mapped_data = [map_format(sample[1]) for sample in batch]

    #     batch_output = xcomet.predict(mapped_data, batch_size=len(mapped_data), gpus=1, )
    #     spans.append(batch_output)

    json.dump(spans, open(f"mt/res/iwslt17/xcomet_{model_name}.json", "w"))

    print(f"finished model_name: {model_name}")
    return spans


def plot_scatterplots(dataset, ne_accs, model_names, source, target):
    n = len(ne_accs)
    colors = [
        "red",
        "green",
        "blue",
        "cyan",
        "magenta",
        "orange",
        "purple",
        "brown",
        "grey",
    ]
    colors = colors[:n]

    # Calculate the number of columns and rows needed
    ncols = 3  # Max number of columns
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)  # Flatten the axes array for easy indexing

    # Looping through each set of data to create subplots
    for i, (ax, data, title) in enumerate(zip(axes, ne_accs, model_names)):
        # Unpacking accuracy and NE count from the data
        y, x = zip(*data)  # x is accuracy, y is NE count in the training set
        ax.scatter(x, y, alpha=0.5, color=colors[i])  # Plotting the scatter plot

        # Setting titles, labels, and grid for each subplot
        ax.set_title(title)
        ax.set_xlabel("NE Count in Training Set")
        ax.grid(True)

    # This ensures the y-axis label is only set once for all subplots
    axes[0].set_ylabel("Accuracy")

    # If any extra axes, hide them
    for j in range(n, nrows * ncols):
        axes[j].axis("off")

    plt.tight_layout()  # Adjusts subplot params to give some padding
    plt.savefig(f"plots/{dataset}-{source}-{target}-scatterplot.png")
    plt.show()


def plot_ne_freqs(dataset, ne_accs, model_names, source, target):

    n = 3  # There are three categories: Unseen, Regular, Frequent

    cmap = plt.cm.get_cmap("winter", n)  # 'n' is the number of unique colors needed
    colors = [cmap(i) for i in range(n)]

    max_count = max(max(count for _, count in model_data) for model_data in ne_accs)
    bins = [0, 1, 16, 110000]
    bucket_labels = ["Unseen", "Regular", "Frequent"]  # Updated bucket labels

    # Assuming 2 rows, dynamically determine the number of columns based on the number of models
    cols = 4
    rows = (
        len(model_names) + cols - 1
    ) // cols  # Adjust rows based on the number of models

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 2.5, rows * 3), sharey="row", sharex="col"
    )
    axes = np.array(axes).reshape(-1)  # Flatten the axes array for easy indexing

    i = 0
    for ax, model_data, model_name in zip(axes, ne_accs, model_names):
        binned_accuracies = [[] for _ in range(len(bins) - 1)]
        for accuracy, count in model_data:
            bucket_index = np.digitize(count, bins) - 1
            binned_accuracies[bucket_index].append(accuracy)

        mean_accuracies = [
            np.mean(bucket) if bucket else 0 for bucket in binned_accuracies
        ]  # Handle empty buckets
        ax.bar(bucket_labels, mean_accuracies, width=0.8, alpha=0.7, color=colors)
        ax.set_title(model_name)
        if i % cols == 0:  # Set y-label only on the first column
            ax.set_ylabel("Average Accuracy")
        if i // cols == rows - 1:  # Set x-label only on the last row
            ax.set_xlabel("NE Frequency")
        ax.tick_params(axis="x", rotation=60)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_ylim(0.5, 1)  # Set y-axis limits

        i += 1

    # Hide unused subplots if any
    for j in range(i, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"plots/{dataset}-{source}-{target}-ne_freqs.pdf")
    plt.show()


def plot_ne_freqs_shifted_bars(dataset, ne_accs, model_names, source, target):
    n = 3  # There are three categories: Unseen, Regular, Frequent
    colors = [
        to_hex(plt.cm.winter(0.2)),
        to_hex(plt.cm.winter(0.5)),
        to_hex(plt.cm.winter(0.8)),
    ]

    bins = [0, 1, 16, 284960]
    bucket_labels = ["Unseen", "Regular", "Frequent"]  # Updated bucket labels

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_ylim(0.5, 1)  # Set y-axis limits

    width = 0.2  # Width of each bar
    bar_positions = np.arange(len(model_names))  # X positions for each model

    for i, (model_data, model_name) in enumerate(zip(ne_accs, model_names)):
        binned_accuracies = [[] for _ in range(len(bins) - 1)]
        for accuracy, count in model_data:
            bucket_index = np.digitize(count, bins) - 1
            binned_accuracies[bucket_index].append(accuracy)

        mean_accuracies = [
            np.mean(bucket) if bucket else 0 for bucket in binned_accuracies
        ]  # Handle empty buckets

        # Create bar positions for the current model
        for j in range(n):
            position = bar_positions[i] + (j - 1) * width
            ax.bar(
                position,
                mean_accuracies[j],
                width=width,
                alpha=0.7,
                color=colors[j],
                label=bucket_labels[j] if i == 0 else "",  # Add label only once
            )

    print([len(bucket) for bucket in binned_accuracies])

    # Set the x-axis labels to be in the middle of the grouped bars
    ax.set_title(f"Named Entity Recall Accuracy - {dataset} {source}-{target}")

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(model_names, rotation=0)
    ax.set_ylabel("Average Accuracy")
    ax.set_xlabel("Model")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(title="NE Frequency", loc="lower right")
 

    plt.tight_layout()
    plt.savefig(f"plots/{dataset}-{source}-{target}-ne_freqs.pdf")
    plt.show()


def plot_ne_freqs_line(dataset, ne_accs, model_names, source, target):
    n = len(ne_accs)

    cmap = plt.cm.get_cmap("Accent", n)  # 'n' is the number of unique colors needed
    colors = [cmap(i) for i in range(n)]
    bins = [0, 1, 4, 8, 16, 32, 64, 1028]
    bucket_labels = [f"{int(bins[i-1])}-{int(bins[i])}" for i in range(1, len(bins))]

    fig, ax = plt.subplots(figsize=(len(bins) * 2, 6))  # Single subplot

    for model_data, model_name, color in zip(ne_accs, model_names, colors):
        binned_accuracies = [[] for _ in range(1, len(bins))]
        for accuracy, count in model_data:
            bucket_index = np.digitize(count, bins) - 1
            binned_accuracies[bucket_index].append(accuracy)

        mean_accuracies = [np.mean(bucket) for bucket in binned_accuracies]
        ax.plot(
            bucket_labels, mean_accuracies, marker="o", label=model_name, color=color
        )

    ax.set_title(f"{dataset} {source}-{target} NE Recall Accuracy by Frequency")
    ax.set_xlabel("# Tokens")
    ax.set_ylabel("Average Accuracy")
    ax.legend(loc="upper left")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"plots/{dataset}-{source}-{target}-bleu_lens_kine.png")
    plt.show()


def plot_ne_lens(dataset, ne_accs, model_names, source, target):
    n = len(ne_accs)

    cmap = plt.cm.get_cmap("winter", 5)  # 'n' is the number of unique colors needed
    colors = [cmap(3) for _ in range(n)]

    max_count = max(max(count for _, count in model_data) for model_data in ne_accs)
    bins = [0, 16, 32, 64, 80, 96, 128]
    bucket_labels = [f"{int(bins[i-1])}-{int(bins[i])}" for i in range(1, len(bins))]

    # Assuming 2 rows, dynamically determine the number of columns based on the number of models
    cols = 4
    rows = 2

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 2.5, 6), sharey="row", sharex="col"
    )
    axes = np.array(axes).reshape(-1)  # Flatten the axes array for easy indexing

    i = 0
    for ax, model_data, model_name in zip(axes, ne_accs, model_names):
        binned_accuracies = [[] for _ in range(1, len(bins))]
        for accuracy, count in model_data:
            bucket_index = np.digitize(count, bins) - 1
            binned_accuracies[bucket_index].append(accuracy)

        mean_accuracies = [np.mean(bucket) for bucket in binned_accuracies]
        ax.bar(bucket_labels, mean_accuracies, width=0.7, alpha=0.7, color=colors[i])
        ax.set_title(model_name)
        if i % cols == 0:  # Set y-label only on the first column
            ax.set_ylabel("Average Accuracy")
        if i >= len(model_names) - cols:  # Set x-label only on the last row
            ax.set_xlabel("# Tokens")
        ax.tick_params(axis="x", rotation=60)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        i += 1

    print([len(bucket) for bucket in binned_accuracies])
    # Hide unused subplots if any
    for j in range(len(model_names), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"plots/{dataset}-{source}-{target}-ne_lens.pdf")
    plt.show()


def plot_bleu_lens(dataset, bleu_scores, model_names, source, target):
    n = len(bleu_scores)

    cmap = plt.cm.get_cmap("winter", n)  # 'n' is the number of unique colors needed
    colors = [cmap(i) for i in range(n)]

    max_count = max(max(count for _, count in model_data) for model_data in bleu_scores)
    bins = [0, 16, 32, 48, 64, 80, 96, 128]
    bucket_labels = [f"{int(bins[i-1])}-{int(bins[i])}" for i in range(1, len(bins))]

    # Assuming 2 rows, dynamically determine the number of columns based on the number of models
    cols = min(len(model_names), 4)
    rows = (len(model_names) + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(len(bins) * 2, 6), sharey="row", sharex="col"
    )
    axes = np.array(axes).reshape(-1)  # Flatten the axes array for easy indexing

    i = 0
    for ax, model_data, model_name in zip(axes, bleu_scores, model_names):
        binned_accuracies = [[] for _ in range(1, len(bins))]
        for accuracy, count in model_data:
            bucket_index = np.digitize(count, bins) - 1
            binned_accuracies[bucket_index].append(accuracy)

        mean_accuracies = [np.mean(bucket) for bucket in binned_accuracies]
        ax.bar(bucket_labels, mean_accuracies, width=0.7, alpha=0.7, color=colors[i])
        ax.set_title(model_name)
        if i % cols == 0:  # Set y-label only on the first column
            ax.set_ylabel("BLEU Score")
        if i >= len(model_names) - cols:  # Set x-label only on the last row
            ax.set_xlabel("# Tokens")
        ax.tick_params(axis="x", rotation=75)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        i += 1

    print([len(bucket) for bucket in binned_accuracies])
    # Hide unused subplots if any
    for j in range(len(model_names), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"plots/{dataset}-{source}-{target}-bleu_lens.png")
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def plot_bleu_lens_line(dataset, bleu_scores, model_names, source, target):
    n = len(bleu_scores)

    cmap = plt.cm.get_cmap("Accent", n)  # 'n' is the number of unique colors needed
    colors = [cmap(i) for i in range(n)]
    bins = [0, 16, 32, 48, 64, 80, 96, 128]
    bucket_labels = [f"{int(bins[i-1])}-{int(bins[i])}" for i in range(1, len(bins))]

    fig, ax = plt.subplots(figsize=(len(bins) * 2, 6))  # Single subplot

    for model_data, model_name, color in zip(bleu_scores, model_names, colors):
        binned_accuracies = [[] for _ in range(1, len(bins))]
        for accuracy, count in model_data:
            bucket_index = np.digitize(count, bins) - 1
            binned_accuracies[bucket_index].append(accuracy)

        mean_accuracies = [np.mean(bucket) for bucket in binned_accuracies]
        ax.plot(
            bucket_labels, mean_accuracies, marker="o", label=model_name, color=color
        )

    ax.set_title(f"{dataset} {source}-{target} Translation Accuracy by Token Length")
    ax.set_xlabel("# Tokens")
    ax.set_ylabel("BLEU Score")
    ax.legend(loc="upper left")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"plots/{dataset}-{source}-{target}-bleu_lens_kine.png")
    plt.show()


def plot_comet_lens(dataset, comet_scores, model_names, source, target):

    n = len(comet_scores)

    cmap = plt.cm.get_cmap("Blues", n)  # 'n' is the number of unique colors needed
    colors = [cmap(i) for i in range(n)]


    # colors = [to_hex(plt.cm.Blues(0.5))]
    max_count = max(
        max(count for _, count in model_data) for model_data in comet_scores
    )
    bins = [0, 16, 32, 64, 128, 256]
    bucket_labels = [f"{int(bins[i-1])}-{int(bins[i])}" for i in range(1, len(bins))]

    # Assuming 2 rows, dynamically determine the number of columns based on the number of models
    cols = 4
    rows = 2

    fig, axes = plt.subplots(
        rows, cols, figsize=(len(bins) * 2, 6), sharey="row", sharex="col"
    )
    axes = np.array(axes).reshape(-1)  # Flatten the axes array for easy indexing

    i = 0
    for ax, model_data, model_name in zip(axes, comet_scores, model_names):
        binned_accuracies = [[] for _ in range(1, len(bins))]
        for accuracy, count in model_data:
            bucket_index = np.digitize(count, bins) - 1
            binned_accuracies[bucket_index].append(accuracy)

        mean_accuracies = [np.mean(bucket) for bucket in binned_accuracies]
        ax.plot(
            bucket_labels,
            mean_accuracies,
            marker="o",
            color=colors[5],
            label=model_name,
        )
        ax.set_title(model_name)
        if i % cols == 0:  # Set y-label only on the first column
            ax.set_ylabel("COMET Score")
        if i >= len(model_names) - cols:  # Set x-label only on the last row
            ax.set_xlabel("# Tokens")
        ax.tick_params(axis="x", rotation=75)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        i += 1

    print([len(bucket) for bucket in binned_accuracies])
    # Hide unused subplots if any
    for j in range(len(model_names), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"plots/{dataset}-{source}-{target}-comet_lens.pdf")
    plt.show()


if __name__ == "__main__":

    args, _ = parser.parse_known_args()

    source, target = args.language_pair
    dataset = args.dataset

    ds = build_dataset(dataset, source, target, is_encoder_decoder=False)

    models = args.models
    models = [build_model(task="mt", name=model) for model in models]
    outputs = [
        get_gen_outputs(
            f"mt/res/{dataset}/{dataset}-{source}-{target}-{model.model_name}.json"
        )
        for model in models
    ]

    ne_counter = get_ne_counter(args.ne_counter_path, ds)
    test_named_entities = get_test_ne(ds, source, target)

    ne_accs = [
        get_ne_freq_accuracy(ne_counter, test_named_entities, outputs[i])
        for i in range(len(models))
    ]

    plot_scatterplots(
        dataset, ne_accs, [model.model_name for model in models], source, target
    )
    plot_ne_freqs(
        dataset, ne_accs, [model.model_name for model in models], source, target
    )

    ne_lens = [
        get_ne_len_accuracy(test_named_entities, outputs[i], ds, source, target)
        for i in range(len(models))
    ]

    plot_ne_lens(
        dataset, ne_lens, [model.model_name for model in models], source, target
    )
