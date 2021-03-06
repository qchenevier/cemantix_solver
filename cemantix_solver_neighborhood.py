# %%
import os
import urllib.request
from time import sleep

import ipywidgets as widgets
import pandas as pd
import requests
from gensim.models import KeyedVectors
from IPython.display import display
from tqdm import tqdm
import sys
import numpy as np


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, filename):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=filename, file=sys.stdout
    ) as t:
        urllib.request.urlretrieve(
            url, filename=filename, reporthook=t.update_to
        )


def download_if_not_present(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        download_file(url, filename)
    return filename


def get_score(word):
    url = "https://cemantix.herokuapp.com/score"
    response = requests.post(url, {"word": word})
    sleep(1)  # API may refuse too fast requests
    return response.json().get("score", None)


def get_score_from_word_key(word_key, cache):
    if word_key in cache:
        return cache[word_key]
    word = word_key.split("_")[0].lower()
    score = get_score(word)
    cache[word_key] = score
    return score


def add_word_to_scores(word_key, word_score, scores, model):
    if word_key not in scores.key.values:
        scores = pd.concat(
            [
                scores,
                pd.DataFrame.from_records(
                    [
                        {
                            "key": word_key,
                            "score": word_score,
                            "vector": model.get_vector(word_key),
                        }
                    ]
                ),
            ],
            axis=0,
        )
    return scores.sort_values(by="score", ascending=False).reset_index(
        drop=True
    )


def add_random_word_to_scores(scores, vocab, model, cache):
    sample_score = None
    while sample_score is None:
        sample = vocab.sample(1).iloc[0]
        sample_score = get_score_from_word_key(sample.key, cache)
    return add_word_to_scores(sample.key, sample_score, scores, model)


def search(button):
    N_vocab = int(widget_N_vocab.value)
    N_neighborhood = widget_N_neighborhood.value
    vocab_selection = vocab.head(N_vocab)
    with OUTPUT:
        print(f"Search start.")
    cache = dict()
    scores = pd.DataFrame(columns=["key", "score", "vector"])
    scores = add_random_word_to_scores(scores, vocab_selection, model, cache)

    score = 0
    while score < 1:
        key, score, vector = scores.iloc[0].tolist()
        neighborhood = [
            w
            for w, _ in model.most_similar(
                key, restrict_vocab=N_vocab, topn=N_neighborhood
            )
        ]
        for new_key in neighborhood:
            new_score = get_score_from_word_key(new_key, cache)
            if new_score is not None and new_score > score:
                with OUTPUT:
                    print(f"query n??{len(cache)} - {new_key}: new best score={new_score}")
                scores = add_word_to_scores(new_key, new_score, scores, model)
                break
        if new_score is None or new_score <= score:
            with OUTPUT:
                print(
                    "Neighborhood explored without finding new best option. "
                    "Please increase Neighborhood size."
                )
        if new_score == 1:
            with OUTPUT:
                print(f"Finished in {len(cache)} requests. The solution is '{new_key}'.")
            return new_key


# %%
url = (
    "https://embeddings.net/embeddings/"
    "frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin"
)
embeddings_filename = download_if_not_present(url)

# %%
model = KeyedVectors.load_word2vec_format(
    embeddings_filename,
    binary=True,
    unicode_errors="ignore",
)
vocab = pd.DataFrame({"key": model.index_to_key}).assign(
    word=lambda df: df.key.str.split("_").str[0].str.lower()
)
widget_N_vocab = widgets.FloatLogSlider(value=3000, min=0, max=np.log10(vocab.shape[0]))
widget_N_neighborhood = widgets.IntSlider(value=300, min=0, max=1000)

button = widgets.Button(description="Search")
button.on_click(search)

OUTPUT = widgets.Output()

display(widgets.HBox([widgets.Label("Vocabulary size (log scale):"), widget_N_vocab]))
display(widgets.HBox([widgets.Label("Neighborhood search size:"), widget_N_neighborhood]))
display(button, OUTPUT)
