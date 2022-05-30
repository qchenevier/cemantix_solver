# %%
import os
import warnings
from urllib3.exceptions import InsecureRequestWarning
from time import sleep

import requests
import pandas as pd

from gensim.models import KeyedVectors


def disable_SSL_verification():
    warnings.filterwarnings("ignore", category=InsecureRequestWarning)
    os.environ[
        "CURL_CA_BUNDLE"
    ] = ""  # https://stackoverflow.com/a/48391751/7539771


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


def get_new_word_key_from_vectors_weighted_average(scores, N_vocab):
    weighted_average_vector = (
        (scores)
        .head(50)
        .sample(10)
        .assign(
            score_norm=lambda df: df.score.pipe(lambda s: s - s.min()).pipe(
                lambda s: s / s.sum()
            )
        )
        .pipe(lambda df: df.vector * df.score_norm)
        .sum()
    )
    return [
        w
        for w, _ in model.most_similar(
            weighted_average_vector, restrict_vocab=N_vocab
        )
    ]


# %%
disable_SSL_verification()

# %%
model = KeyedVectors.load_word2vec_format(
    "frwiki-20181020.treetag.2__2019-01-24_10.41__.s500_w5_skip.word2vec.bin",
    binary=True,
    unicode_errors="ignore",
)
vocab = pd.DataFrame({"key": model.index_to_key}).assign(
    word=lambda df: df.key.str.split("_").str[0].str.lower()
)

# %%
N_vocab = 30000
vocab_selection = vocab.head(N_vocab)
cache = dict()

# %%
scores = pd.DataFrame(columns=["key", "score", "vector"])
while len(scores) < 20:
    scores = add_random_word_to_scores(scores, vocab_selection, model, cache)

# %%
for i in range(500):
    word_keys = get_new_word_key_from_vectors_weighted_average(scores, N_vocab)
    word_score = None
    for word_key in word_keys:
        if word_key not in scores.key.values:
            word_score = get_score_from_word_key(word_key, cache)
            if word_score is not None:
                print(f"{i} - adding new word key: {word_key}: {word_score}")
                scores = add_word_to_scores(word_key, word_score, scores, model)
                break
    if word_score is None:
        scores = add_random_word_to_scores(
            scores, vocab_selection, model, cache
        )
    if word_score == 1:
        print(f"finished in {len(cache)} requests")
        break

    if len(scores) > 200:
        print("pruning")
        scores = scores.head(100).sample(40)

scores

# %%
