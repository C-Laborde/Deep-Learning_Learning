# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from keras.datasets import imdb
# -

# Only the top 10000 most frequently occurring words in the training date are kept
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

train_data.shape
test_data.shape

# Decoding back to English
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)
decoded_review =' '.join(reverse_word_index.get(i-3, '?') for i in train_data[0])

decoded_review


