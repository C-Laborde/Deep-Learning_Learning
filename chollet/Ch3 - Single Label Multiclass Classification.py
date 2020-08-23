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

# Reuters dataset, a set of short newswires and their topics, published by Reuters in 1986.
# There are 46 different topics; some topics are more represented than others, but each topic has at least 10 examples in the training set.

# Imports

# +
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np

from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
# -

# Loading dataset

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

len(train_data)
len(test_data)

# %pprint
train_data[3]

# Decoding newswires back to text
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[3]])

decoded_newswire


# +
# Obs: There's no reverse_word_index equivalent to convert the labels into strings
# -

# Data vectorization. See Ch3 - Binary Classification for reference
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# We need to vectorize also the laels. There are two options:
# 1) Cast the label list as an integer tensor </br>
# 2) Use one-hot incoding: Embedding each label as an all-zero vector with a 1 in the place of the label index

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


