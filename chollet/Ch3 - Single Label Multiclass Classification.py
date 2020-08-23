# -*- coding: utf-8 -*-
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

import matplotlib.pyplot as plt
import numpy as np
from utils import utils

from keras.datasets import reuters
from keras import layers
from keras import models
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

# ### Network Architecture

# In a stack of Dense layers each layer can only access information present in the output of the previous layer. If one layer drops some information relevant to the classification problem, this information can never be recovered by later layers: each layer can potentially become an information bottleneck. In the previous example, we used 16-dimensional intermediate layers, but a 16-dimensional space may be too limited to learn to separate 46 different classes: such small layers may act as information bottlenecks, permanently dropping relevant information. For this reason we’ll use larger layers. Let’s go with 64 units.
#
# **OBS: DISCUSS THIS**

# Model definition
model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(46, activation="softmax"))

# **OBS:** The last layer uses a softmax activation. It means the network will output a probability distribution over the 46 different output classes—for every input sample, the network will produce a 46- dimensional output vector, where output[i] is the probability that the sample belongs to class i. The 46 scores will sum to 1.
#
# The best loss function to use in this case is categorical_crossentropy. It measures the distance between two probability distributions: here, between the probability dis- tribution output by the network and the true distribution of the labels. By minimizing the distance between these two distributions, you train the network to output some- thing as close as possible to the true labels. For more on Categorial Cross Entropy, see https://gombru.github.io/2018/05/23/cross_entropy_loss/ 

# Model compilation
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# +
# Model validation
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
# -

# Model training
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# +
loss = history.history["loss"]
val_loss = history.history["val_loss"]
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

utils.training_plots(loss, val_loss, acc, val_acc)
# -

# The NN begins to overfit after nine epochs. We will train a new network from scratch for nine epochs.

# Retraining

