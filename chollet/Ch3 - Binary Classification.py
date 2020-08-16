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

# +
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import matplotlib.pyplot as plt
import numpy as np

from keras import models
from keras import layers
# -

from keras.datasets import imdb

# Only the top 10000 most frequently occurring words in the training data are kept
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

train_data.shape
test_data.shape

# + jupyter={"outputs_hidden": true}
# Each sample is a list of words forming a review, encoded as a sequence of integers
review_nr = 1
train_data[1]
# -

# Decoding back to English
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)
decoded_review =' '.join(reverse_word_index.get(i-3, '?') for i in train_data[review_nr])

decoded_review


# ### Encoding the integer sequences into a binary matrix

# (From Chollet's book) The training lists have to be turned into tensors.
# There are two ways to do that:
# - Pad your lists so that they all have the same length, turn them into an integer tensor of shape (samples, word_indices), and then use as the first layer in your network a layer capable of handling such integer tensors (the Embedding layer, which we’ll cover in detail later in the book).
# - One-hot encode your lists to turn them into vectors of 0s and 1s. This would mean, for instance, turning the sequence [3, 5] into a 10,000-dimensional vector that would be all 0s except for indices 3 and 5, which would be 1s. Then you could use as the first layer in your network a Dense layer, capable of handling floating-point vector data.
#
# **QUESTION: In the one-hot encoder, how do you indicate if a word shows up more than once?**

# Manual implementation of second way for clarity
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

len(x_train[1])
x_train[1]

# Labels should be vectorized as well
# QUESTION Why do we transform into array if they are array already?
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# ### Network arquitecture ##
#
# The input data is vectors, and the labels are scalars (1s and 0s): this is the EASIEST setup you'll ever encounter. A type of vector that performs well on such problem i a simple stack of fully connected (Dense) layers with _relu_ activations.
# **QUESTION: Why is this network suitable?** </br>
# **QUESTION: How does the math work? Each input is a vector of size (10000) of 0 and 1, each encoded value is multiplied by the weighs?**

# (S1, S2, ..., SN)     dot

# | W11 | ... | W1-16 |
# | --- | --- | --- |
# | W21 | ... | W2-16 |
# | ... | ... | ... |
# | WN1 | ... | WN-16 |

# (S1 * W1-1 + S2 * W2-1 + ... + SN * WN-1, </br>
#  S1 * W1-2 + S2 * W2-2 + ... + SN * WN-2, </br>
#  ..., </br>
#  S1 * W1-16 + S2 * W2-16 + ... + SN * WN-16)

# S1 * W1-1 = [1, 0 ,0 , 1, ...., 0] * W1-1

model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

#

# Activation function: This is a binary classification problem and the output of the network is a probability (that's why we end the network with a single-unit layer with a sigmoid activation), it's best to use the _binary_crossentropy_ loss"

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# **Alternative 1) You want to configure parameters of the optimizer**
# model.compile(optimizer=optimizers.RMSprop(lr=0.001), </br>
#               &emsp; loss="binary_crossentropy", </br>
#               &emsp; metrics=["accuracy"])
# </br>
#
# **Alternative 2) You want to pass a custom loss function or metric function**
# model.compile(optimizer=optimizers.RMSprop(lr=0.001), </br>
#               &emsp;  loss=losses.binary_crossentropy, </br>
#               &emsp;  metrics=[metrics.binary_accuracy])

# +
# Validation
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
# -

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()

# ### Training plots

# +
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
acc = history_dict["accuracy"]
acc_val = history_dict["val_accuracy"]
epochs = range(1, len(acc) + 1)

InteractiveShell.ast_node_interactivity = "last"
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
# -

plt.plot(epochs, acc, "bo", label="Traing accuracy")
plt.plot(epochs, acc_val, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# The validation loss and accuracy peak at around the 4th epoch. We are overfitting! </br>
# Let's try using less epochs


