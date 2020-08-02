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
# -

from keras.datasets import mnist
from keras import layers
from keras import models
from keras.utils import to_categorical

# ### Data load

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

"Train images dims and shape:"
train_images.ndim
train_images.shape
"Train labels shape:"
train_labels.shape
"Test images shape:"
test_images.shape
"Test labels.shape:"
test_labels.shape

train_images.min()
train_images.max()
train_images.ndim

# +
digit = train_images[4]

plt.imshow(digit, cmap=plt.cm.binary)
# -

# ### Network

# +
# Arquitecture
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
network.add(layers.Dense(10, activation='softmax'))

# Compilator
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Data preparation: Reshape and normalization
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / train_images.max()

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / test_images.max()

# Labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# -

# Training
network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
"test_acc:"
test_acc


