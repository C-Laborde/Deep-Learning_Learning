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

# ## Boston Housing Price Prediction

# +
import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import boston_housing
# -

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 404 samples, 13 features
train_data.shape

test_data.shape

# Targets are the median values of owner-ocupied homes, in thousands of dollars
train_targets.shape

# ### Preparing the data

# Feature-wise normalization: for each feature in the input data, subtract the mean of the feature and divide by the standard deviation, so that the feature is centered around 0 and has a unit standard deviation. 

# +
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

# OBS: the quantities used for normalizing the test data are computed using the training
# data. You should never use in your workflow any quantity computed on the test data, even
# for something as simple as data normalization
test_data -= mean
test_data /= std
# -

# ### Network architecture

# +
from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation="relu"))
    # No activation in the output layer, linear layer, so we don't constrain the output values
    model.add(layers.Dense(1))
    model.compile(optimizer="rmsprop",
                  loss="mse",
                  metrics=['mae'])
    return model


# -

# ### Model validation using k-fold

# To evaluate the network while we keep adjusting its parameters (such as the number of epochs used for training), we could split the data into a training set and a valida- tion set. But because we have so few data points, the validation set would end up being very small (for instance, about 100 examples). As a consequence, the validation scores might change a lot depending on which data points we chose to use for validation and which we chose for training: the validation scores might have a high variance with regard to the validation split. This would prevent us from reliably evaluating the model.
#

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print("Processing fold #", i)
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate(
        [train_data[: i * num_val_samples],
         train_data[(i+1) * num_val_samples :]],
        axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[: i * num_val_samples],
         train_targets[(i+1) * num_val_samples :]]
    )
    
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs = num_epochs, batch_size=1, verbose=0)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

all_scores

np.mean(all_scores)

# We will try with longer epochs

# +
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print("Processing fold #", i)
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate(
        [train_data[: i * num_val_samples],
         train_data[(i+1) * num_val_samples :]],
        axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[: i * num_val_samples],
         train_targets[(i+1) * num_val_samples :]]
    )
    
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        epochs = num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['mae']
    all_mae_histories.append(mae_history)
# -

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")


# +
# For better plotting we apply an exponential moving average to smooth the curve

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smooted_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# -

smoothed_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
