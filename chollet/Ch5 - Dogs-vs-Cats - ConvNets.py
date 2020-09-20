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

import os
import shutil

# +
train_data_path = "data/dogs-vs-cats/train"

# We will work with a smaller data set to challenge the model
subset_path = "data/dogs-vs-cats/subdata"

train_cats_dir = os.path.join(subset_path, "train", "cats")
train_dogs_dir = os.path.join(subset_path, "train", "dogs")

val_cats_dir = os.path.join(subset_path, "validation", "cats")
val_dogs_dir = os.path.join(subset_path, "validation", "dogs")

test_cats_dir = os.path.join(subset_path, "test", "cats")
test_dogs_dir = os.path.join(subset_path, "test", "dogs")
# -

# Cats

# +
fnames = [f'cat.{i}.jpg' for i in range(1000)]

for fname in fnames:
    src = os.path.join(train_data_path, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# +
fnames = [f'cat.{i}.jpg' for i in range(1000, 1500)]

for fname in fnames:
    src = os.path.join(train_data_path, fname)
    dst = os.path.join(val_cats_dir, fname)
    shutil.copyfile(src, dst)

# +
fnames = [f'cat.{i}.jpg' for i in range(1500, 2000)]

for fname in fnames:
    src = os.path.join(train_data_path, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
# -

# Dogs

# +
fnames = [f'dog.{i}.jpg' for i in range(1000)]
for fname in fnames:
    src = os.path.join(train_data_path, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = [f'dog.{i}.jpg' for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(train_data_path, fname)
    dst = os.path.join(val_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
    
fnames = [f'dog.{i}.jpg' for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(train_data_path, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

# +
# Sanity check

print("Total training cat images:", len(os.listdir(train_cats_dir)))
print("Total training dog images:", len(os.listdir(train_dogs_dir)))
print("Total validation cat images:", len(os.listdir(val_cats_dir)))
print("Total validation dog images:", len(os.listdir(val_dogs_dir)))
print("Total test cat images:", len(os.listdir(test_cats_dir)))
print("Total test dog images:", len(os.listdir(test_dogs_dir)))
# -


