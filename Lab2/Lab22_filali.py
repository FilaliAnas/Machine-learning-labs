# Lab22: Classification fruites & vegetebales
# Realise par : Anas FILALI - EMSI 2023/2024
# Reférence : https://colab.research.google.com/drive/1rFEArW44Tt5N_Wy2GdPKzzRF5NUaYv5P#scrollTo=AZXzzvArbrR7

import tensorflow as tf
import matplotlib.pyplot as plt
# Step 1: Dataset
img_height, img_width = 32, 32
batch_size = 20 # batch == lot

train_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/fruits/train",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/fruits/validation",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/fruits/test",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
# Data visualizations
class_names = ["apple", "banana", "orange"]
plt.figure(figsize=(10,10))
for images, labels in test_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
# Step 2: Model
model = tf.keras.Sequential(
    [
     tf.keras.layers.Rescaling(1./255),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128, activation="relu"),
     tf.keras.layers.Dense(3)
    ]
)
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=['accuracy']
)
# Step 3: Train (fit)
model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 20
)
# Step 4: Test
print(model.evaluate(test_ds))
# Test visualization
import numpy

plt.figure(figsize=(10,10))
for images, labels in test_ds.take(1):
  classifications = model(images)
  # print(classifications)

  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    index = numpy.argmax(classifications[i])
    plt.title("Pred: " + class_names[index] + " | Real: " + class_names[labels[i]])
plt.show()