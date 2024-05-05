import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense, Conv2D
import mpi4py
import os


batch_size = 128
img_height = 224
img_width = 224
scratch = os.environ['SCRATCH']
train_dir = os.path.join(scratch,'imagenette/imagenette2/train/')
val_dir = os.path.join(scratch,'imagenette/imagenette2/val/')

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
)


# normalization_layer = tf.keras.layers.Rescaling(1./255)
# normalized_train_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
# normalized_val_ds = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# image_batch, labels_batch = next(iter(normalized_train_ds))
# image_batch, labels_batch = next(iter(normalized_val_ds))

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)


