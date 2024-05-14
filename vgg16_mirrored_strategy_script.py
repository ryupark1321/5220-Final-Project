import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense, Conv2D, Rescaling
import os
import time


# Create model
def vgg16():
    model = Sequential([
        Rescaling(1./255),
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3), strides=1),
        Conv2D(64, (3, 3), activation='relu', padding='same', strides=1),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same', strides=1),
        Conv2D(128, (3, 3), activation='relu', padding='same', strides=1),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Conv2D(256, (3, 3), activation='relu', padding='same', strides=1),
        Conv2D(256, (3, 3), activation='relu', padding='same', strides=1),
        Conv2D(256, (3, 3), activation='relu', padding='same', strides=1),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Conv2D(512, (3, 3), activation='relu', padding='same', strides=1),
        Conv2D(512, (3, 3), activation='relu', padding='same', strides=1),
        Conv2D(512, (3, 3), activation='relu', padding='same', strides=1),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Conv2D(512, (3, 3), activation='relu', padding='same', strides=1),
        Conv2D(512, (3, 3), activation='relu', padding='same', strides=1),
        Conv2D(512, (3, 3), activation='relu', padding='same', strides=1),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(1000, activation='softmax')
    ])
    return model

# Create a checkpoint directory to store the checkpoints.
# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


def train_step(inputs):
  images, labels = inputs

  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = compute_loss(labels, predictions, model.losses)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_accuracy.update_state(labels, predictions)
  return loss

def test_step(inputs):
  images, labels = inputs

  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss.update_state(t_loss)
  test_accuracy.update_state(labels, predictions)


# `run` replicates the provided computation and runs it
# with the distributed input.
@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

@tf.function
def distributed_test_step(dataset_inputs):
  return strategy.run(test_step, args=(dataset_inputs,))


if __name__ == "__main__":
  EPOCHS = 50
  gpus = tf.config.list_logical_devices('GPU')
  print(gpus)
  # Load data and distribute it

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

  strategy = tf.distribute.MirroredStrategy() 

  distributed_dataset = strategy.experimental_distribute_dataset(train_dataset)
  test_dist_dataset = strategy.experimental_distribute_dataset(val_dataset) 

  with strategy.scope():
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE)
    def compute_loss(labels, predictions, model_losses):
      per_example_loss = loss_object(labels, predictions)
      loss = tf.nn.compute_average_loss(per_example_loss)
      if model_losses:
        loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
      return loss

  with strategy.scope():
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

  # A model, an optimizer, and a checkpoint must be created under `strategy.scope`.
  with strategy.scope():
    model = vgg16()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


  time_start = time.time()

  for epoch in range(EPOCHS):
    # TRAIN LOOP
    total_loss = 0.0
    num_batches = 0
    for x in distributed_dataset:
      total_loss += distributed_train_step(x)
      num_batches += 1
    train_loss = total_loss / num_batches

  # TEST LOOP
  for x in test_dist_dataset:
    distributed_test_step(x)

  # if epoch % 2 == 0:
    # checkpoint.save(checkpoint_prefix)
  total_time = time.time() - time_start
  template = ("Time: {} Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
              "Test Accuracy: {}")
  print(template.format(total_time, epoch + 1, train_loss,
                        train_accuracy.result() * 100, test_loss.result(),
                        test_accuracy.result() * 100))

  test_loss.reset_states()
  train_accuracy.reset_states()
  test_accuracy.reset_states()
