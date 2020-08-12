import tensorflow_datasets as tfds
import tensorflow as tf
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = datasets['train'], datasets['test']

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 18000
BATCH_SIZE = 9000


def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label


train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).repeat().batch(2000)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(14, 14, 32)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu', input_shape=(6, 6, 64)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

model.fit(train_dataset, validation_data=eval_dataset, steps_per_epoch=int(num_train_examples/BATCH_SIZE+1), validation_steps=int(num_train_examples/BATCH_SIZE+1), epochs=100)
model.save("tensorflow_single_mnist_model.ckpt")
