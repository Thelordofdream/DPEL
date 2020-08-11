import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = datasets['train'], datasets['test']

strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
# strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# 您还可以执行 info.splits.total_num_examples 来获取总数
# 数据集中的样例数量。

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 18000

BATCH_SIZE_PER_REPLICA = 9000
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label


train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

with strategy.scope():
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

# 定义检查点（checkpoint）目录以存储检查点（checkpoints）
checkpoint_dir = './training_checkpoints'
# 检查点（checkpoint）文件的名称
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


# 在每个 epoch 结束时打印LR的回调（callbacks）。
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr))


callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    # PrintLR()
]

# model.fit(train_dataset, epochs=100, callbacks=callbacks)
model.fit(train_dataset, epochs=100)

