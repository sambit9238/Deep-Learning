from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import tensorflow as tf
import datetime

from tensorboard import program
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', "."])
url = tb.launch()
print("tensorboard url: ", url)


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape)

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
print(x_train.shape)

#tf.data to batch and shuffle the dataset
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(32)

print(train_ds.element_spec)
print(train_ds)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(256, activation='relu')
        self.d2 = tf.keras.layers.Dense(128, activation='relu')
        self.d3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x

# optimizer and loss function to train
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()


# metrics to measure the loss and the accuracy of the model
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')

'''
With eager execution enabled, Tensorflow will calculate the values of tensors as they occur in your code. This means that it won't precompute a static graph for which inputs are fed in through placeholders. This means to back propagate errors, you have to keep track of the gradients of your computation and then apply these gradients to an optimiser.

This is very different from running without eager execution, where you would build a graph and then simply use sess.run to evaluate your loss and then pass this into an optimiser directly.

Fundamentally, because tensors are evaluated immediately, you don't have a graph to calculate gradients and so you need a gradient tape. It is not so much that it is just used for visualisation, but more that you cannot implement a gradient descent in eager mode without it.

Obviously, Tensorflow could just keep track of every gradient for every computation on every tf.Variable. However, that could be a huge performance bottleneck. They expose a gradient tape so that you can control what areas of your code need the gradient information. Note that in non-eager mode, this will be statically determined based on the computational branches that are descendants of your loss but in eager mode there is no static graph and so no way of knowing.
'''


@tf.function
def train_step(model, optimizer, images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(model, images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

EPOCHS = 5

# Create an instance of the model
model = MyModel()

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(model, optimizer, images, labels)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

  # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
