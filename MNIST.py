import tensorflow as tf
import tensorflow.python.keras.api._v1.keras as keras


class MyCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training! Epoch ", epoch)
            self.model.stop_training = True


mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255
test_images = test_images / 255

train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

model = keras.models.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation=keras.activations.relu,
                            input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation=keras.activations.relu),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.keras.activations.relu),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
        ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callback = MyCallback()
model.fit(train_images, train_labels, epochs=10, callbacks=[callback])
model.evaluate(test_images, test_labels)
