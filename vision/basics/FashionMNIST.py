import tensorflow as tf
import matplotlib.pyplot as plt


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.4:
            print("\nReached 60% accuracy so cancelling training! Epoch ", epoch)
            self.model.stop_training = True


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    plt.imshow(train_images[0])
    print(train_images[0])
    print(train_labels[0])

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    # train_images = train_images / 1.0
    # test_images = test_images / 1.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    callback = MyCallback()
    model.fit(train_images, train_labels, epochs=5, callbacks=[callback])

    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])
