import tensorflow as tf
import tensorflow.python.keras.api._v1.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras import models


def build_model() -> keras.models.Sequential:
    _model = keras.models.Sequential([
            # The number of convolutions you want to generate. Purely arbitrary, but good to start with something in
            # the order of 32
            # The size of the Convolution, in this case a 3x3 grid
            # The activation function to use -- in this case we'll use relu, which you might recall is the equivalent
            # of returning x when x>0, else returning 0
            # In the first layer, the shape of the input data.
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu',
                                input_shape=(28, 28, 1)),
            # You'll follow the Convolution with a MaxPooling layer which is then designed to compress the image,
            # while maintaining the content of the features that were highlighted by the convlution. By specifying (
            # 2,2) for the MaxPooling, the effect is to quarter the size of the image. Without going into too much
            # detail here, the idea is that it creates a 2x2 array of pixels, and picks the biggest one, thus turning
            # 4 pixels into 1. It repeats this across the image, and in so doing halves the number of horizontal,
            # and halves the number of vertical pixels, effectively reducing the image by 25%.
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            # Now flatten the output. After this you'll just have the same DNN structure as the non convolutional
            # version
            keras.layers.Flatten(),
            # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dense(units=10, activation='softmax')
            ])

    _model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # You can call model.summary() to see the size and shape of the network, and you'll notice that after every
    # MaxPooling layer, the image size is reduced in this way.
    _model.summary()
    return _model


def display_conv_layers(model_seq: keras.models.Sequential):
    f, axarr = plt.subplots(3, 4)
    FIRST_IMAGE = 0
    SECOND_IMAGE = 23
    THIRD_IMAGE = 28
    CONVOLUTION_NUMBER = 1

    # There are six layers in the layer_outputs
    # Layer0 - Tensor("conv2d/Identity:0", shape=(None, 26, 26, 64), dtype=float32)
    # Layer1 - Tensor("max_pooling2d/Identity:0", shape=(None, 13, 13, 64), dtype=float32)
    # Layer2 - Tensor("conv2d_1/Identity:0", shape=(None, 11, 11, 64), dtype=float32)
    # Layer3 - Tensor("max_pooling2d_1/Identity:0", shape=(None, 5, 5, 64), dtype=float32)
    # Layer4 - Tensor("flatten/Identity:0", shape=(None, 1600), dtype=float32)
    # Layer5 - Tensor("dense/Identity:0", shape=(None, 128), dtype=float32)
    # Layer6 - Tensor("dense_1/Identity:0", shape=(None, 10), dtype=float32)

    layer_outputs = [layer.output for layer in model.layers]
    # model_seq.input - Tensor("conv2d_input:0", shape=(None, 28, 28, 1), dtype=float32)
    activation_model = keras.models.Model(inputs=model_seq.input, outputs=layer_outputs)

    for x in range(0, 4):
        # Reshape size
        # 1 - number of images
        # 28X28 - size of the image
        # 1 - Channels for the image

        # activation_model.predict contains 6 array elements, I am assuming its one for every layer
        # In this example only 4 layers are displayed, two convolution to and two max pooling
        # It is not required to loop 4 times, instead predict can be performed once and x could be replaces with 0
        # through 3(5)
        f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
        # f1[0 is the first element
        # :,: is probably for all the x,y pixels
        # CONVOLUTION_NUMBER could the filter number at particular layer
        axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[0, x].grid(False)
        f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[1, x].grid(False)
        f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[2, x].grid(False)

    plt.show()


if __name__ == '__main__':
    # (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    # Step 1 is to gather the data. You'll notice that there's a bit of a change here in that the training data
    # needed to be reshaped. That's because the first convolution expects a single tensor containing everything,
    # so instead of 60,000 28x28x1 items in a list, we have a single 4D list that is 60,000x28x28x1, and the same for
    # the test images. If you don't do this, you'll get an error when training as the Convolutions do not recognize
    # the shape.
    train_images = train_images.reshape(60000, 28, 28, 1)
    test_images = test_images.reshape(10000, 28, 28, 1)

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = build_model()
    model.fit(train_images, train_labels, epochs=3)
    test_loss = model.evaluate(test_images, test_labels)
    print(test_labels[:100])

    display_conv_layers(model)
