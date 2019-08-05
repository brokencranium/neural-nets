import csv
import numpy as np
import tensorflow.python.keras.api._v1.keras as keras
from tensorflow.python.keras.api._v1.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.api._v1.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

weights_file = '../data/inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
train_url = 'https://www.kaggle.com/datamunge/sign-language-mnist/downloads/sign-language-mnist.zip'
download_path = '../data/'
train_path = '../data/signs/train/sign_mnist_train.csv'
val_path = '../data/signs/val/sign_mnist_test.csv'


class CheckAccuracy(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.95:
            print("Reached 95% accuracy")
            self.model.stop_training = True


def pre_processing(train_x, train_y, val_x, val_y):
    _train_gen = ImageDataGenerator(rescale=1.0 / 255.0,
                                    rotation_range=60,
                                    featurewise_center=True,
                                    featurewise_std_normalization=True,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True)
    _train_flow = _train_gen.flow(train_x,
                                  train_y,
                                  batch_size=32,
                                  shuffle=True)

    _val_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    _val_flow = _train_gen.flow(val_x,
                                val_y,
                                batch_size=32,
                                shuffle=True)

    return _train_flow, _val_flow


def get_data(file_name):
    with open(file_name) as input_iter:
        csv_reader = csv.reader(input_iter, delimiter=',')
        images = []
        labels = []
        image_array = np.zeros([78, 78, 3])
        skip_flag = 'N'
        for line in csv_reader:
            if skip_flag == 'N':
                skip_flag = 'Y'
            else:
                img = np.reshape(line[1:], (-1, 28))
                img = np.pad(img, [(25, 25), (25, 25)], mode='constant', constant_values=0)
                image_array[:, :, 0] = img
                image_array[:, :, 1] = img
                image_array[:, :, 2] = img
                label = line[0]
                # im = Image.fromarray(image_array.astype(np.uint8))
                # im.save("test.jpeg")

                images.append(image_array)
                labels.append(label)

    return np.array(images).astype('float'), np.array(labels).astype('float')


def get_model():
    _model = InceptionV3(input_shape=(78, 78, 3),
                         include_top=False,
                         weights=None)
    _model.load_weights(weights_file)

    for layer in _model.layers:
        layer.trainable = False

    _model.summary()
    return _model


def update_model(_model):
    last_layer = _model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output
    # Flatten the output layer to 1 dimension
    x = keras.layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = keras.layers.Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = keras.layers.Dropout(0.2)(x)
    # fc
    x = keras.layers.Dense(512, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = keras.layers.Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    x = keras.layers.Dense(26, activation='softmax')(x)
    return x


def plot_model_performance(_history):
    acc = _history.history['acc']
    val_acc = _history.history['val_acc']

    loss = _history.history['loss']
    val_loss = _history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, max(plt.ylim())])
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == '__main__':
    # set_data()
    train_images, train_labels = get_data(train_path)
    val_images, val_labels = get_data(val_path)
    print("Train : {} {}".format(train_images.shape, train_labels.shape))
    print("Validation : {} {}".format(val_images.shape, val_labels.shape))

    train_flow, val_flow = pre_processing(train_images, train_labels, val_images, val_labels)

    pre_trained_model = get_model()

    updated_model = update_model(pre_trained_model)

    model = keras.Model(pre_trained_model.input, updated_model)

    model.compile(optimizer=RMSprop(lr=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    check_accuracy = CheckAccuracy()
    history = model.fit_generator(train_flow,
                                  validation_data=val_flow,
                                  steps_per_epoch=len(train_images)/32,
                                  epochs=20,
                                  validation_steps=len(val_images)/32,
                                  callbacks=[check_accuracy],
                                  verbose=2)

    model.save(download_path + 'signs.h5')
    plot_model_performance(history)
