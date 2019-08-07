import matplotlib.pyplot as plt
from tensorflow.python.keras.api._v1.keras.applications.inception_v3 import InceptionV3
import tensorflow.python.keras.api._v1.keras as keras
from tensorflow.python.keras.api._v1.keras.optimizers import RMSprop
from tensorflow.python.keras.api._v1.keras.preprocessing.image import ImageDataGenerator
import os
import random
from shutil import copyfile

# !wget --no-check-certificate \
#     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
#             -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

LOCAL_WEIGHTS_FILE = '../data/inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

CAT_SOURCE_DIR = "../data/catsdogs/kaggle/PetImages/Cat/"
TRAINING_CATS_DIR = "../data/catsdogs/kaggle/PetImages/train/cats/"
TESTING_CATS_DIR = "../data/catsdogs/kaggle/PetImages/validation/cats/"

DOG_SOURCE_DIR = "../data/catsdogs/kaggle/PetImages/Dog/"
TRAINING_DOGS_DIR = "../data/catsdogs/kaggle/PetImages/train/dogs/"
TESTING_DOGS_DIR = "../data/catsdogs/kaggle/PetImages/validation/dogs/"

TRAIN_DIR = '../data/catsdogs/kaggle/PetImages/train'
VALIDATION_DIR = '../data/catsdogs/kaggle/PetImages/validation'


class CheckAccuracy(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.95:
            print("Reached 95% accuracy")
            self.model.stop_training = True


def get_model():
    _model = InceptionV3(input_shape=(150, 150, 3),
                         include_top=False,
                         weights=None)

    _model.load_weights(LOCAL_WEIGHTS_FILE)

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
    # Add a final sigmoid layer for classification
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    return x


def split_data(source, training, validation, split_size):
    files = []
    for filename in os.listdir(source):
        file = source + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * split_size)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = source + filename
        destination = training + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = source + filename
        destination = validation + filename
        copyfile(this_file, destination)


def pre_processing(train_path, validation_path):
    train_generator = ImageDataGenerator(rescale=1.0 / 255.0,
                                         rotation_range=60,
                                         featurewise_center=True,
                                         featurewise_std_normalization=True,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True,
                                         vertical_flip=True)

    # Do not augment validation dataset
    validation_generator = ImageDataGenerator(rescale=1.0 / 255.0)

    _train_flow = train_generator.flow_from_directory(directory=train_path,
                                                      target_size=(150, 150),
                                                      color_mode='rgb',
                                                      class_mode='binary',
                                                      batch_size=128,
                                                      shuffle=True,
                                                      interpolation='nearest'
                                                      )
    _validation_flow = validation_generator.flow_from_directory(directory=validation_path,
                                                                target_size=(150, 150),
                                                                color_mode='rgb',
                                                                class_mode='binary',
                                                                batch_size=128,
                                                                shuffle=True,
                                                                interpolation='nearest'
                                                                )

    return _train_flow, _validation_flow


def plot_data(model_in: keras.models.Sequential):
    acc = model_in.history['acc']
    val_acc = model_in.history['val_acc']
    loss = model_in.history['loss']
    val_loss = model_in.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    plt.show()


if __name__ == '__main__':
    pre_trained_model = get_model()

    updated_model = update_model(pre_trained_model)

    model = keras.Model(pre_trained_model.input, updated_model)

    model.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # split_size = .9
    # split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
    # split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

    train_flow, validation_flow = pre_processing(TRAIN_DIR, VALIDATION_DIR)
    check_accuracy = CheckAccuracy()
    model.fit_generator(
        train_flow,
        validation_data=validation_flow,
        steps_per_epoch=100,
        epochs=20,
        validation_steps=50,
        callbacks=[check_accuracy],
        verbose=2)

    plot_data(model)
