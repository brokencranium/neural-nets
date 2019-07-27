import urllib.request
import zipfile
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow.python.keras.api._v1.keras as keras
from matplotlib.figure import Figure
from tensorflow.python.keras.api._v1.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.python.keras.api._v1.keras.preprocessing import image

data_dir = '../data/catsdogs/kaggle/'

dogs_path = '../data/catsdogs/kaggle/PetImages/Dog/'
cats_path = '../data/catsdogs/kaggle/PetImages/Cat/'

train_path = data_dir + 'PetImages/train'
validation_path = data_dir + 'PetImages/validation'
test_path = data_dir + 'PetImages/test'

dogs_train_path = '../data/catsdogs/kaggle/PetImages/train/dogs'
cats_train_path = '../data/catsdogs/kaggle/PetImages/train/cats'

dogs_validation_path = '../data/catsdogs/kaggle/PetImages/validation/dogs'
cats_validation_path = '../data/catsdogs/kaggle/PetImages/validation/cats'


class CheckAccuracy(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("Reached 95% accuracy")
            self.model.stop_training = True


def download_data(zip_file_path: str):
    url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'
    urllib.request.urlretrieve(url, zip_file_path)


def extract_data(zip_file: str):
    zip_ref = zipfile.ZipFile(file=zip_file, mode="r")
    zip_ref.extractall(data_dir)
    zip_ref.close()


def display_images(cats_list, dogs_list):
    row_count = 4
    col_count = 4

    fig: Figure = plt.gcf()
    fig.set_size_inches(row_count * 4, col_count * 4)

    for i, image_path in enumerate(cats_list[0:8] + dogs_list[0:8]):
        sub_plot = plt.subplot(row_count, col_count, i + 1)
        sub_plot.axis('Off')
        img = mpimg.imread(image_path)
        plt.imshow(img)

    plt.show()


def set_up_data(split_size):
    cats_list_path = [os.path.join(cats_path, i) for i in os.listdir(cats_path)]
    dogs_list_path = [os.path.join(dogs_path, i) for i in os.listdir(dogs_path)]

    display_images(cats_list_path, dogs_list_path)

    try:
        os.makedirs(dogs_train_path, exist_ok=True)
        os.makedirs(cats_train_path, exist_ok=True)
        os.makedirs(dogs_validation_path, exist_ok=True)
        os.makedirs(cats_validation_path, exist_ok=True)
    except OSError as err:
        print(err)
        pass
        return

    cats_count = len(cats_list_path)
    dogs_count = len(dogs_list_path)
    train_count = int(split_size * dogs_count)

    dogs_mask = np.concatenate([np.ones(train_count, dtype=bool), np.zeros(dogs_count - train_count, dtype=bool)])
    np.random.shuffle(dogs_mask)

    cats_mask = np.concatenate([np.ones(train_count, dtype=bool), np.zeros(cats_count - train_count, dtype=bool)])
    np.random.shuffle(cats_mask)

    dogs_train = np.array(dogs_list_path)[dogs_mask]
    dogs_validation = np.array(dogs_list_path)[np.logical_not(dogs_mask)]

    cats_train = np.array(cats_list_path)[cats_mask]
    cats_validation = np.array(cats_list_path)[np.logical_not(cats_mask)]

    print("Dogs :", dogs_count)
    print("Dogs Train :", len(dogs_train))
    print("Dogs validation :", len(dogs_validation))
    print("Cats :", cats_count)
    print("Cats Train :", len(cats_train))
    print("Cats validation:", len(cats_validation))

    #
    for file_path in dogs_train:
        shutil.move(file_path, dogs_train_path)

    for file_path in cats_train:
        shutil.move(file_path, cats_train_path)

    for file_path in dogs_validation:
        shutil.move(file_path, dogs_validation_path)

    for file_path in cats_validation:
        shutil.move(file_path, cats_validation_path)


def pre_processing():
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

    validation_generator = ImageDataGenerator(rescale=1.0 / 255.0,
                                              rotation_range=60,
                                              featurewise_center=True,
                                              featurewise_std_normalization=True,
                                              width_shift_range=0.2,
                                              height_shift_range=0.2,
                                              shear_range=0.2,
                                              zoom_range=0.2,
                                              horizontal_flip=True,
                                              vertical_flip=True)

    _train_flow = train_generator.flow_from_directory(directory=train_path,
                                                      target_size=(256, 256),
                                                      color_mode='rgb',
                                                      class_mode='binary',
                                                      batch_size=128,
                                                      shuffle=True,
                                                      interpolation='nearest'
                                                      )
    _validation_flow = validation_generator.flow_from_directory(directory=validation_path,
                                                                target_size=(256, 256),
                                                                color_mode='rgb',
                                                                class_mode='binary',
                                                                batch_size=128,
                                                                shuffle=True,
                                                                interpolation='nearest'
                                                                )

    return _train_flow, _validation_flow


def build_model():
    _model = keras.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=(256, 256, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    _model.summary()
    return _model


def test_model(model_in):
    for test_img in os.listdir(test_path):
        img = image.load_img(test_path + 'cat.jpg', target_size=(256, 256))
        test_img_array = image.img_to_array(img)
        test_img_array = np.expand_dims(test_img_array, axis=0)
        image_vstack = np.vstack([test_img_array])
        classes = model_in.predict(image_vstack, batch_size=1)
        print(classes[0])
        if classes[0] > 0:
            print(test_img + " is a dog")
        else:
            print(test_img + " is a cat")


def display_convolution_layers(model_in: keras.models.Sequential):
    # figure, axis = plt.sublots(3, 4)
    layer_outputs = [layer.output for layer in model_in.layers]
    layer_names = [layer.name for layer in model_in.layers]

    test_img_path = os.path.join(test_path, 'cat.jpg')
    img = load_img(test_img_path, target_size=(256, 256))
    img_arr = img_to_array(img)
    # number of images is 1
    img_arr = img_arr.reshape((1,) + img_arr.shape)
    img_arr = img_arr / 255.0

    visualization_model = keras.models.Model(inputs=model_in.input, outputs=layer_outputs)
    predict_feature_maps = visualization_model.predict(img_arr)

    for layer_name, feature_map in zip(layer_names, predict_feature_maps):
        if len(feature_map.shape) == 4:
            n_features = feature_map.shape[-1]
            size = feature_map.shape[1]

            display_grid = np.zeros((size, size * n_features))

            for i in range(n_features):
                img_feature = feature_map[0, :, :, i]
                img_feature -= img_feature.mean()
                img_feature /= img_feature.std()
                img_feature *= 64
                img_feature += 126
                img_feature = np.clip(img_feature, 0, 255).astype('uint8')
                display_grid[:, i * size: (i + 1) * size] = img_feature

            scale = 20.0 / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()


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
    # download_data(data_dir + 'dogs_cats.zip')
    # extract_data(data_dir + 'dogs_cats.zip')
    # set_up_data(split_size=0.9)
    train_flow, validation_flow = pre_processing()
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    check_accuracy = CheckAccuracy()

    model.fit_generator(generator=train_flow,
                        epochs=10,
                        steps_per_epoch=88,
                        callbacks=[check_accuracy],
                        validation_data=validation_flow,
                        validation_steps=10,
                        use_multiprocessing=True
                        )

    test_model()
    display_convolution_layers(model)
    plot_data(model)
