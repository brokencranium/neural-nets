import os
import zipfile
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow.python.keras.api._v1.keras as keras
from tensorflow.python.keras.api._v1.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.python.keras.api._v1.keras.preprocessing import image
import numpy as np

data_dir = './data/catsdogs/'
train_dir = data_dir + 'train/'
validation_dir = data_dir + 'validation/'
test_dir = data_dir + 'test/'


class CheckAccuracy(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.999:
            print("Reached 99.9% accuracy")
            self.model.stop_training = True


def extract_data(zip_file: str):
    zip_ref = zipfile.ZipFile(file=zip_file, mode="r")
    zip_ref.extractall(data_dir)
    zip_ref.close()


def display_data():
    train_cats = [os.path.join(train_dir, 'cats/', image_name) for image_name in os.listdir(train_dir + 'cats/')][0:8]
    train_dogs = [os.path.join(train_dir, 'dogs/', image_name) for image_name in os.listdir(train_dir + 'dogs/')][0:8]

    row_count = 4
    col_count = 4

    fig = plt.gcf()
    fig.set_size_inches(row_count * 4, col_count * 4)

    for i, image_path in enumerate(train_cats + train_dogs):
        # subplot indices start at one
        sub_plot = plt.subplot(row_count, col_count, i + 1)
        sub_plot.axis('Off')

        img = mpimg.imread(image_path)
        plt.imshow(img)

    plt.show()


def build_model():
    _model = keras.models.Sequential([
            keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)),
            keras.layers.MaxPooling2D(2, 2),

            keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),

            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),

            keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),

            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
            ])
    _model.summary()
    return _model


def pre_processing():
    train_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    validation_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    _train_gen = train_data_gen.flow_from_directory(directory=train_dir,
                                                    target_size=(150, 150),
                                                    color_mode='rgb',
                                                    batch_size=10,
                                                    class_mode='binary'
                                                    )
    _validation_gen = validation_data_gen.flow_from_directory(directory=validation_dir,
                                                              target_size=(150, 150),
                                                              color_mode='rgb',
                                                              batch_size=64,
                                                              class_mode='binary'
                                                              )
    return _train_gen, _validation_gen


def test_model(model_in):
    for test_img in os.listdir(test_dir):
        img = image.load_img(test_dir + 'cat1.jpg', target_size=(150, 150))
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

    test_img_path = os.path.join(test_dir, 'cat1.jpg')
    img = load_img(test_img_path, target_size=(150, 150))
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


if __name__ == '__main__':
    # extract_data('./data/catsdogs/cats_and_dogs_filtered.zip')
    display_data()
    model = build_model()
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy']
                  )
    train_gen, validation_gen = pre_processing()
    check_accuracy = CheckAccuracy()
    model.fit_generator(generator=train_gen,
                        validation_data=validation_gen,
                        steps_per_epoch=10,
                        epochs=2,
                        callbacks=[check_accuracy],
                        validation_steps=50)

    test_model(model)
    display_convolution_layers(model)
