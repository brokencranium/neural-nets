import zipfile
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow.python.keras.api._v1.keras as keras
from tensorflow.python.keras.api._v1.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.api._v1.keras.preprocessing import image
from tensorflow.python.keras.api._v1.keras.applications.inception_v3 import InceptionV3

data_dir = '../data/humanhorses/'
train_dir = data_dir + 'train/'
validation_dir = data_dir + 'validation/'
train_horses_dir = data_dir + 'train/horses/'
train_humans_dir = data_dir + 'train/humans/'
test_dir = data_dir + 'test/'
local_weights_file = '../data/inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


class CheckAccuracy(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.95:
            print("Reached 95% accuracy")
            self.model.stop_training = True


def extract_images(zip_file_path: str, output_dir: str):
    zip_oref = zipfile.ZipFile(zip_file_path, 'r')
    zip_oref.extractall(output_dir)
    zip_oref.close()


def data_inspect():
    train_horses_names = os.listdir(train_horses_dir)
    train_humans_names = os.listdir(train_humans_dir)

    print((train_horses_names[:10]))
    print((train_humans_names[:10]))

    print('Total training horse images:', len(os.listdir(train_horses_dir)))
    print('Total training human images:', len(os.listdir(train_humans_dir)))


def display_images():
    num_rows = 4
    num_cols = 4

    img_index = 0

    fig = plt.gcf()
    fig.set_size_inches(num_cols * 4, num_rows * 4)
    img_index += 8
    next_horse_pics = [os.path.join(train_horses_dir, file_name) for file_name in
                       os.listdir(train_horses_dir)[img_index - 8: img_index]]
    next_human_pics = [os.path.join(train_humans_dir, file_name) for file_name in
                       os.listdir(train_humans_dir)[img_index - 8: img_index]]

    for i, img_path in enumerate(next_horse_pics + next_human_pics):
        sub_plot = plt.subplot(num_rows, num_cols, i + 1)
        sub_plot.axis('Off')

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()


def get_model():
    _model = InceptionV3(input_shape=(300, 300, 3),
                         include_top=False,
                         weights=None)

    _model.load_weights(local_weights_file)

    for layer in _model.layers:
        layer.trainable = False

    _model.summary()
    return _model


def update_model(model_in: keras.models.Sequential):
    last_layer = model_in.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    x = last_layer.output
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=1024, activation='relu')(x)
    x = keras.layers.Dropout(rate=0.3)(x)
    x = keras.layers.Dense(units=1, activation='sigmoid')(x)
    return x


def data_generator(train_dir_path, val_dir_path):
    # Image generator provides the labels too, since this is a binary classification the class_mode is binary
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

    _train_flow = _train_gen.flow_from_directory(directory=train_dir_path,
                                                 target_size=(300, 300),
                                                 batch_size=128,
                                                 class_mode='binary')

    _val_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    _val_flow = _val_gen.flow_from_directory(directory=val_dir_path,
                                             target_size=(300, 300),
                                             batch_size=128,
                                             class_mode='binary')

    return _train_flow, _val_flow


def get_test_image(image_name: str):
    img = image.load_img(os.path.join(test_dir, image_name), target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255
    # return np.vstack([img_array])


def execute_model(in_model):
    predict_image(in_model, np.vstack([get_test_image('horse.jpeg')]))
    predict_image(in_model, np.vstack([get_test_image('human.png')]))


def predict_image(in_model, test_image):
    classes = in_model.predict(test_image, batch_size=1)
    print(classes[0])
    if classes[0] > 0.5:
        print('Input image belongs to horse')
    else:
        print('Input image belongs to human')


def plot_data(model_in):
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


def visualize_layers(in_model):
    print('test')
    # Skip the first one
    outputs = [layer.output for layer in in_model.layers[1:]]
    # outputs = [ layer.output for layer in in_model.layers[1:]]
    visualization_model = keras.models.Model(inputs=in_model.input, outputs=outputs)

    img = get_test_image('horse.jpeg')
    # I think (1,) is a tuple and img shape is being added to it, result is (1,300,300,3)
    # 3 is the channel and when the PIL img is converted to numpy array, channel value
    # becomes prominent
    # x = img.reshape((1,) + img.shape)

    # Rescale to 255
    x = img / 255

    feature_maps = visualization_model.predict(x)
    layers_names = [layer for layer in in_model.layers]

    for layer_name, feature_map in zip(layers_names, feature_maps):
        # The shape of the feature map is set to 4 for only Conv2D and Maxpool layers
        if len(feature_map.shape) == 4:
            # Number of features
            features_count = feature_map.shape[-1]

            # feature map shape (1(image count), size, size, feature count)
            size = feature_map.shape[1]

            # Creating a tile, rows are set to 300 and columns size is set to 300 * number of features
            display_grid = np.zeros((size, size * features_count))

            for i in range(features_count):
                # Postprocess the feature to make it visually palatable
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # Fill the tile with each feature
                display_grid[:, i * size: (i + 1) * size] = x

            scale = 20. / features_count
            plt.figure(figsize=(scale * features_count, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()


if __name__ == '__main__':
    # extract_images(train_dir)
    data_inspect()
    display_images()
    pre_trained_model = get_model()
    updated_model = update_model(pre_trained_model)
    model = keras.Model(pre_trained_model.input, updated_model)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
                  metrics=['acc'])
    check_accuracy = CheckAccuracy()
    train_gen, val_gen = data_generator(train_dir, validation_dir)
    train_model = model.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        steps_per_epoch=8,
        epochs=2,
        callbacks=[check_accuracy],
        verbose=1
    )

    execute_model(model)
    # visualize_layers(model)
    plot_data(train_model)
