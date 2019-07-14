import zipfile
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow.python.keras.api._v1.keras as keras
from tensorflow.python.keras.api._v1.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.api._v1.keras.preprocessing import image

data_dir = './data/humanhorses'
train_dir = data_dir + 'train/'
validation_dir = data_dir + 'validation/'
train_horses_dir = data_dir + 'train/horses/'
train_humans_dir = data_dir + 'train/humans/'
test_dir = data_dir + 'test/'


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


def build_model():
    _model = keras.models.Sequential([
            keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(300, 300, 3)),
            keras.layers.MaxPooling2D(2, 2),

            keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),

            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),

            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),

            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),

            keras.layers.Dense(1, activation='sigmoid')
            ])

    _model.summary()
    return _model


def data_generator(dir_name):
    # Image generator provides the labels too, since this is a binary classification the class_mode is binary
    data_gen = ImageDataGenerator(rescale=1.0 / 255)
    return data_gen.flow_from_directory(directory=dir_name,
                                        target_size=(300, 300),
                                        batch_size=128,
                                        class_mode='binary')


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
    model = build_model()
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
                  metrics=['acc'])

    train_model = model.fit_generator(
        generator=data_generator(train_dir),
        validation_data=data_generator(validation_dir),
        steps_per_epoch=8,
        epochs=2,
        verbose=1
        )

    execute_model(model)
    visualize_layers(model)
