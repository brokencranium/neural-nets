import zipfile
import os
import matplotlib.pyplot as plt
import matplotlib.image as mat_img
import tensorflow.python.keras.api._v1.keras as keras
from tensorflow.python.keras.api._v1.keras.preprocessing.image import ImageDataGenerator
import numpy as np


class CheckAccuracy(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.999:
            print('Reached 99.9% accuracy')
            self.model.stop_training = True


'''
!wget --no-check-certificate \
    "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
    -O "./data/happysad/happy-or-sad.zip"
'''
data_dir = './data/happysad/'
train_dir = './data/happysad/train/'
eval_dir = './data/happysad/eval/'
test_dir = './data/happysad/test/'
happy_train_dir = train_dir + 'happy/'
sad_train_dir = train_dir + 'sad/'


def get_data(zip_file_path: str):
    zip_ref = zipfile.ZipFile(zip_file_path, mode="r")
    zip_ref.extractall(path=data_dir)
    zip_ref.close()


def data_inspect():
    train_happy_names = os.listdir(happy_train_dir)
    train_sad_names = os.listdir(sad_train_dir)

    print(train_happy_names[:10])
    print(train_sad_names[:10])

    print('Total happy images', len(train_happy_names))
    print('Total sad images', len(train_sad_names))
    image = mat_img.imread(happy_train_dir + train_happy_names[0])
    print('Image size', image.shape)


def display_original_images(index: int = 1):
    num_rows = 4
    num_cols = 4

    fig = plt.gcf()
    fig.set_size_inches(num_cols * 4, num_rows * 4)

    happy_pics = [os.path.join(happy_train_dir, image_name) for image_name in
                  os.listdir(happy_train_dir)[index: index + 8]]
    sad_pics = [os.path.join(sad_train_dir, image_name) for image_name in os.listdir(sad_train_dir)[index: index + 8]]

    for i, image in enumerate(happy_pics + sad_pics):
        sub_plot = plt.subplot(num_rows, num_cols, i + 1)
        sub_plot.axis('Off')

        img = mat_img.imread(image)
        plt.imshow(img)

    plt.show()


def build_model_layers():
    print("Layers")
    _model = keras.models.Sequential([
            keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),

            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
            ])
    _model.summary()
    return _model


def get_data_generator(src_dir: str):
    data_generator = ImageDataGenerator(rescale=1.0 / 255, )
    return data_generator.flow_from_directory(directory=src_dir,
                                              target_size=(150, 150),
                                              color_mode='rgb',
                                              class_mode='binary',
                                              batch_size=10)


def test_model(in_model):
    test_images = [os.path.join(test_dir + image_name) for image_name in os.listdir(test_dir)]

    for image in test_images:
        img = mat_img.imread(image)
        img = img[:, :, :3]
        img = img.reshape(1, 150, 150, 3)
        img /= 255.0
        img = np.vstack([img])

        classes = in_model.predict(img, batch_size=1)
        print(classes[0, 0])
        print(classes[0, 1])
        if classes[0, 0] > classes[0, 1]:
            print(":-)")
        else:
            print(":-(")


if __name__ == '__main__':
    # get_data(data_dir + 'happy-or-sad.zip')
    data_inspect()
    display_original_images(index=1)
    model = build_model_layers()
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    check_accuracy = CheckAccuracy()
    model.fit_generator(generator=get_data_generator(train_dir),
                        steps_per_epoch=8,
                        epochs=15,
                        verbose=1,
                        callbacks=[check_accuracy]
                        )
    test_model(model)
