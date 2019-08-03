import urllib.request as request
import zipfile
import os
from tensorflow.python.keras.api._v1.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.api._v1.keras.preprocessing import image
import tensorflow.python.keras.api._v1.keras as keras
import matplotlib.pyplot as plt
import numpy as np

# import matplotlib.image as mpimg

train_url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip'
val_url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip'
download_path = '../data/'
train_path = '../data/rocscipap/train/'
val_path = '../data/rocscipap/val/'
test_path = '../data/rocscipap/test/rps-validation'


class CheckAccuracy(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("Reached 95% accuracy")
            self.model.stop_training = True


def extract_data(zip_path: str, output_dir: str):
    zip_ref = zipfile.ZipFile(zip_path, mode='r')
    zip_ref.extractall(output_dir)
    zip_ref.close()


def download_file(url_path: str, output_path: str):
    request.urlretrieve(url_path, output_path)


def set_data():
    download_zip = download_path + 'temp.zip'
    download_file(train_url, download_zip)
    extract_data(download_zip, train_path)
    os.remove(download_zip)
    download_file(val_url, download_zip)
    extract_data(download_zip, val_path)
    os.remove(download_zip)


def build_model():
    _model = keras.models.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    _model.summary()
    return _model


def pre_processing():
    print('Pre-processing')
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

    _train_flow = train_generator.flow_from_directory(directory=train_path + 'rps',
                                                      target_size=(150, 150),
                                                      color_mode='rgb',
                                                      class_mode='categorical',
                                                      batch_size=32,
                                                      shuffle=True,
                                                      interpolation='nearest'
                                                      )

    val_generator = ImageDataGenerator(rescale=1.0 / 255.0)
    _val_flow = val_generator.flow_from_directory(directory=train_path + 'rps',
                                                  target_size=(150, 150),
                                                  color_mode='rgb',
                                                  class_mode='categorical',
                                                  batch_size=32,
                                                  shuffle=True,
                                                  interpolation='nearest'
                                                  )
    return _train_flow, _val_flow


def plot_model_performance(_history):
    acc = _history.history['acc']
    val_acc = _history.history['val_acc']
    loss = _history.history['loss']
    val_loss = _history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()


# def view_data():
# pic_index = 2
# next_rock = [os.path.join(rock_dir, fname)
#              for fname in rock_files[pic_index-2:pic_index]]
# next_paper = [os.path.join(paper_dir, fname)
#               for fname in paper_files[pic_index-2:pic_index]]
# next_scissors = [os.path.join(scissors_dir, fname)
#                  for fname in scissors_files[pic_index-2:pic_index]]
#
# for i, img_path in enumerate(next_rock+next_paper+next_scissors):
#     #print(img_path)
#     img = mpimg.imread(img_path)
#     plt.imshow(img)
#     plt.axis('Off')
#     plt.show()


def predict_images(_model: keras.Sequential):
    test_images = [os.path.join(test_path, file) for file in os.listdir(test_path)]

    for test_image in test_images:
        img = image.load_img(test_image, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=1)
        print('File name: ', test_image)
        print(classes)


if __name__ == '__main__':
    # set_data()
    train_flow, val_flow = pre_processing()
    model: keras.Sequential = build_model()
    check_accuracy = CheckAccuracy()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    history = model.fit_generator(generator=train_flow,
                                  epochs=25,
                                  validation_data=val_flow,
                                  use_multiprocessing=True,
                                  callbacks=[check_accuracy],
                                  verbose=2
                                  )
    model.save(download_path + 'rocscipap.h5')

    predict_images(model)
