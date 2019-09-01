import numpy as np
import tensorflow_datasets as tfds
import tensorflow.python.keras.api._v1.keras as keras
import matplotlib.pyplot as plt
import io
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

data_path = '../data/'
output_path = data_path + 'imdb_reviews/'
vocab_size = 100000
max_len = 120
padding_type = 'post'
truncating_type = 'post'
embedding_dim = 16
num_epochs = 10


def get_data(path: str):
    imdb, info = tfds.load("imdb_reviews",
                           with_info=True,
                           as_supervised=True,
                           data_dir=path)
    _train_data, _val_data = imdb['train'], imdb['test']

    _train_sentences = []
    _val_sentences = []
    _train_labels = []
    _val_labels = []

    for s, l in _train_data:
        _train_sentences.append((s.numpy()).decode('UTF-8'))
        _train_labels.append(l.numpy())

    for s, l in _val_data:
        _val_sentences.append((s.numpy()).decode('UTF-8'))
        _val_labels.append(l.numpy())

    return _train_sentences, np.array(_train_labels), _val_sentences, np.array(_val_labels)


def decode_review(text, _reverse_word_index):
    return ' '.join([_reverse_word_index.get(i, '?') for i in text])


def build_model():
    _model = keras.Sequential([
        keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        # keras.layers.Flatten(), Flatten may not work
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])
    print("Model summary: ", _model.summary())
    return _model


def plot_data(_history):
    acc = _history['accuracy']
    val_acc = _history['val_accuracy']
    loss = _history['loss']
    val_loss = _history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    plt.show()


def download_embeddings(_model, _reverse_word_index):
    e = model.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape)  # shape: (vocab_size, embedding_dim)

    out_meta = io.open(output_path + 'meta.tsv', 'w', encoding='utf-8')
    out_vecs = io.open(output_path + 'vecs.tsv', 'w', encoding='utf-8')

    for word_num in range(1, vocab_size):
        word = _reverse_word_index[word_num]
        embeddings = weights[word_num]

        out_meta.write(word + "\n")
        out_vecs.write("\t".join([str(x) for x in embeddings]) + "\n")

    out_meta.close()
    out_vecs.close()


def get_padded_sequence():
    _train_sequence = text_tokenizer.texts_to_sequences(train_data)
    _train_padded = pad_sequences(sequences=_train_sequence,
                                  maxlen=max_len,
                                  padding=padding_type,
                                  truncating=truncating_type)
    # train_labels_sequence = label_tokenizer.texts_to_sequences(train_labels)
    # val_labels_sequence = label_tokenizer.texts_to_sequences(val_labels)
    _val_sequence = text_tokenizer.texts_to_sequences(val_data)
    _val_padded = pad_sequences(sequences=_val_sequence,
                                maxlen=max_len,
                                padding=padding_type,
                                truncating=truncating_type)

    return _train_padded, _val_padded


if __name__ == '__main__':
    train_data, train_labels, val_data, val_labels = get_data(data_path)

    text_tokenizer = Tokenizer(num_words=vocab_size, oov_token='OOV')
    text_tokenizer.fit_on_texts(train_data)
    word_index = text_tokenizer.word_index
    print("Word count", text_tokenizer.num_words)
    print("Word index ", text_tokenizer.word_index)

    train_padded, val_padded = get_padded_sequence()

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    print(decode_review(train_padded[1], reverse_word_index))
    print(train_data[1])

    model = build_model()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x=train_padded,
                        y=train_labels,
                        epochs=num_epochs,
                        validation_data=(val_padded, val_labels))
    plot_data(history.history)

    download_embeddings(model, reverse_word_index)
