import tensorflow_datasets as tfds
import tensorflow.python.keras.api._v1.keras as keras
import io
import matplotlib.pyplot as plt

data_dir = '../data/imdb_subwords/'
output_path = data_dir + 'imdb_subwords/'
max_len = 120
adding_type = 'post'
truncating_type = 'post'
embedding_dim = 16
num_epochs = 10
buffer_size = 10000
batch_size = 64


def get_data():
    _train_data = None
    _val_data = None

    imdb, info = tfds.load(name="imdb_reviews/subwords8k",
                           with_info=True,
                           as_supervised=True,
                           data_dir=data_dir)
    _train_data, _val_data = imdb['train'], imdb['test']

    _train_data = _train_data.shuffle(buffer_size)
    _train_data = _train_data.padded_batch(batch_size, _train_data.output_shapes)

    _val_data = _val_data.padded_batch(batch_size, _val_data.output_shapes)

    _tokenizer = info.features['text'].encoder

    return _train_data, _val_data, _tokenizer


def build_model(_tokenizer):
    _model = keras.Sequential([
        keras.layers.Embedding(input_dim=_tokenizer.vocab_size, output_dim=embedding_dim, input_length=max_len),
        # keras.layers.Flatten(), Flatten may not work
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.summary()
    return _model


def build_lstm_model(_tokenizer):
    _model = keras.Sequential([
        keras.layers.Embedding(input_dim=_tokenizer.vocab_size, output_dim=embedding_dim, input_length=max_len),
        # keras.layers.Flatten(), Flatten may not work
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.summary()
    return _model


def build_gru_model(_tokenizer):
    _model = keras.Sequential([
        keras.layers.Embedding(input_dim=_tokenizer.vocab_size, output_dim=embedding_dim, input_length=max_len),
        # keras.layers.Flatten(), Flatten may not work
        keras.layers.Conv1D(128, 5, activation='relu'),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.summary()
    return _model


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()


def download_embeddings(_model, _reverse_word_index, _tokenizer):
    e = _model.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape)  # shape: (_tokenizer.vocab_size, embedding_dim)

    out_meta = io.open(output_path + 'meta.tsv', 'w', encoding='utf-8')
    out_vecs = io.open(output_path + 'vecs.tsv', 'w', encoding='utf-8')

    for word_num in range(1, _tokenizer.vocab_size):
        word = _reverse_word_index[word_num]
        embeddings = weights[word_num]

        out_meta.write(word + "\n")
        out_vecs.write("\t".join([str(x) for x in embeddings]) + "\n")

    out_meta.close()
    out_vecs.close()


if __name__ == '__main__':
    train_data, val_data, tokenizer = get_data()
    model = build_model(tokenizer)
    history = model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
