from tensorflow.python.keras.api._v1.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.api._v1.keras.preprocessing.sequence import pad_sequences
import tensorflow.python.keras.api._v1.keras as keras
import matplotlib.pyplot as plt

file_path = '../data/poetry/irish.txt'


def get_data():
    data = open(file=file_path, mode='r').read()
    _corpus = data.lower().split("\n")
    print('Corpus length: ', len(_corpus))
    return _corpus


def get_ngram_padded_sequence(_corpus, _tokenizer):
    _max_len = 0
    input_sequence: list = []

    _tokenizer.fit_on_texts(_corpus)
    _total_words = len(_tokenizer.word_index) + 1

    print('Tokenizer word index: ', _tokenizer.word_index)
    print('Total words: ', _total_words)

    for text in _corpus:
        tokens = _tokenizer.texts_to_sequences(texts=[text])[0]

        if len(tokens) > _max_len:
            _max_len = len(tokens)

        for i in range(1, len(tokens)):
            input_sequence.append(tokens[:i + 1])

    return pad_sequences(sequences=input_sequence, maxlen=_max_len, padding='pre'), _total_words, _max_len


def build_model(_total_words, _max_len):
    _model = keras.Sequential()
    _model.add(keras.layers.Embedding(input_dim=_total_words, output_dim=100, input_length=_max_len - 1))
    _model.add(keras.layers.Bidirectional(keras.layers.LSTM(150)))
    _model.add(keras.layers.Dense(units=_total_words, activation='softmax'))
    print(_model.summary())
    return _model


def build_model_v2(_total_words, _max_len):
    _model = keras.Sequential()
    _model.add(keras.layers.Embedding(input_dim=_total_words, output_dim=100, input_length=_max_len - 1))
    _model.add(keras.layers.Bidirectional(keras.layers.LSTM(150)))
    _model.add(keras.layers.LSTM(150))
    _model.add(keras.layers.Dense(_total_words / 2, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    _model.add(keras.layers.Dense(_total_words, activation='softmax'))
    _model.add(keras.layers.Dense(units=_total_words, activation='softmax'))
    print(_model.summary())
    return _model


def plot_graphs(_history, _metric):
    plt.plot(_history.history[_metric])
    plt.xlabel("Epochs")
    plt.ylabel(_metric)
    plt.show()


def predict_text(_model, _tokenizer, _max_len):
    seed_text = "I wanna see"
    next_words = 100

    for i in range(next_words):
        _tokens = _tokenizer.texts_to_sequences([seed_text])[0]
        _tokens = pad_sequences([_tokens], maxlen=_max_len - 1, padding='pre')
        _predicted = _model.predict_classes(_tokens, verbose=0)
        predicted_word = ""

        for word, index in tokenizer.word_index.items():
            if index == _predicted:
                predicted_word = word
                break
        seed_text += " " + predicted_word

    print(seed_text)


if __name__ == '__main__':
    # Get irish poems
    corpus = get_data()

    tokenizer = Tokenizer()

    # Get ngram padded sequence
    padded_sequence, total_words, max_len = get_ngram_padded_sequence(corpus, tokenizer)
    # padded_sequence = np.array(padded_sequence)

    # Input and labels
    X, labels = padded_sequence[:, :-1], padded_sequence[:, -1]

    # Convert labels to one hot encoding form
    y = keras.utils.to_categorical(labels, num_classes=total_words)

    model: keras.Sequential = build_model(total_words, max_len)
    adam = keras.optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # epochs = 500
    history = model.fit(x=X, y=y, epochs=10, verbose=1)
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')

    predict_text(model, tokenizer, max_len)
