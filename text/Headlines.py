import json
import re
from tensorflow.python.keras.api._v1.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.api._v1.keras.preprocessing.sequence import pad_sequences

# !wget --no-check-certificate \
#     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv \
#             -O /tmp/bbc-text.csv

file_path = '../data/headlines/Sarcasm_Headlines_Dataset_v2.json'


def get_stop_words():
    return ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
            "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
            "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
            "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
            "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
            "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
            "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so",
            "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
            "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
            "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what",
            "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
            "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]


def get_data(_file_path: str):
    stop_words = get_stop_words()
    with open(file_path) as json_file:
        data = json.load(json_file)

    _corpus, _result = [], []
    for text in data:
        text_clean = [word for word in re.split('\W+', text['headline'])
                      if word.lower() not in stop_words and len(word) > 2]
        _corpus.append(' '.join(text_clean))
        _result.append(text['is_sarcastic'])
    # _corpus, _result = map(list, zip(
    #     *[(''.join(word), text['is_sarcastic']) for text in data for word in re.split('\W+', text['headline'])
    #       if word.lower() not in stop_words and len(word) > 2]))

    print("Corpus: {}, Result: {}".format(len(_corpus), len(_result)))
    return _corpus, _result


# corpus,result=map(list,zip(*[(text['headline'],text['is_sarcastic'])for text in data]))

if __name__ == '__main__':
    corpus, result = get_data(file_path)
    tokens = Tokenizer(oov_token='oov')
    tokens.fit_on_texts(corpus)
    print(tokens.num_words, tokens.word_index)

    sequence = tokens.texts_to_sequences(corpus)
    padded_sequence = pad_sequences(sequence, padding='post')
    print("Padded Data: ", padded_sequence.shape)
