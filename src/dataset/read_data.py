import pickle
import json
import sys

def read_data(data_dir, seed_words_dir):
    data_dir = sys.path[0] + data_dir
    seed_words_dir = sys.path[0] + seed_words_dir
    data = pickle.load(open(data_dir, 'rb'))
    with open(seed_words_dir) as fp:
        seed_words = json.load(fp)
    sentences = data.sentence.values
    labels = data.label.values
    return sentences, labels, seed_words
