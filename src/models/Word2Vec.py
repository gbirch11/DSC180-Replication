from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec
from nltk.corpus import stopwords
from scipy import spatial
from sklearn.metrics import f1_score
import string
from copy import copy

def Word2Vec(sentences, labels, seed_words, dataset):
    tokenized_sentences = tokenize(sentences, dataset)
    model = word2vec.Word2Vec(tokenized_sentences, epochs=25, workers = 24, min_count = 5, window = 5, sample=1e-3)
    seed_weights = get_seed_weights(seed_words, model)
    doc_weights = get_doc_weights(tokenized_sentences, model)
    preds = getPrediction(doc_weights, seed_weights)
    macro_f1 = getF1(preds, labels, 'macro')
    micro_f1 = getF1(preds, labels, 'micro')
    return macro_f1, micro_f1

def tokenize(sentences, dataset):
    words = []
    punctuation = string.punctuation
    tokenizer = TfidfVectorizer().build_tokenizer()
    stop_words = set(stopwords.words('english'))

    for sentence in sentences:
        sentence = sentence.lower()
        if dataset == '20news':
            sentence = [i for i in sentence if not (i in punctuation)]
            sentence = ''.join(sentence)
        currWords = tokenizer(sentence)
        currWords = [w for w in currWords if w not in stop_words]
        words.append(currWords)
    return words

def get_seed_weights(seed_words, model):
    seed_weights = {}
    for seed_cat in seed_words:
        for word in seed_words[seed_cat]:
            if word in model.wv:
                try:
                    seed_weights[seed_cat] += model.wv[word]
                except:
                    seed_weights[seed_cat] = model.wv[word]
        seed_weights[seed_cat] = seed_weights[seed_cat] / len(seed_words[seed_cat])
    return seed_weights

def get_doc_weights(tokenized_sentences, model):
    doc_weights = []
    for sentence in tokenized_sentences:
        cnt = 0
        doc_w = None
        for word in sentence:
            if word in model.wv:
                cnt += 1
                if doc_w is None:
                    doc_w = copy(model.wv[word])
                else:
                    doc_w += model.wv[word]
        doc_weights.append(doc_w / cnt)
    return doc_weights

def getPrediction(doc_weights, seed_weights):
    preds = []
    for doc, val in enumerate(doc_weights):
        sim = -10
        pred = 'N/A'
        for seed, seed_val in seed_weights.items():
            currSim = 1 - spatial.distance.cosine(val, seed_val)
            if currSim > sim:
                sim = currSim
                pred = seed
        preds.append(pred)
    return preds

def getF1(preds, obs, type):
    if type not in ['macro', 'micro']:
        raise Exception('No valid type of f1 test specified, use macro or micro.')
    return f1_score(preds, obs, average=type)
