import numpy as np
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

def TFIDF(sentences, labels, seed_words):
    tfidf_sentences = getTFIDF(sentences, seed_words)
    preds = getPrediction(tfidf_sentences, seed_words)
    macro_f1 = getF1(preds, labels, 'macro')
    micro_f1 = getF1(preds, labels, 'micro')
    return macro_f1, micro_f1

def getTFIDF(sentences, seed_words):
    ## Get all seed words for Vectorizer
    all_seed_words = []
    for cat in seed_words.values():
        for word in cat:
            all_seed_words.append(word)
    vectorizer = TfidfVectorizer(vocabulary=all_seed_words)
    return vectorizer.fit_transform(sentences).toarray()

def getPrediction(tfidf_sentences, seed_words):
    preds = []
    for doc in tfidf_sentences:
        tfidf_vals = {}
        idx = 0
        for cat in seed_words:
            for word in seed_words[cat]:
                try:
                    tfidf_vals[cat] += doc[idx]
                except:
                    tfidf_vals[cat] = doc[idx]
                idx += 1
        preds.append(max(tfidf_vals, key = tfidf_vals.get))
    return preds

def getF1(preds, obs, type):
    if type not in ['macro', 'micro']:
        raise Exception('No valid type of f1 test specified, use macro or micro.')
    return f1_score(preds, obs, average=type)

