#!/usr/bin/env python

import sys
from src.dataset.read_data import *
from src.models.TFIDF import *
from src.models.Word2Vec import *


def main(targets):
    if 'data' in targets:
        ## Check which dataset, make sure its valid
        ## Check which version (coarse or fine), make sure its valid
        if len(targets) != 3:
            sys.exit("Error Occured. Run -> python run.py data [20news, NYT] [coarse, fine]")
        dataset = targets[1]
        grained = targets[2]
        if dataset not in ['20news', 'nyt']:
            sys.exit('No Dataset {} Found. Make sure you are specifying exactly \'20news\' or \'nyt\'.'.format(dataset))
        if grained not in ['fine', 'coarse']:
            sys.exit('No Grainularity {} Found. Only coarse and fine are supported.'.format(grained))
        data_dir = '/data/{}/{}/df.pkl'.format(dataset, grained)
        seeds_dir = '/data/{}/{}/seedwords.json'.format(dataset, grained)
        sentences, labels, seeds = read_data(data_dir, seeds_dir)

        ## Print dataset used
        print('Dataset = {}'.format(dataset.upper()))
        ## TF-IDF
        macro_TFIDF, micro_TFIDF = TFIDF(sentences, labels, seeds)
        print('------------------------------------')
        print('Model: TF-IDF')
        print('{} grained data'.format(grained))
        print('Micro F1 Score = {}'.format(micro_TFIDF))
        print('Macro F1 Score = {}'.format(macro_TFIDF))
        print('------------------------------------')
        print('\n')

        ## Word2Vec
        macro_Word2Vec, micro_Word2Vec = Word2Vec(sentences, labels, seeds, dataset)
        print('------------------------------------')
        print('Model: Word2Vec')
        print('{} grained data'.format(grained))
        print('Micro F1 Score = {}'.format(micro_Word2Vec))
        print('Macro F1 Score = {}'.format(macro_Word2Vec))
        print('------------------------------------')

    elif 'test' in targets:
        data_dir = '/test/testdata/data.pkl'
        seeds_dir = '/test/testdata/seedwords.json'

        print('Dataset = Test')
        sentences, labels, seeds = read_data(data_dir, seeds_dir)
        ## TF-IDF
        macro_TFIDF, micro_TFIDF = TFIDF(sentences, labels, seeds)
        print('------------------------------------')
        print('Model: TF-IDF')
        print('Micro F1 Score = {}'.format(micro_TFIDF))
        print('Macro F1 Score = {}'.format(macro_TFIDF))
        print('------------------------------------')
        print('\n')


        ## Word2Vec
        macro_Word2Vec, micro_Word2Vec = Word2Vec(sentences, labels, seeds, 'test')
        print('------------------------------------')
        print('Model: Word2Vec')
        print('Micro F1 Score = {}'.format(micro_Word2Vec))
        print('Macro F1 Score = {}'.format(macro_Word2Vec))
        print('------------------------------------')


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)