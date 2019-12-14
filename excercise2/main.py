import nltk
import string
import os
import io
import requests
import argparse
import logging
import shutil
import zipfile
from collections import Counter
from math import log2
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('exc2')


def clean_file_content(content):
    return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--word-list-url',
                        default='www2.mta.ac.il/~gideon/courses/nlp/data/word_list_20k.txt')
    parser.add_argument('-c', '--corpora-zip-url',
                        default='http://www2.mta.ac.il/~gideon/courses/nlp/data/corpus_combined.zip')
    parser.add_argument('-t', '--training-corpus',
                        default='the_life_and_adventures_of_nicholas_nickleby.txt')
    parser.add_argument('-e', '--testing-corpus',
                        default='persuasion.txt')
    args = parser.parse_args()
    word_list = requests.get(args.word_list_url).split('\n')
    with requests.get(args.corpora_zip_url, stream=True) as response:
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extract(args.training_corpus)
        z.extract(args.testing_corpus)
    with open(args.training_corpus, 'r') as file:
        training = file.read()
        clean_training = clean_file_content(training)
    with open(args.testing_corpus, 'r') as file:
        testing = file.read()
        clean_testing = clean_file_content(testing)
    for language_model_type in [MLE]:
        for ngram in range(2):
            train, _ = padded_everygram_pipeline(ngram, clean_training)
            _, vocab = padded_everygram_pipeline(ngram, word_list)
            model = language_model_type(vocabulary=vocab, order=ngram)
            cross_entropy = model.entropy(clean_testing)
            logger.info('Cross Entropy for N={}: {}'.format(ngram, cross_entropy))


