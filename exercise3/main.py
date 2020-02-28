import nltk
import os
import html
import argparse
import logging
import random

from exercise3.classifiers import OneNearestNeighbor
from exercise3.corpus import Corpus
from exercise3.document import Document
from math import log10
from typing import List, Dict
from bs4 import BeautifulSoup
from utils.preprocessing import SentanceTokenizer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4', message='.*looks like a*')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('exc3')


def documents_from_file(path) -> List[Document]:
    with open(path, 'r', encoding='utf-8') as file:
        raw_lines = file.readlines()
        documents = []
        for index, raw_line in enumerate(raw_lines):
            documents.append(Document.from_raw_line(raw_line, index))
    return documents


def create_corpus(args, tokenizer):
    Document.TOKENIZER = tokenizer
    with open(args.categories, 'r') as categories_file:
        categories = [category.strip() for category in categories_file.readlines()]
    train_documents = documents_from_file(args.train)

    if not args.use_train_other_half:
        classify_documents = documents_from_file(args.classify)
    else:
        half_marker = int(len(train_documents) / 2)
        classify_documents = train_documents[half_marker:]
        train_documents = train_documents[:half_marker]

    corpus = Corpus(train_documents, categories)
    nearest_clasifier = OneNearestNeighbor(corpus)
    classify_results = nearest_clasifier.classify_all(classify_documents)
    return classify_results, classify_documents


def check_run_accuracy(classify_results, classify_documents):
    success = 0
    bad_documents = {}
    for index, document in enumerate(classify_documents):
        if document.category and classify_results[index] == document.category:
            success += 1
        else:
            bad_documents[index] = document
    precentage = (success / len(classify_results)) * 100
    return precentage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        default=os.path.join(os.curdir, 'train_data.txt'))
    parser.add_argument('--categories',
                        default=os.path.join(os.curdir, 'categories.txt'))
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--use-train-other-half',
                            action='store_true',
                            default=True)
    test_group.add_argument('--classify',
                            default=os.path.join(os.curdir, 'train_data.txt'))
    args = parser.parse_args()

    first = SentanceTokenizer(keep_non_english_letters=False,
                              keep_numbers=True,
                              keep_spaces=False,
                              stemming=True)
    second = SentanceTokenizer(keep_non_english_letters=False,
                               keep_numbers=False,
                               keep_spaces=False,
                               stemming=True)

    first_results, first_classify = create_corpus(args, first)
    print('First Results: {}'.format(check_run_accuracy(first_results, first_classify)))
    second_results, second_classify = create_corpus(args, second)
    print('Second Results: {}'.format(check_run_accuracy(second_results, second_classify)))
    for index, category in enumerate(first_results):
        other_category = second_results[index]
        if category != other_category:
            print('{}!={}'.format(category, other_category))
            print('Mismatch in line #{}: {}, vs: {}'.format(index, first_classify[index], second_classify[index]))


if __name__ == '__main__':
    try:
        logger.debug('Exercise 3')
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('words')
        main()
    finally:
        logger.debug('Bye bye...')
