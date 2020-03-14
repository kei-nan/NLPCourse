import nltk
import os
import argparse
import logging
import random

from exercise3.classifiers import OneNearestNeighbor, NaiveBayes, Maxent
from exercise3.document import Document
from exercise3.corpus import Corpus
from exercise3.score_strategy import TfIdfStrategy, BinaryStrategy
from typing import List, Dict, Tuple
from utils.preprocessing import SentanceTokenizer

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='bs4', message='.*looks like a*')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('exc3')


def lines_from_file(path) -> List[str]:
    with open(path, 'r', encoding='utf-8') as file:
        return file.readlines()


def documents_from_lines(lines: List[str], tokenizer: SentanceTokenizer) -> List[Document]:
    documents = []
    for line in lines:
        documents.append(Document.from_raw_line(line, tokenizer))
    return documents


def run_once(train_lines: List[str],
             classify_lines: List[str],
             categories: List[str],
             split_factor: int,
             tokenizer_settings: Dict[str, bool]) -> Dict[str, List[str]]:
    train_lines_for_run = train_lines
    if classify_lines is None:
        # random.shuffle(train_lines)
        split_marker = int(len(train_lines) * split_factor)
        classify_lines = train_lines[split_marker:]
        train_lines_for_run = train_lines[:split_marker]

    tokenizer = SentanceTokenizer(**tokenizer_settings)

    train_documents = documents_from_lines(train_lines_for_run, tokenizer)
    classify_documents = documents_from_lines(classify_lines, tokenizer)

    results = {}
    corpus = Corpus(train_documents, categories)
    corpus.print_statisitcs_as_csv()
    first = OneNearestNeighbor(score_strategy=TfIdfStrategy(corpus=corpus))
    second = NaiveBayes(corpus=corpus)
    # third = OneNearestNeighbor(score_strategy=BinaryStrategy(corpus=corpus))
    # fourth = Maxent(corpus=corpus)
    for classifier in [first, second]:
        classifier_results = classifier.classify_all(classify_documents)
        results[classifier.name] = classifier_results
        classifier.show_most_informative_features(10)
    return classify_documents, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        default=os.path.join(os.curdir, 'train_data.txt'))
    parser.add_argument('--categories',
                        default=os.path.join(os.curdir, 'categories.txt'))
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--use-train-other-half',
                            action='store_true',
                            default=False)
    test_group.add_argument('--classify',
                            default=os.path.join(os.curdir, 'train_data.txt'))
    args = parser.parse_args()

    with open(args.categories, 'r') as categories_file:
        categories = [category.strip() for category in categories_file.readlines()]
    train_lines = lines_from_file(args.train)
    classify_lines = None if args.use_train_other_half else lines_from_file(args.classify)
    tokenizer_settings = {"keep_non_english_letters": False,
                          "keep_numbers": False,
                          "keep_spaces": False,
                          "stemming": False}

    split_factor = 0.8
    split_factor_text = f', split factor: {split_factor}' if split_factor else ''
    # max_mask = 2**len(tokenizer_settings)
    max_mask = 2
    for settings_mask in range(1, max_mask):
        settings = {}
        for index, key in enumerate(tokenizer_settings.keys()):
            bit = (1 << index)
            settings[key] = (bit & settings_mask) == bit
        print(f'Settings: {settings}{split_factor_text}')
        results = {}
        iteration_count = 1 if classify_lines is None else 1
        for iteration in range(iteration_count):
            print(f'{iteration+1}/{iteration_count}', end='\r')
            classify_documents, results = run_once(train_lines=train_lines,
                                                   classify_lines=classify_lines,
                                                   categories=categories,
                                                   split_factor=split_factor,
                                                   tokenizer_settings=settings)
            classifier_to_accuracy = {}
            for classifier_name, classifier_results in results.items():
                success = 0
                content = ''
                for index, classified in enumerate(classifier_results):
                    content += classified + '\n'
                    expected = classify_documents[index].category
                    if expected is not None and expected == classified:
                        success += 1
                with open(f'{classifier_name}-{iteration}.txt', 'w') as output:
                    output.write(content)
                classifier_to_accuracy[classifier_name] = success / len(classifier_results) * 100
            print(f'Wrote iteration {iteration} results, accuracy: {classifier_to_accuracy}')
        #     if iteration == 0:
        #         results = result
        #     else:
        #         for key, value in result.items():
        #             results[key] += value
        # for key, value_sum in results.items():
        #     avg_value = value_sum / iteration_count
        #     print(f'{key} Average Accuracy: {avg_value}')


if __name__ == '__main__':
    try:
        logger.debug('Exercise 3')
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('words')
        main()
    finally:
        logger.debug('Bye bye...')
