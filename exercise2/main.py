import nltk
import os
import io
import requests
import argparse
import logging
import zipfile
from nltk.lm import MLE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('exc2')


def cleanup_text(text, keep_non_english_letters=False, keep_spaces=False):
    from utils.preprocessing import tokenize_lines
    text = text.lower()
    lines = text.splitlines()
    tokens = tokenize_lines(lines, keep_non_english_letters, keep_spaces)
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--word-list-url',
                        default='http://www2.mta.ac.il/~gideon/courses/nlp/data/word_list_20k.txt')
    parser.add_argument('-c', '--corpora-zip-url',
                        default='http://www2.mta.ac.il/~gideon/courses/nlp/data/corpus_combined.zip')
    parser.add_argument('-t', '--training-corpus',
                        default='the_life_and_adventures_of_nicholas_nickleby.txt')
    parser.add_argument('-e', '--testing-corpus',
                        default='persuasion.txt')
    args = parser.parse_args()
    word_list = requests.get(args.word_list_url).content.decode('utf-8').split('\n')
    if not os.path.exists(args.training_corpus) or not os.path.exists(args.testing_corpus):
        with requests.get(args.corpora_zip_url, stream=True) as response:
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extract(args.training_corpus)
            z.extract(args.testing_corpus)
    with open(args.training_corpus, 'r') as file:
        training = file.read()
        clean_training = cleanup_text(training)
    with open(args.testing_corpus, 'r') as file:
        testing = file.read()
        clean_testing = cleanup_text(testing)
    for language_model_type in [MLE]:
        for ngram in range(2, 3):
            vocab = list(nltk.everygrams(sequence=word_list, min_len=ngram, max_len=ngram))
            train_data = list(nltk.ngrams(sequence=clean_training, n=ngram, pad_right=True))
            model = language_model_type(order=ngram, vocabulary=nltk.lm.Vocabulary(vocab))
            model.fit(text=train_data)
            cross_entropy = model.entropy(clean_testing)
            logger.info('Cross Entropy for N={}: {}'.format(ngram, cross_entropy))


if __name__ == '__main__':
    try:
        logger.debug('Exercise 2')
        nltk.download('stopwords')
        nltk.download('punkt')
        main()
    finally:
        logger.debug('Bye bye...')
