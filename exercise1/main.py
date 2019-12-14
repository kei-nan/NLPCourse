import nltk
import string
import urllib.request
import argparse
import logging
from collections import Counter
from math import log2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('exc1')


# Cleansup the text by:
# 1) Moving to lowercase
# 2) Removes punctuation
# 3) Tokenizes text
# 4) Removes header and chapter keywords
def cleanup_text(text, keep_non_english_letters=False):
    from utils.preprocessing import remove_header_and_footer, remove_chapters, tokenize_lines
    lines = text.splitlines()
    lines = remove_header_and_footer(lines)
    lines = remove_chapters(lines)
    tokens = tokenize_lines(lines, keep_non_english_letters)
    return tokens


def print_conditional_letter_probability(letter_pairs_counter, letters_frequency):
    ans = []
    alphabet = list(string.ascii_lowercase + ' ')
    second_letter: string
    print('  '.join(alphabet))
    for second_letter in alphabet:
        text = f'P({second_letter}|_)'
        for first_letter in alphabet:
            prob = letter_pairs_counter[(first_letter, second_letter)] / letters_frequency[second_letter]
            text += '{0:.3f}, '.format(prob)
        print(text)
    print("conditional_letter_probability" + str(ans))


def plot_word_frequencies(cleaned_content):
    import matplotlib.pyplot as plt

    english_content = [token for token in cleaned_content if not token.isspace()]
    word_to_frequency = nltk.FreqDist(english_content)
    frequent_words = sorted(word_to_frequency.keys(), key=lambda k: word_to_frequency[k], reverse=True)
    logger.debug('Top Frequent Word: {}'.format(frequent_words[0]))
    top_frequent_words = {k: v for (k, v) in word_to_frequency.items() if frequent_words.index(k) < 5}
    logger.debug('Top Frequent Words: {}'.format(top_frequent_words))

    log_frequencies = []
    log_ranks = []
    for rank, word in enumerate(frequent_words):
        frequency = word_to_frequency[word]
        log_freq = log2(frequency)
        log_frequencies.append(log_freq)
        log_rank = log2(rank + 1)
        log_ranks.append(log_rank)

    # plot the function
    plt.plot(log_ranks, log_frequencies)
    plt.xlabel('ranks')
    plt.ylabel('frequency')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--text',
                              default=None,
                              help='Text')
    source_group.add_argument('--file',
                              default=None,
                              help='File path to read the text from')
    source_group.add_argument('--url',
                              help='Where to download the text from, e.g: http://www.gutenberg.org/files/84/84-0.txt')
    args = parser.parse_args()
    if args.url:
        page = urllib.request.urlopen(args.url)
        content = page.read().decode('utf-8')
    elif args.file:
        with open(args.file, 'r') as f:
            content = f.read()
    elif args.text:
        content = args.text
    cleaned_content = cleanup_text(text=content)

    from nltk.lm import MLE
    from nltk.util import bigrams
    from nltk.lm.preprocessing import padded_everygram_pipeline

    letter_pairs = []
    for token in cleaned_content:
        token_bigrams = list(bigrams(token))
        if len(token_bigrams) > 0:
            letter_pairs += token_bigrams
    n = 2
    train, vocab = padded_everygram_pipeline(n, cleaned_content)
    language_model = MLE(order=n)
    language_model.fit(text=train, vocabulary_text=vocab)
    bigram_cross_entropy = language_model.entropy(letter_pairs)
    print(f'Cross Entropy(bigram): {bigram_cross_entropy}')

    letters_frequency = nltk.FreqDist(''.join(cleaned_content))
    for c, freq in letters_frequency.items():
        print('Character: {}, Frequency: {}'.format(c, freq))

    unigram_order = 1
    train, vocab = padded_everygram_pipeline(unigram_order, cleaned_content)
    unigram_language_model = MLE(order=unigram_order)
    unigram_language_model.fit(train, vocab)
    unigram_cross_entropy = unigram_language_model.entropy(''.join(cleaned_content))
    print(f'Cross Entropy(unigram): {unigram_cross_entropy}')

    prob = nltk.MLEProbDist(freqdist=letters_frequency)

    pairs_counter = Counter(letter_pairs)
    print_conditional_letter_probability(pairs_counter, letters_frequency)

    print(f'Letter Frequency: {repr(letters_frequency)}')
    print(f'Token Count: {len(cleaned_content)}')
    print(f'Word Type Count: {len(set(cleaned_content))}')
    print(f'Entropy: {nltk.entropy(prob)}')
    plot_word_frequencies(cleaned_content)


if __name__ == '__main__':
    try:
        logger.debug('Hello')
        nltk.download('stopwords')
        nltk.download('punkt')
        main()
    finally:
        logger.debug('Bye bye...')
