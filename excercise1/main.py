import nltk
import string
import urllib.request
import argparse
import logging
from collections import Counter
from math import log2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('exc1')


def estimate_cross_entropy(q):
    q = [i for i in q if i > 0]
    length = len(q)
    return -sum((1 / length) * log2(q[i]) for i in range(length))


def _create_letter_pairs_counter(clean_text_list):
    clean_text_str = ''.join(clean_text_list)
    list_of_letter_pairs = [clean_text_str[i:i + 2] for i in range(len(clean_text_str) - 1)]
    return Counter(list_of_letter_pairs)


def _calc_conditional_letter_probability(letter_pairs_counter, letter_probability, letters_frequency):
    ans = []
    alphabet = list(string.ascii_lowercase + ' ')
    second_letter: string
    for second_letter in alphabet:
        ans += ([letter_pairs_counter[first_letter + second_letter] / letters_frequency[second_letter]
                 for first_letter in alphabet])
    return ans


def calc_cross_entropy_from_probability(clean_text_list, letter_probability, letters_frequency):
    letter_pairs_counter =\
        _create_letter_pairs_counter(clean_text_list)
    return estimate_cross_entropy(
        _calc_conditional_letter_probability(letter_pairs_counter, letter_probability, letters_frequency))


def remove_header_and_footer(lines, start_barrier='*** START OF THIS PROJECT', end_barrier='*** END OF THIS PROJECT'):
    header_and_footer_positions = []
    start = 0
    end = len(lines)
    for pos, line in enumerate(lines):
        if line.startswith(start_barrier):
            start = pos
        elif line.startswith(end_barrier):
            end = pos
        else:
            continue
    logger.info('Header: {}, Footer: {}'.format(start, end))
    lines = lines[start + 1:end]
    return lines


# http://www.gutenberg.org/files/60773/60773-0.txt
def find_chapter_words(lines,
                       minimal_word_length=4,
                       appearance_threshold=2,
                       words_in_sentance=4):
    class WordInfo:
        def __init__(self):
            self.count = 0
            self.positions = []

        def __repr__(self):
            return repr(self.positions)

    word_to_info = {}
    chapter_punctuation = string.punctuation
    for line_number, line in enumerate(lines):
        words = line.split()
        if not words or len(words) < 1:
            continue
        # First word should look like Chapter or CHAPTER or chapter.
        # We want to avoid connecting words such as OF and THE
        first_word = words[0].strip(chapter_punctuation)
        if len(first_word) < minimal_word_length:
            continue
        # last word should be the chapter id, e.g: 1, 11, X, I, A
        # so it is either a number or a capital text
        last_word = words[-1].strip(chapter_punctuation)
        # Should mostly be in the format of "Chapter _" so we expect at most two words
        is_last_word_chapter_index = last_word.isdigit() or last_word.isupper()
        is_chapter_candidate = first_word.isalpha() and is_last_word_chapter_index
        is_sentance = len(words) >= words_in_sentance
        if not is_chapter_candidate or is_sentance:
            continue
        logger.debug(f'chapter candidate: {words}')
        if first_word not in word_to_info:
            word_to_info[first_word] = WordInfo()
        info = word_to_info[first_word]
        info.count += 1
        info.positions.append(line_number)
    meta_content_words = {k: v for (k, v) in word_to_info.items()
                          if v.count >= appearance_threshold
                          and (k.istitle() or k.isupper())}
    return meta_content_words.keys()


def remove_chapters(lines, chapter_expected_apperances=2):
    class LineInfo:
        def __init__(self):
            self.count = 0
            self.positions = []

        def __repr__(self):
            return repr(self.positions)

    chapter_lines = {}
    chapter_words = find_chapter_words(lines)
    print('Chapter Words: {}'.format(chapter_words))
    for line_number, line in enumerate(lines):
        stripped_line = line.strip()
        for word in chapter_words:
            if stripped_line.startswith(word):
                if line not in chapter_lines:
                    chapter_lines[stripped_line] = LineInfo()
                chapter_line = chapter_lines[stripped_line]
                chapter_line.count += 1
                chapter_line.positions.append(line_number)
                break
    chapter_lines = {k: v for (k, v) in chapter_lines.items() if v.count == chapter_expected_apperances}
    largest_apperances = []
    for line_info in chapter_lines.values():
        # we iterate line by line so positions are already sorted
        largest_position = line_info.positions[1]
        largest_apperances.append(largest_position)
    # sorting in reverse so we can remove lines without affecting next line deletion
    largest_apperances = sorted(largest_apperances, reverse=True)
    first_chapter_second_apperance = largest_apperances[-1]
    last_chapter_second_apperance = largest_apperances[0]
    lines = lines[:last_chapter_second_apperance]
    for line_pos in largest_apperances[1:]:
        del lines[line_pos]
    lines = lines[first_chapter_second_apperance:]
    return lines


def tokenize_lines(lines, keep_non_english_letters):
    from nltk.corpus import stopwords
    from nltk.tokenize import RegexpTokenizer

    # can be a string containing spaces with a punctuation inside
    def clean_space(token):
        return ' ' * token.count(' ')

    def clean_char(character):
        if not keep_non_english_letters and character not in string.ascii_lowercase:
            return ''
        else:
            return character

    # Tokenize text
    blacklisted_words = set(stopwords.words('english'))
    pattern = string.punctuation + r'\s'
    tokenizer = RegexpTokenizer(r'[{}]+'.format(pattern), gaps=True)
    tokens = []
    space_list = []
    for line in lines:
        prev_end = None
        words_span_in_line = tokenizer.span_tokenize(text=line)
        for span in words_span_in_line:
            start, end = span
            if prev_end is not None:
                space_token = line[prev_end:start]
                space_token = clean_space(space_token)
                if space_token:
                    tokens.append(space_token)
                    space_list.append(space_token)
            prev_end = end
            text_token = line[start: end].lower()
            text_token = ''.join([clean_char(c) for c in text_token])
            if not text_token or text_token in blacklisted_words:
                continue
            tokens.append(text_token)
    logger.debug('Spaces: {}'.format(''.join(space_list).strip()))
    return tokens


# Cleansup the text by:
# 1) Moving to lowercase
# 2) Removes punctuation
# 3) Tokenizes text
# 4) Removes header and chapter keywords
def cleanup_text(text, keep_non_english_letters=False):
    lines = text.splitlines()
    lines = remove_header_and_footer(lines)
    lines = remove_chapters(lines)
    tokens = tokenize_lines(lines, keep_non_english_letters)
    return tokens


def plot_word_frequencies(cleaned_content):
    import matplotlib.pyplot as plt
    import numpy as np

    english_content = [token for token in cleaned_content if not token.isspace()]
    word_to_frequency = nltk.FreqDist(english_content)
    word_count = sum(word_to_frequency.values())
    word_to_probability = {k: (v / word_count) for (k, v) in word_to_frequency.items()}
    frequent_words = sorted(word_to_frequency.keys(), key=lambda k: word_to_frequency[k], reverse=True)
    print('Top Frequent Word: {}'.format(frequent_words[0]))
    top_frequent_words = {k: v for (k, v) in word_to_frequency.items() if frequent_words.index(k) < 5}
    print('Top Frequent Words: {}'.format(top_frequent_words))

    probabilities = sorted(word_to_probability.values())
    log_prob = [np.log(p) for p in probabilities]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    plt.plot(log_prob, 'r')
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

    letters_frequency = nltk.FreqDist(''.join(cleaned_content))
    letters_frequency_sum = sum(letters_frequency.values())
    letter_probability = {k: v / letters_frequency_sum for (k, v) in letters_frequency.items()}

    conditional_letter_probability = calc_cross_entropy_from_probability(cleaned_content, letter_probability, letters_frequency)

    letters_probability_sum = sum(letter_probability.values())

    print(letters_frequency.keys())
    # not sure about the entropy
    prob = nltk.MLEProbDist(freqdist=letters_frequency)
    print(f'Letter Frequency: {repr(letters_frequency)}')
    print(f'Token Count: {len(cleaned_content)}')
    print(f'Word Type Count: {len(set(cleaned_content))}')
    print(f'Entropy: {nltk.entropy(prob)}')
    print(f'Entropy from conditional letter probability: {conditional_letter_probability}')
    plot_word_frequencies(cleaned_content)


if __name__ == '__main__':
    try:
        logger.debug('Hello')
        nltk.download('stopwords')
        nltk.download('punkt')
        main()
    finally:
        logger.debug('Bye bye...')
