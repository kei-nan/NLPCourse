import nltk
import string
import urllib.request
import argparse
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('exc1')


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
    lines = lines[start+1:end]
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
    meta_content_words = {k:v for (k,v) in word_to_info.items()
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
    chapter_lines = {k: v for (k,v) in chapter_lines.items() if v.count == chapter_expected_apperances}
    largest_apperances = []
    for line_info in chapter_lines.values():
        # we iterate line by line so positions are already sorted
        largest_position = line_info.positions[1]
        largest_apperances.append(largest_position)
    # sorting in reverse so we can remove lines without affecting next line deletion
    largest_apperances = sorted(largest_apperances, reverse=True)
    print(largest_apperances)
    first_chapter_second_apperance = largest_apperances[-1]
    last_chapter_second_apperance = largest_apperances[0]
    lines = lines[:last_chapter_second_apperance]
    for line_pos in largest_apperances[1:]:
        del lines[line_pos]
    lines = lines[first_chapter_second_apperance:]
    return lines


# Cleansup the text by:
# 1) Moving to lowercase
# 2) Removes punctuation
# 3) Tokenizes text
# 4) Removes header and chapter keywords
def cleanup_text(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    # Move to lowercase
    lines = text.splitlines()
    print(lines)
    lines = remove_header_and_footer(lines)
    print(lines)
    lines = remove_chapters(lines)

    text_in_lowercase = text.lower()
    # Remove punctuation
    def remove_puncutation(c):
        return '' if c in string.punctuation else c

    # Tokenize text
    blacklisted_words = set(stopwords.words('english'))
    # tokenize = RegexpTokenizer('\w+')
    word_tokens = word_tokenize(text_in_lowercase)

    first_chapter = None
    is_chapter = False
    filtered_lowercase_text = []
    chapters = {}
    for w in word_tokens:
        w = ''.join([remove_puncutation(c) for c in w])
        if not w or w in blacklisted_words:
            continue
        elif w.startswith('chapter'):
            if first_chapter is None:
                first_chapter = len(filtered_lowercase_text)
            is_chapter = True
        elif is_chapter:
            # We expect each chapter to appear at most twice, once in headline and another in text
            if w not in chapters:
                chapters[w] = True
            elif chapters[w] is True:
                chapters[w] = False
            else:
                raise Exception(f'Chapter \'{w}]\' was encountered more than twice, encountered first chapter: {first_chapter}')
            is_chapter = False
        else:
            filtered_lowercase_text.append(w)
    if first_chapter is not None:
        filtered_lowercase_text = filtered_lowercase_text[first_chapter:]
    return filtered_lowercase_text


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
    letters_probability_sum = sum(letter_probability.values())
    print(letters_frequency.keys())


    word_frequency = nltk.FreqDist(cleaned_content)
    # not sure about the entropy
    prob = nltk.MLEProbDist(freqdist=letters_frequency)
    print(f'Letter Frequency: {repr(letters_frequency)}')
    print(f'Word Frequency: {repr(word_frequency)}')
    print(f'Token Count: {len(cleaned_content)}')
    print(f'Word Type Count: {len(set(cleaned_content))}')
    print(f'Entropy: {nltk.entropy(prob)}')


if __name__ == '__main__':
    try:
        logger.debug('Hello')
        nltk.download('stopwords')
        nltk.download('punkt')
        main()
    finally:
        logger.debug('Bye bye...')