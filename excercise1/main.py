import nltk
import string
import urllib.request
import argparse
import logging


logging.basicConfig()
logger = logging.getLogger('exc1')


# Cleansup the text by:
# 1) Moving to lowercase
# 2) Removes punctuation
# 3) Tokenizes text
# 4) Removes header and chapter keywords
def cleanup_text(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    # Move to lowercase
    text_in_lowercase = text.lower()

    # Remove punctuation
    def remove_puncutation(c):
        return '' if c in string.punctuation else c

    # Tokenize text
    blacklisted_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text_in_lowercase)

    first_chapter = None
    is_chapter = False
    filtered_lowercase_text = []
    chapters = {}
    for w in word_tokens:
        w = ''.join([remove_puncutation(c) for c in w])
        if not w or w in blacklisted_words:
            continue
        elif w == 'chapter':
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
                raise Exception(f'Chapter {w} was encountered more than twice')
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
    word_frequency = nltk.FreqDist(cleaned_content)
    print(f'Letter Frequency: {repr(letters_frequency)}')
    print(f'Word Frequency: {repr(word_frequency)}')
    print(f'Token Count: {len(cleaned_content)}')
    print(f'Word Type Count: {len(set(cleaned_content))}')


if __name__ == '__main__':
    try:
        logger.debug('Hello')
        nltk.download('stopwords')
        nltk.download('punkt')
        main()
    finally:
        logger.debug('Bye bye...')