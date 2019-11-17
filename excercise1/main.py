import nltk
import string
import urllib.request
import argparse
import logging


logging.basicConfig()
logger = logging.getLogger('exc1')


def cleanup_text(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    # Move to lowercase
    text_in_lowercase = text.lower()

    # Remove punctuation
    def replace_punctuation_with_space(c):
        return ' ' if c in string.punctuation else c
    no_punctuation_lowercase_text = ''.join([replace_punctuation_with_space(c) for c in text_in_lowercase])

    # Tokenize text
    blacklisted_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(no_punctuation_lowercase_text)
    filtered_lowercase_text = [w for w in word_tokens if w not in blacklisted_words]

    # Remove non content
    chapter_pos = filtered_lowercase_text.rfind('chapter')
    chapter_number = int(filtered_lowercase_text[chapter_pos + 1])
    for iteration in range(chapter_number):
        pos = filtered_lowercase_text.find('chapter')
        if pos == -1 or filtered_lowercase_text[pos + 1] != str(iteration):
            raise Exception(f'Could not locate chapter {iteration} in headline')
        next_pos = filtered_lowercase_text.find('chapter', pos)
        if pos == -1 or filtered_lowercase_text[pos + 1] != str(iteration):
            raise Exception(f'Could not locate chapter {iteration} in content')
        filtered_lowercase_text = filtered_lowercase_text[:pos] + filtered_lowercase_text[pos+2:]
        filtered_lowercase_text = filtered_lowercase_text[:next_pos] + filtered_lowercase_text[next_pos+2:]
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
    print(cleaned_content[:100])


if __name__ == '__main__':
    try:
        logger.debug('Hello')
        nltk.download('stopwords')
        nltk.download('punkt')
        main()
    finally:
        logger.debug('Bye bye...')