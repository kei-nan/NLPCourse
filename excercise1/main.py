import nltk
import urllib.request
import argparse
import logging


logging.basicConfig()
logger = logging.getLogger('exc1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url',
                        default='http://www.gutenberg.org/files/84/84-0.txt',
                        help='Where to download the text from')
    args = parser.parse_args()
    page = urllib.request.urlopen(args.url)
    content = page.read()
    logger.warning(content)


if __name__ == '__main__':
    try:
        logger.debug('Hello')
        main()
    finally:
        logger.debug('Bye bye...')