import nltk
import os
import html
import argparse
import logging

from math import log10
from typing import List, Dict
from bs4 import BeautifulSoup
from utils.preprocessing import SentanceTokenizer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4', message='.*looks like a*')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('exc3')


tokenizer = SentanceTokenizer(keep_non_english_letters=False,
                              keep_spaces=False)


class LineInfo:
    def __init__(self, number: int, subject: str, content: str, category: str = None):
        self.number = number
        self.subject = subject
        self.category = category
        self.content = content if content else ''
        subject_and_content = self.subject + '\n'+ self.content
        self.subject_and_content = self.__cleanup_content(subject_and_content)
        self.vector = None

    class WordStatistics:
        def __init__(self, line_number, count):
            self.line_number = line_number
            self.count = count

    def fill_word_statistics(self, word_statistics: Dict[str, WordStatistics]):
        word_to_count = {}
        for sentence in self.subject_and_content:
            for word in sentence:
                if word not in word_to_count:
                    word_to_count[word] = 0
                word_to_count[word] += 1

        for word, count in word_to_count.items():
            word_statistics[word] = LineInfo.WordStatistics(self.number, count)

    @staticmethod
    def __cleanup_content(content):
        # convert &#xd;&lt;br&gt;&lt to html tags
        unescaped_content = html.unescape(content)
        # after unescaping we can have '\r' in the text without '\n'
        clean_lines = unescaped_content.splitlines()
        # Each line can contain html tags e.g <br>
        if content != unescaped_content:
            clean_lines = [BeautifulSoup(line, 'lxml').text.lower() for line in clean_lines]
        # Now tokenize it
        return tokenizer.tokenize_sentances(clean_lines)

    def __repr__(self):
        return '#{}, Subject And Content: {}, Category: {}'.format(self.number, self.subject_and_content, self.category)

    @staticmethod
    def from_raw_line(line: str, number: int):
        # We expect the line to always be ordered, subject, content and maybe mainCat
        def simple_extract_xml_tag(line, tag_name, start_pos):
            start_tag = '<' + tag_name + '>'
            end_tag = '</' + tag_name + '>'
            tag_start_pos = line.find(start_tag, start_pos)
            if tag_start_pos == -1:
                return None, start_pos
            tag_end_pos = line.find(end_tag, tag_start_pos)
            if tag_end_pos == -1:
                return None, tag_start_pos
            tag_start_pos += len(start_tag)
            return line[tag_start_pos:tag_end_pos], tag_end_pos

        subject, pos = simple_extract_xml_tag(line, 'subject', 0)
        content, pos = simple_extract_xml_tag(line, 'content', pos)
        category, _ = simple_extract_xml_tag(line, 'maincat', pos)
        return LineInfo(number=number, subject=subject, content=content, category=category)


def line_information_from_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        raw_lines = file.readlines()
        lines = []
        for index, raw_line in enumerate(raw_lines):
            lines.append(LineInfo.from_raw_line(raw_line, index))
    return lines


def compute_lines_statistics(lines):
    word_to_statistics: Dict[str, List[LineInfo.WordStatistics]] = {}
    line_to_statistics: Dict[int, Dict[str, LineInfo.WordStatistics]] = {}
    for line_info in lines:
        line_word_statistics = {}
        line_info.fill_word_statistics(line_word_statistics)
        line_to_statistics[line_info.number] = line_word_statistics
        for word, word_statistic in line_word_statistics.items():
            if word not in word_to_statistics:
                word_to_statistics[word] = []
                word_to_statistics[word].append(word_statistic)
    return word_to_statistics, line_to_statistics


def compute_inverse_document_frequency(word_statistics: Dict[str, List[LineInfo.WordStatistics]], document_count: int):
    word_inverse_frequency = {}
    for word, statistics in word_statistics.items():
        count_documents_with_word = len(statistics)
        word_inverse_document_frequency = log10(document_count / count_documents_with_word)
        word_inverse_frequency[word] = word_inverse_document_frequency
    return word_inverse_frequency


def compute_weighted_representation(lines: List[LineInfo]):
    word_to_statistics, line_to_statistics = compute_lines_statistics(lines)
    # compute IDF(w_j)
    word_to_inverse_document_frequency = compute_inverse_document_frequency(word_to_statistics, document_count=len(lines))
    word_types = word_to_statistics.keys()
    word_type_count = len(word_types)
    word_to_index = {}

    for index, word in enumerate(word_types):
        word_to_index[word] = index
    for line_info in lines:
        line_term_weighted_vector = [0] * word_type_count
        term_frequencies_for_line: Dict[str, LineInfo.WordStatistics] = line_to_statistics[line_info.number]
        for word, word_statistics in term_frequencies_for_line.items():
            inverse_document_frequency_for_word = word_to_inverse_document_frequency[word]
            term_frequency = word_statistics.count
            word_weighted_representation = inverse_document_frequency_for_word * term_frequency
            word_index = word_to_index[word]
            line_term_weighted_vector[word_index] = word_weighted_representation
        line_info.vector = line_term_weighted_vector
    return word_to_index


def format_line_vector(vector):
    text = ''
    zero_sequence = 0
    for element in vector:
        if element == 0:
            zero_sequence += 1
            continue
        if text:
            text += ', '
        if zero_sequence > 0:
            text += '0^{}, '.format(zero_sequence)
            zero_sequence = 0
        text += '{}'.format(element)
    return text


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
    train_lines = line_information_from_file(args.train)

    if not args.use_train_other_half:
        classify_lines = line_information_from_file(args.classify)
    else:
        half_marker = int(len(train_lines) / 2)
        classify_lines = train_lines[:half_marker]
        train_lines = train_lines[half_marker:]
    word_to_index = compute_weighted_representation(train_lines)
    category_to_lines = {}
    for category in categories:
        category_to_lines[category] = []
    for line in train_lines:
        category_to_lines[line.category].append(line)

    category_to_average_weighted_representation = {}
    for category in categories:
        category_vector = [0] * len(word_to_index)
        category_lines = category_to_lines[category]
        for line in category_lines:
            for index, value in enumerate(line.vector):
                category_vector[index] += value
        for index, value in enumerate(category_vector):
            category_vector[index] = value / len(category_lines)
        category_to_average_weighted_representation[category] = category_vector

    for category in categories:
        print('{}: {}'.format(category, format_line_vector(category_to_average_weighted_representation[category])))


if __name__ == '__main__':
    try:
        logger.debug('Exercise 3')
        nltk.download('stopwords')
        nltk.download('punkt')
        main()
    finally:
        logger.debug('Bye bye...')
