import html
from typing import Dict, List, Tuple
from bs4 import BeautifulSoup


class Document:
    TOKENIZER = None

    def __init__(self, subject: str, content: str, category: str = None):
        self.subject: str = subject
        self.category: str = category
        self.content: str = content if content else ''
        subject_and_content: str = self.subject + '\n' + self.content
        self.subject_and_content: str = self.__cleanup_content(subject_and_content)
        self.word_to_word_count: Dict[str, Document.WordCount] = self.__count_words()

    class WordCount:
        def __init__(self, document, count):
            self.document = document
            self.count = count

    def __count_words(self, count_threshold=0):
        word_count: Dict[str, Document.WordCount] = {}
        for sentence in self.subject_and_content:
            for word in sentence:
                if word not in word_count:
                    word_count[word] = Document.WordCount(self, 0)
                word_count[word].count += 1
        return {word: value for word, value in word_count.items() if value.count > count_threshold}


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
        tokenized_sentances = Document.TOKENIZER.tokenize_sentances(clean_lines)
        return tokenized_sentances

    def __repr__(self):
        first_sentance = '.*'.join(self.subject_and_content[0])
        return 'First Sentence: {}, Real Category: {}'.format(first_sentance, self.category)

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
        return Document(subject=subject, content=content, category=category)
