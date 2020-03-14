from math import log10
from typing import Dict, List, Tuple
from exercise3.document import Document


class CategoryStatistics:
    def __init__(self):
        self.word_count = 0
        self.document_count = 0
        self.subject_only_documents = 0

    def update(self, document: Document):
        self.word_count += document.number_of_words
        self.document_count += 1
        self.subject_only_documents += 1 if document.has_only_subject() else 0

    @staticmethod
    def get_csv_headers():
        return ['Frequency', 'Average Length', 'Subject Only Ratio']

    def csv_format(self, total_document_count):
        return [str(self.document_count / total_document_count),
                str(self.word_count / self.document_count),
                str(self.subject_only_documents / self.document_count)]

class Corpus:
    def __init__(self, documents: List[Document], categories: List[str]):
        self.categories = categories
        self.documents: List[Document] = documents
        # TFj - Term frequency for word j
        self.word_to_document_occurrences: Dict[str, List[Document.WordCount]] = \
            self.__compute_word_to_document_occurrences()
        self.category_statistics: Dict[str, CategoryStatistics] = {}
        for category in self.categories:
            self.category_statistics[category] = CategoryStatistics()
        for document in self.documents:
            self.category_statistics[document.category].update(document)
        self.sorted_category_avg_word_count: List[Tuple[str, float]] = self.__sorted_category_avg_word_count()

    def print_statisitcs_as_csv(self):
        frequency_table = {}
        total_document_count = len(self.documents)
        for category, statistics in self.category_statistics.items():
            frequency_table[category] = statistics.document_count / total_document_count
        print(f'Word Types: {self.word_type_count()}, Total: {self.total_word_count()}')
        sorted_categories_by_frequency = [k for k, v in sorted(frequency_table.items(), key=lambda x: x[1], reverse=True)]
        ranking = 1
        headers = ['Ranking', 'Name'] + CategoryStatistics.get_csv_headers()
        print(', '.join(headers))
        for category in sorted_categories_by_frequency:
            statistic = self.category_statistics[category]
            csv_line = [str(ranking), category] + statistic.csv_format(total_document_count)
            print(', '.join(csv_line))
            ranking += 1

    def word_type_count(self):
        return len(self.word_to_document_occurrences)

    def total_word_count(self):
        count = 0
        for statistics in self.category_statistics.values():
            count += statistics.word_count
        return count

    def __sorted_category_avg_word_count(self) -> List[Tuple[str, float]]:
        category_to_avg_word_count = {}
        for category, statistics in self.category_statistics.items():
            category_to_avg_word_count[category] = \
                statistics.word_count / statistics.document_count if statistics.document_count else 0.0
        return sorted(category_to_avg_word_count.items(), key=lambda x: x[1])

    def percentage_of_subject_only_documents(self) -> Dict[str, float]:
        category_to_percentage_of_subject_only_documents = {}
        for category, statistics in self.category_statistics.items():
            category_to_percentage_of_subject_only_documents[category] = \
                statistics.subject_only_documents / statistics.document_count if statistics.document_count else 0.0
        return category_to_percentage_of_subject_only_documents

    def __compute_word_to_document_occurrences(self) -> Dict[str, List[Document.WordCount]]:
        word_to_document_occurrences: Dict[str, Document.WordCount] = {}
        for document_number, document in enumerate(self.documents):
            for word, count in document.word_to_word_count.items():
                if word not in word_to_document_occurrences:
                    word_to_document_occurrences[word] = []
                word_to_document_occurrences[word].append(count)
        return word_to_document_occurrences
