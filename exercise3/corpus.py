from math import log10
from typing import Dict, List, Tuple
from exercise3.document import Document


class Corpus:
    def __init__(self, documents: List[Document], categories: List[str]):
        self.categories = categories
        self.documents: List[Document] = documents
        # TFj - Term frequency for word j
        self.word_to_document_occurrences: Dict[str, List[Document.WordCount]] = \
            self.__compute_word_to_document_occurrences()
        self.category_to_document_count = self.__compute_category_to_document_count(categories=categories)
        self.__idf = self.__compute_inverse_document_frequency()
        self.word_types = self.word_to_document_occurrences.keys()
        self.word_to_weighted_category: Dict[str, Dict[str, float]] = \
            self.__compute_word_to_weighted_category(categories=categories)

    def __compute_category_to_document_count(self, categories):
        category_to_document_count = {}
        for category in categories:
            category_to_document_count[category] = 0
        for document in self.documents:
            category_to_document_count[document.category] += 1
        return category_to_document_count

    def __compute_word_to_document_occurrences(self) -> Dict[str, List[Document.WordCount]]:
        word_to_document_occurrences: Dict[str, Document.WordCount] = {}
        for document_number, document in enumerate(self.documents):
            for word, count in document.word_to_word_count.items():
                if word not in word_to_document_occurrences:
                    word_to_document_occurrences[word] = []
                word_to_document_occurrences[word].append(count)
        return word_to_document_occurrences

    # IDF - inverse document frequency
    def __compute_inverse_document_frequency(self) -> Dict[str, float]:
        document_count = len(self.documents)
        inverse_document_frequency: Dict[str, float] = {}
        for word, document_occurrences in self.word_to_document_occurrences.items():
            count_documents_with_word = len(document_occurrences)
            word_inverse_document_frequency = log10(document_count / count_documents_with_word)
            inverse_document_frequency[word] = word_inverse_document_frequency
        return inverse_document_frequency

    def get_word_weight_in_corpus_for_document(self, occurence: Document.WordCount, word: str) -> Tuple[float, str]:
        document: Document = occurence.document
        term_frequency = document.word_to_word_count[word].count if word in document.word_to_word_count else 0
        return self.__idf[word] * term_frequency, document.category

    def __compute_word_to_weighted_category(self, categories: List[str]) -> Dict[str, Dict[str, float]]:
        word_to_weighted_categories: Dict[str, Dict[str, float]] = {}
        for word, occurrences in self.word_to_document_occurrences.items():
            word_to_weighted_categories[word] = {}
            for category in categories:
                word_to_weighted_categories[word][category] = 0.0
            for occurrence in occurrences:
                weighted_score, category = self.get_word_weight_in_corpus_for_document(occurrence, word)
                word_to_weighted_categories[word][category] += weighted_score
            for category in categories:
                document_count_in_category = self.category_to_document_count[category]
                word_to_weighted_categories[word][category] /= document_count_in_category
        return word_to_weighted_categories
