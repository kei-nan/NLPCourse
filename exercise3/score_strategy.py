import abc

from math import log10
from exercise3.corpus import Corpus
from exercise3.document import Document
from typing import Dict, List, Tuple


class ScoreStrategy(abc.ABC):
    @abc.abstractmethod
    def __init__(self, corpus: Corpus, **kwargs):
        self.corpus = corpus

    @abc.abstractmethod
    # return dict from word to score
    def score_document(self, document: Document) -> Tuple[List[float], Dict[str, List[float]]]:
        pass


class TfIdfStrategy(ScoreStrategy):
    def __init__(self, **kwargs):
        super(TfIdfStrategy, self).__init__(**kwargs)
        self.category_to_document_count = self.__compute_category_to_document_count(categories=self.corpus.categories)
        self.__idf = self.__compute_inverse_document_frequency()
        self.word_to_weighted_category: Dict[str, Dict[str, float]] = \
            self.__compute_word_to_weighted_category(categories=self.corpus.categories)
        self.word_types: List[str] = list(self.corpus.word_to_document_occurrences.keys())

    def __compute_category_to_document_count(self, categories):
        category_to_document_count = {}
        for category in categories:
            category_to_document_count[category] = 0
        for document in self.corpus.documents:
            category_to_document_count[document.category] += 1
        return category_to_document_count

    # IDF - inverse document frequency
    def __compute_inverse_document_frequency(self) -> Dict[str, float]:
        document_count = len(self.corpus.documents)
        inverse_document_frequency: Dict[str, float] = {}
        for word, document_occurrences in self.corpus.word_to_document_occurrences.items():
            count_documents_with_word = len(document_occurrences)
            word_inverse_document_frequency = log10(document_count / count_documents_with_word)
            inverse_document_frequency[word] = word_inverse_document_frequency
        return inverse_document_frequency

    def get_word_weight_in_corpus_for_document(self, occurence: Document.WordCount, word: str) -> Tuple[float, str]:
        document: Document = occurence.document
        term_frequency = document.word_to_word_count[word].count if word in document.word_to_word_count else 0
        return self.__idf.get(word, 0) * term_frequency, document.category

    def __compute_word_to_weighted_category(self, categories: List[str]) -> Dict[str, Dict[str, float]]:
        word_to_weighted_categories: Dict[str, Dict[str, float]] = {}
        for word, occurrences in self.corpus.word_to_document_occurrences.items():
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

    def get_word_score_for_categories(self, word: str) -> Dict[str, float]:
        return self.word_to_weighted_category.get(word, None)

    # return dict from word to score
    def score_document(self, document: Document) -> Tuple[List[float], Dict[str, List[float]]]:
        category_word_score_for_document: Dict[str, List[float]] = {}
        for category in self.corpus.categories:
            category_word_score_for_document[category] = []
        document_word_scoring: List[float] = []
        for word, word_count in document.word_to_word_count.items():
            category_weights_for_word: Dict[str, float] = self.get_word_score_for_categories(word)
            if category_weights_for_word is not None:
                for category, score in category_weights_for_word.items():
                    category_word_score_for_document[category].append(score)
            else:
                for category in self.corpus.categories:
                    category_word_score_for_document[category].append(0)
            document_score_for_word, _ = self.get_word_weight_in_corpus_for_document(word=word,
                                                                                     occurence=word_count)
            document_word_scoring.append(document_score_for_word)
        return document_word_scoring, category_word_score_for_document


class BinaryStrategy(ScoreStrategy):
    def __init__(self, **kwargs):
        super(BinaryStrategy, self).__init__(**kwargs)
        self.is_word_in_category: Dict[str, Dict[str, bool]] = self.__calc_is_word_in_category()

    def __calc_is_word_in_category(self) -> Dict[str, Dict[str, bool]]:
        is_word_in_category: Dict[str, Dict[str, bool]] = {}
        for word in self.corpus.word_to_document_occurrences.keys():
            is_category_with_word = {}
            for category in self.corpus.categories:
                is_category_with_word[category] = False
            for d in self.corpus.word_to_document_occurrences[word]:
                is_category_with_word[d.document.category] = True
            is_word_in_category[word] = is_category_with_word
        return is_word_in_category

    def score_document(self, document: Document) -> Tuple[List[float], Dict[str, List[float]]]:
        category_word_score_for_document: Dict[str, List[float]] = {}
        for category in self.corpus.categories:
            category_word_score_for_document[category] = []
        document_word_scoring: List[float] = []
        for word, word_count in document.word_to_word_count.items():
            categories_containing_words = self.is_word_in_category.get(word, {})
            for category in self.corpus.categories:
                category_word_score_for_document[category].append(1.0 if categories_containing_words.get(category, False) else 0.0)
            document_word_scoring.append(1.0)
        return document_word_scoring, category_word_score_for_document
