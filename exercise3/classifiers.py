import abc
import math
from typing import List, Tuple, Dict
from exercise3.corpus import Corpus
from exercise3.document import Document


class Classifier(abc.ABC):
    @abc.abstractmethod
    def classify(self, document) -> str:
        pass

    def classify_all(self, documents: List[Document]) -> List[str]:
        return [self.classify(document) for document in documents]


class OneNearestNeighbor(Classifier):
    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def classify(self, document: Document) -> str:
        category_word_weights_for_document: Dict[str, List[float]] = {}
        for category in self.corpus.categories:
            category_word_weights_for_document[category] = []
        document_word_weights: List[float] = []
        for word, word_count in document.word_to_word_count.items():
            category_weights_for_word: Dict[str, Dict[str, float]] = \
                self.corpus.word_to_weighted_category.get(word, None)
            if category_weights_for_word is not None:
                for category, weights in category_weights_for_word.items():
                    category_word_weights_for_document[category].append(weights)
            else:
                for category in self.corpus.categories:
                    category_word_weights_for_document[category].append(0)
            document_weight_for_word, _ = self.corpus.get_word_weight_in_corpus_for_document(word=word,
                                                                                             occurence=word_count)
            document_word_weights.append(document_weight_for_word)
        nearest_category = None
        min_distance = None
        for category, weights in category_word_weights_for_document.items():
            distance = math.dist(weights, document_word_weights)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                nearest_category = category
        return nearest_category
