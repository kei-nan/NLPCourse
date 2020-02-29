import abc
import math
import nltk
import numpy
from typing import List, Tuple, Dict
from exercise3.corpus import Corpus
from exercise3.document import Document
from nltk import NaiveBayesClassifier, MaxentClassifier, corpus


class Classifier(abc.ABC):

    @abc.abstractmethod
    def classify_all(self, documents: List[Document]) -> List[str]:
        pass

    def show_most_informative_features(self, number):
        pass


class OneNearestNeighbor(Classifier):
    def __init__(self, corpus: Corpus, **kwargs):
        self.corpus = corpus

    def classify_all(self, documents: List[Document]) -> List[str]:
        classify_results = [self.classify(document) for document in documents]
        success = 0
        for index, document in enumerate(documents):
            if document.category and classify_results[index] == document.category:
                success += 1
        return success / len(documents)

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

    def show_most_informative_features(self, number):
        pass


class NltkClassifier(Classifier):
    TOP_WORDS = 1000

    def __init_subclass__(cls, nltk_classifier):
        cls.classifier_type = nltk_classifier
        super(NltkClassifier, cls).__init_subclass__()

    def __create_feature_set(self, documents: List[Document]):
        def document_features(document: Document, word_types: List[str]):
            features = {}
            for word in word_types:
                features[f'contains({word})'] = word in document.word_to_word_count
            return features

        def make_feature(document: Document, word_types: List[str]) -> Tuple[dict, str]:
            return document_features(document, word_types), document.category
        return [make_feature(d, self.most_frequent_words) for d in documents]

    def __init__(self, corpus: Corpus, **kwargs):
        sorted_words_by_occurences = sorted(corpus.word_to_document_occurrences.items(),
                                            key=lambda x: len(x[1]),
                                            reverse=True)
        sorted_words_by_occurences = sorted_words_by_occurences[:NaiveBayes.TOP_WORDS]
        self.most_frequent_words = [k for (k, v) in sorted_words_by_occurences]
        train_set = self.__create_feature_set(corpus.documents)
        self.classifier = self.classifier_type.train(train_set)

    def classify(self, document):
        return nltk.classify(document=document)

    def classify_all(self, documents: List[Document]):
        test_set = self.__create_feature_set(documents)
        return nltk.classify.accuracy(self.classifier, test_set)


class NaiveBayes(NltkClassifier, nltk_classifier=nltk.NaiveBayesClassifier):
    def show_most_informative_features(self, number):
        self.classifier.show_most_informative_features(number)


# http://www.nltk.org/api/nltk.classify.html?highlight=naivebayesclassifier
class MaxentClassifier(NltkClassifier, nltk_classifier=MaxentClassifier):
    def show_most_informative_features(self, number):
        pass
