import abc
import math
import nltk
from bisect import bisect_left
from typing import List, Tuple, Dict
from exercise3.corpus import Corpus
from exercise3.document import Document
from exercise3.score_strategy import ScoreStrategy
from nltk import NaiveBayesClassifier, MaxentClassifier, corpus


class Classifier(abc.ABC):
    def __init__(self, name):
        self.name = name
        
    @abc.abstractmethod
    def classify_all(self, documents: List[Document]) -> List[str]:
        pass

    def show_most_informative_features(self, number):
        pass


class OneNearestNeighbor(Classifier):
    def __init__(self, score_strategy: ScoreStrategy, **kwargs):
        super(OneNearestNeighbor, self).__init__(name='1NearestNeighbor({})'.format(score_strategy.name), **kwargs)
        self.corpus = score_strategy.corpus
        self.score_strategy = score_strategy

    def classify_all(self, documents: List[Document]) -> List[str]:
        classify_results = [self.classify(document) for document in documents]
        return classify_results

    @staticmethod
    def __calc_distance(left, right):
        return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(left, right)))

    def classify(self, document: Document) -> str:
        document_scoring, category_scoring = self.score_strategy.score_document(document)
        nearest_category = None
        min_distance = None
        for category, weights in category_scoring.items():
            distance = OneNearestNeighbor.__calc_distance(weights, document_scoring)
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
            # values = [value for _, value in self.sorted_category_avg_word_count]
            # index = min(bisect_left(values, document.number_of_words), len(values) - 1)
            # closest_category_word_wise = self.sorted_category_avg_word_count[index][0]
            # features['closestCategoryWithWordCount'] = closest_category_word_wise
            return features

        def make_feature(document: Document, word_types: List[str]) -> Tuple[dict, str]:
            return document_features(document, word_types), document.category
        return [make_feature(d, self.most_frequent_words) for d in documents]

    def __init__(self, name: str, corpus: Corpus, **kwargs):
        super(NltkClassifier, self).__init__(name=name)
        self.corpus = corpus
        sorted_words_by_occurences = sorted(self.corpus.word_to_document_occurrences.items(),
                                            key=lambda x: len(x[1]),
                                            reverse=True)
        sorted_words_by_occurences = sorted_words_by_occurences[:NaiveBayes.TOP_WORDS]
        self.most_frequent_words = [k for (k, v) in sorted_words_by_occurences]
        self.sorted_category_avg_word_count = corpus.sorted_category_avg_word_count
        train_set = self.__create_feature_set(corpus.documents)
        self.classifier = self.classifier_type.train(train_set)

    def classify(self, document):
        return nltk.classify(document=document)

    def classify_all(self, documents: List[Document]):
        test_set = self.__create_feature_set(documents)
        results = self.classifier.classify_many([fs for (fs, l) in test_set])
        return results


class NaiveBayes(NltkClassifier, nltk_classifier=nltk.NaiveBayesClassifier):
    def __init__(self, **kwargs):
        super(NaiveBayes, self).__init__(name='NaiveBayes', **kwargs)

    def show_most_informative_features(self, number):
        self.classifier.show_most_informative_features(number)


# http://www.nltk.org/api/nltk.classify.html?highlight=naivebayesclassifier
class Maxent(NltkClassifier, nltk_classifier=MaxentClassifier):
    def __init__(self, **kwargs):
        super(Maxent, self).__init__(name='Maxent', **kwargs)

    def show_most_informative_features(self, number):
        pass
