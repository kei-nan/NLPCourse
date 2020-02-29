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

    def __compute_word_to_document_occurrences(self) -> Dict[str, List[Document.WordCount]]:
        word_to_document_occurrences: Dict[str, Document.WordCount] = {}
        for document_number, document in enumerate(self.documents):
            for word, count in document.word_to_word_count.items():
                if word not in word_to_document_occurrences:
                    word_to_document_occurrences[word] = []
                word_to_document_occurrences[word].append(count)
        return word_to_document_occurrences
