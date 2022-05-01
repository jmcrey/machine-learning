from typing import List, Tuple, Generator
from collections import defaultdict
from functools import reduce

import os
import operator
import numpy as np


class Ngram:

    def __init__(self, n: int) -> None:
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"'n' must be an integer greater than or equal to 1, not {n.__class__.__name__}")
        self.n = n
        self.ngram = None
        self.start_token = "<SOS>"
        self.end_token = "<EOS>"
        self.unknown_token = "<UNK>"
        self.pad_left = False
        self.pad_right = False
    
    def fit(self, corpus: List[List[str]], pad_left: bool = False, pad_right: bool = False) -> None:
        """ Fits the Ngram to the corpus of text given. 
        
            Parameters:
                corpus (List[str]): A tokenized corpus of text
                pad_left (bool = True): Whether or not to add a start symbol to each sentence
                pad_right (bool = True): Whether or not to add an end symbol to each sentence
    
            Returns:
                None. Initializes the internal model 'ngram'
        """
        self.pad_left = pad_left
        self.pad_right = pad_right
        self._calculate_frequency(corpus)
        self._fit()

    def _calculate_frequency(self, corpus: List[List[str]]) -> None:
        """ Calculates the frequency of the words and the pair of words 

            Parameters:
                None

            Returns:
                None. Sets the internal variables 'word_frequency' and 'ngram_frequency'
            
            Example:
                >>> corpus = [
                    ['This', 'is', 'the', 'first', 'sentence', '.'],
                    ['This', 'sentence', 'is', 'the', 'second', 'sentence', '.'],
                    ['And', 'this', 'is', 'the', 'third', 'one', '.'],
                    ['Is', 'this', 'the', 'first', 'sentence', '?']
                ]
                >>> model = Ngram(2)
                >>> model.fit(corpus)
                >>> dict(model.ngram)
                {('This',): defaultdict(<class 'float'>, {'is': 0.5, 'sentence': 0.5}),
                 ('is',): defaultdict(<class 'float'>, {'the': 1.0}),
                 ('the',): defaultdict(<class 'float'>, {'first': 0.5, 'second': 0.25, 'third': 0.25}),
                 ('first',): defaultdict(<class 'float'>, {'sentence': 1.0}), ...}
        """
        self.ngram_count = defaultdict(lambda: defaultdict(int))
        for sentence in corpus:
            for ngram in self._ngrams(sentence):
                # Update the ngram frequency
                w_i = ngram[:-1]
                w_n = ngram[-1]
                self.ngram_count[w_i][w_n] += 1

    def _fit(self) -> None:
        """ Calcualates the probability of word-pairs.
        
            P(w_i | w_i-1... w_1)
        """
        self.ngram = defaultdict(lambda: defaultdict(float))
        for ngram, pairs in self.ngram_count.items():
            ngram_frequency = float(sum(pairs.values()))
            for word, word_frequency in pairs.items():
                self.ngram[ngram][word] = word_frequency / ngram_frequency

    def npredict(self, n: int):
        """ Predicts 'n' number of words. """
        pass

    def _predict(self, ngram: Tuple[str]) -> str:
        """"""
        pass
    
    def predict_proba(self, tokens: List[str]) -> float:
        """ Predicts the probability that the sentence would have occurred """
        if not self.ngram:
            raise ValueError("Must call 'fit()' before predict. No model has been initialized!")
        
        proba = []
        for ngram in self._ngrams(tokens):
            w_i = ngram[:-1]
            w_n = ngram[-1]
            proba.append(self.ngram[w_i][w_n])
        return reduce(operator.mul, proba, 1.0)

    def perplexity(self, corpus: List[List[str]]) -> float:
        """ Returns the perplexity of a given corpus. Calculates perplexity using the following formula:

            sqrt(1 / P(w1, w2, ..., wN), N)
        
            Parameters:
                corpus (List[List[str]]): The tokenized corpus of sentences
            
            Returns:
                float: The perplexity score
            
            Example:
                >>> corpus = [
                    ['This', 'is', 'the', 'first', 'sentence', '.'],
                    ['This', 'sentence', 'is', 'the', 'second', 'sentence', '.'],
                    ['And', 'this', 'is', 'the', 'third', 'one', '.'],
                    ['Is', 'this', 'the', 'first', 'sentence', '?']
                ]
                >>> model = Ngram(2)
                >>> model.fit(corpus)
                >>> model.perplexity(corpus)
                1.9200933737095864
        """
        proba = []
        vocabulary = set()
        for sentence in corpus:
            for ngram in self._ngrams(sentence):
                w_i = ngram[:-1]
                w_n = ngram[-1]
                proba.append(self.ngram[w_i][w_n])
                vocabulary.add(ngram)

        proba = reduce(operator.mul, proba, 1.0)
        if not proba:
            return float("inf")
        return pow(1 / proba, 1 / len(vocabulary))

    def _ngrams(self, tokens: List[str]) -> Generator[Tuple[str], None, None]:
        """ Yield the ngrams of the sentence.

            Parameters:
                tokens (List[str]): A list of words that need to be gram'd

            Returns:
                A generator that yield tuples of strings in the size of `self.n`

            Example:
                >>> tokens = ['this', 'is', 'a', 'sentence']
                >>> self.n = 2
                >>> for pair in _ngrams(tokens):
                ...    print(pair)
                ('this', 'is')
                ('is', 'a')
                ('a', 'sentence')
                
        """
        if self.pad_left or self.pad_right:
            tokens = tokens.copy()
            if self.pad_left:
                tokens.insert(0, self.start_token)
            if self.pad_right:
                tokens.append(self.end_token)
        ngrams = [tokens[i:] for i in range(self.n)]
        return zip(*ngrams)


class Unigram(Ngram):

    def __init__(self, dist: str = 'uniform') -> None:
        super().__init__(n=1)
        if dist not in {'uniform', 'relative'}:
            raise ValueError(f"'dist' must be one of 'uniform' or 'relative'. '{dist}' is not supported.")
        self.distribution = dist
        self.vocabulary_count = None

    def _calculate_frequency(self, corpus: List[List[str]]) -> None:
        """ Calculates the frequency of the words and the pair of words 

            Parameters:
                None

            Returns:
                None. Sets the internal variable 'vocabulary_count'
        """
        self.vocabulary_count = defaultdict(int)
        for sentence in corpus:
            for word in sentence:
                self.vocabulary_count[word] += 1

    def _fit(self) -> None:
        """ Calcualates the probability of word-pairs.
        
            P(w_i | w_i-1... w_1)
        """
        if not self.vocabulary_count:
            raise Exception("Unexpected Error: 'vocabulary_count' not yet set. Please call 'calculate_frequency()' before this function.")

        self.ngram = defaultdict(float)
        self.vocabulary = set(self.vocabulary_count.keys())
        size = len(self.vocabulary)
        for word in self.vocabulary:
            if self.distribution == 'relative':
                # Set the probability to the relative count of the words
                self.ngram[word] = self.vocabulary_count[word] / size
            else:
                # Set the probability to a uniform count
                self.ngram[word] = 1 / size
    
    def predict(self, sentence: List[str]):
        """ Predicts the next word given the sentence. """
        pass
    
    def predict_proba(self, tokens: List[str]) -> float:
        """ Predicts the probability that the sentence would have occurred """
        if not self.ngram:
            raise ValueError("Must call 'fit()' before predict. No model has been initialized!")
        
        proba = [self.ngram[word] for word in tokens]
        return reduce(operator.mul, proba, 1.0)

    def perplexity(self, corpus: List[List[str]]) -> float:
        """ Returns the perplexity of a given corpus. Calculates perplexity using the following equivalent formula:

            2^(- (1 / N) log2(P(w1, w2, ..., wN))

            Here, N is the size of the vocabulary. Note that this formula is used because it is more numerically stable
            than the traditional perplexity formula:

            sqrt(1 / P(w1, w2, ..., wN), N)
        
            Parameters:
                corpus (List[List[str]]): The tokenized corpus of sentences
            
            Returns:
                float: The perplexity score
            
            Example:
                >>> corpus = [
                    ['This', 'is', 'the', 'first', 'sentence', '.'],
                    ['This', 'sentence', 'is', 'the', 'second', 'sentence', '.'],
                    ['And', 'this', 'is', 'the', 'third', 'one', '.'],
                    ['Is', 'this', 'the', 'first', 'sentence', '?']
                ]
                >>> model = Unigram()
                >>> model.fit(corpus)
                >>> model.perplexity(corpus)
                169.00000000000003
                >>> model = Unigram(dist='relative')
                >>> model.fit(corpus)
                >>> model.perplexity(corpus)
                31.494993094629706
        """
        proba = []
        vocabulary = set()
        for sentence in corpus:
            for word in sentence:
                vocabulary.add(word)
                proba.append(self.ngram[word])

        proba = reduce(operator.mul, proba, 1.0)
        if not proba:
            return float("inf")
        return pow(1 / proba, 1 / len(vocabulary))


def read(path: str) -> List[List[str]]:
    """ Reads in a file and tokenizes the input """
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist!")
    
    elif not os.path.isfile(path):
        raise ValueError(f"Path {path} is not a file!")
    
    with open(path, 'r') as f:
        content = f.read().splitlines()
    
    return [s.split() for s in content]



ted = read('ted.txt')
reddit = read('reddit.txt')
test_ted = read('test.ted.txt')
test_reddit = read('test.reddit.txt')
test_news = read('test.news.txt')

test_sets = [test_ted, test_reddit, test_news]

model = Unigram()
model.fit(ted)
print([
    model.perplexity(test) for test in test_sets
])

model = Unigram(dist='relative')
model.fit(ted)
print([
    model.perplexity(test) for test in test_sets
])

for i in range(1, 8):
    if i == 1:
        model = Unigram(dist='relative')
    else:
        model = Ngram(i)
    
    model.fit(ted)
    print([
        model.perplexity(test) for test in test_sets
    ])


