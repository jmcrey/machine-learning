from typing import List, Tuple, Generator, Dict
from collections import defaultdict
from functools import reduce

import os
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Ngram:

    def __init__(self, n: int) -> None:
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"'n' must be an integer greater than or equal to 1, not {n.__class__.__name__}")
        self.n = n
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

    def _calculate_frequency(self, corpus: List[List[str]]) -> None:
        """ Calculates the frequency of the words and the pair of words 

            Parameters:
                None

            Returns:
                None. Sets the internal variables 'ngram_count' and 'vocab'
            
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
        self.vocab = set()
        self.ngram_count = defaultdict(lambda: defaultdict(int))
        for sentence in corpus:
            for ngram in self._ngrams(sentence):
                # Update the vocabulary with the items from the Ngram
                self.vocab.update(ngram)
                # Update the ngram frequency
                w_i = ngram[:-1]
                w_n = ngram[-1]
                self.ngram_count[w_i][w_n] += 1

    def score(self, word: str, history: Tuple[str] = None) -> float:
        """ Returns the probability of the word, given the history. """
        counts = self.ngram_count[history]
        # Words that have not been seen have a probability of 0.
        if not counts:
            return 0.
        # Calculate the relative frequency of the words
        word = counts[word]
        total = sum(counts.values())
        return word / total

    def npredict(self, n: int):
        """ Predicts 'n' number of words. """
        if n < 1:
            raise ValueError("'n' must be greater than or equal to 1")

        ngram = defaultdict(lambda: defaultdict(float))
        for history, words in self.ngram_count.items():
            for word in words.keys():
                ngram[history][word] = self.score(word, history)

        text = []
        look_back = self.n - 1
        while len(text) < n:
            history = tuple(text[-look_back:])
            word = self._predict(ngram, history)
            if isinstance(word, tuple):
                text.extend(word)
            else:
                text.append(word)
        return text[:n]

    def _predict(self, ngram: Dict[Tuple[str], Dict[str, float]], history: Tuple[str] = None) -> Tuple[str]:
        """ Given the history and the compute ngram, get the next highest word """
        counts = ngram[history]
        if counts:
            # Get the maximum probability based on the history
            word, _ = max(counts.items(), key=lambda x: x[1])
            return (word,)
        else:
            # Return the word with the maximum probability if there is no context or the model has not seen this combo of words
            scores = [(history, word, score) for history, words in ngram.items() for word, score in words.items()]
            _, _, score = max(scores, key=lambda x: x[2])
            possibilities = list(filter(lambda x: x[2] == score, scores))
            choice = np.random.randint(0, high=len(possibilities))
            return possibilities[choice][0]
    
    def predict_proba(self, tokens: List[str]) -> float:
        """ Predicts the probability that the sentence would have occurred """
        proba = []
        for ngram in self._ngrams(tokens):
            history = ngram[:-1]
            word = ngram[-1]
            p = self.score(word, history=history)
            if p:
                proba.append(p)
        return reduce(operator.mul, proba, 1.0)

    def perplexity(self, corpus: List[List[str]]) -> float:
        """ Returns the perplexity of a given corpus. Calculates perplexity using the following formula:

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
                >>> model = Ngram(2)
                >>> model.fit(corpus)
                >>> model.perplexity(corpus)
                1.6555065597696215
                >>> [model.perplexity([s]) for s in corpus]
                [1.515716566510398, 2.0, 1.4142135623730951, 1.7411011265922482]
                >>> model = Ngram(3)
                >>> model.fit(corpus)
                >>> model.perplexity(corpus)
                1.2970836542335147
                >>> [model.perplexity([s]) for s in corpus]
                [1.5650845800732873, 1.2457309396155174, 1.2457309396155174, 1.189207115002721]

        """
        proba = []
        for sentence in corpus:
            for ngram in self._ngrams(sentence):
                history = ngram[:-1]
                word = ngram[-1]
                p = self.score(word, history=history)
                if p:
                    proba.append(p)

        entropy = -1 * np.mean(np.log2(proba))
        return pow(2, entropy)

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


class Uniform(Ngram):

    """ Same as Ngram model, but scores the word using a uniform distribution. In this case, it is normalized by the size of the vocabulary. """

    def __init__(self, n: int) -> None:
        super().__init__(n)

    def score(self, word: str, history: Tuple[str] = None) -> float:
        """ Returns the probability of the word, given the history. Score is normalized by the length of the vocabulary """
        counts = self.ngram_count[history]
        word = counts[word]
        if not word:
            return 0.0
        else:
            return 1 / len(self.vocab)


class LaplaceNgram(Ngram):
    
    """ Same as Ngram model, but scores the word using add-one smoothing combined with relative frequency. """

    def __init__(self, n: int, factor: int = 1) -> None:
        super().__init__(n)
        self.factor = factor

    def score(self, word: str, history: Tuple[str] = None) -> float:
        """ Returns the probability of the word, given the history. """
        counts = self.ngram_count[history]
        word = counts[word]
        total = sum(counts.values())
        return (word + self.factor) / (total + (len(self.vocab) * self.factor))


def read(path: str) -> List[List[str]]:
    """ Reads in a file and tokenizes the input """
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist!")
    
    elif not os.path.isfile(path):
        raise ValueError(f"Path {path} is not a file!")
    
    with open(path, 'r') as f:
        content = f.read().splitlines()
    
    return [s.split() for s in content]


def run_ngram(n: int, X_train: List[List[str]], X_tests: List[List[List[str]]]) -> List[float]:
    """ Trains an Ngram model on the given training set and calcualtes the perplexity scores on the given
        test sets.

        Parameters:
            n (int): The size of the ngram
            X_train (List[List[str]]):  A tokenized corpus of text
            X_tests (List[List[List[str]]]): A list of tokenized testing corpuses
        
        Returns:
            List[float]: The perplexity scores on the test sets.

        Example:
            >>> corpus = [
                    ['This', 'is', 'the', 'first', 'sentence', '.'],
                    ['This', 'sentence', 'is', 'the', 'second', 'sentence', '.'],
                    ['And', 'this', 'is', 'the', 'third', 'one', '.'],
                    ['Is', 'this', 'the', 'first', 'sentence', '?']
                ]
            >>> run_ngram(n=2, X_train=corpus, X_tests=[corpus])
            [1.6555065597696215]
    """
    model = Ngram(n)
    model.fit(X_train)
    return [model.perplexity(test_set) for test_set in X_tests]


def run_uniform(n: int, X_train: List[List[str]], X_tests: List[List[List[str]]]) -> List[float]:
    """ Trains a Uniform model on the given training set and calcualtes the perplexity scores on the given
        test sets.

        Parameters:
            n (int): The size of the ngram
            X_train (List[List[str]]):  A tokenized corpus of text
            X_tests (List[List[List[str]]]): A list of tokenized testing corpuses
        
        Returns:
            List[float]: The perplexity scores on the test sets.

        Example:
            >>> corpus = [
                    ['This', 'is', 'the', 'first', 'sentence', '.'],
                    ['This', 'sentence', 'is', 'the', 'second', 'sentence', '.'],
                    ['And', 'this', 'is', 'the', 'third', 'one', '.'],
                    ['Is', 'this', 'the', 'first', 'sentence', '?']
                ]
            >>> run_uniform(n=1, X_train=corpus, X_tests=[corpus])
            [13.000000000000004]
    """
    model = Uniform(n)
    model.fit(X_train)
    return [model.perplexity(test_set) for test_set in X_tests]


def run_laplace(n: int, X_train: List[List[str]], X_tests: List[List[List[str]]], factor: int = 1) -> List[float]:
    """ Trains an Ngram model on the given training set and calcualtes the perplexity scores on the given
        test sets.

        Parameters:
            n (int): The size of the ngram
            X_train (List[List[str]]):  A tokenized corpus of text
            X_tests (List[List[List[str]]]): A list of tokenized testing corpuses
        
        Returns:
            List[float]: The perplexity scores on the test sets.

        Example:
            >>> corpus = [
                    ['This', 'is', 'the', 'first', 'sentence', '.'],
                    ['This', 'sentence', 'is', 'the', 'second', 'sentence', '.'],
                    ['And', 'this', 'is', 'the', 'third', 'one', '.'],
                    ['Is', 'this', 'the', 'first', 'sentence', '?']
                ]
            >>> run_laplace(n=2, X_train=corpus, X_tests=[corpus])
            [6.349880138332044]
    """
    model = LaplaceNgram(n, factor=factor)
    model.fit(X_train)
    return [model.perplexity(test_set) for test_set in X_tests]


def report(model_name: str, names: List[str], scores: List[float]) -> None:
    print("#" * 35)
    print(f"## Results for the {model_name}")
    for i in range(len(names)):
        print(f"##   - {names[i]}: {scores[i]}")
    print("#" * 35)


def batch_report(model_name: str, iteration: int, names: List[str], scores: List[float]) -> None:
    if iteration == 1:
        print("#" * 35)
        print(f"## Results for the {model_name}")
        print("  {scores}".format(scores='\t\t'.join(names)))
    print("{i} {scores}".format(i=iteration, scores='\t\t'.join([str(round(s, 2)) for s in scores])))


def plot(title: str, scores: List[List[float]], columns: List[str]) -> None:
    df = pd.DataFrame(scores, columns=columns)
    df['n'] = list(range(1, len(scores) + 1))
    df.plot(x='n', y=columns, title=title)
    plt.savefig(os.path.join('outputs', f'{title}.png'), bbox_inches='tight')


def main():
    # Read in all the training data
    ted = read('ted.txt')
    reddit = read('reddit.txt')

    # Read in the test data
    test_ted = read('test.ted.txt')
    test_reddit = read('test.reddit.txt')
    test_news = read('test.news.txt')
    test_names = ['Ted', 'Reddit', 'News']
    test_sets = [test_ted, test_reddit, test_news]

    # Run the Uniform model
    scores = run_uniform(n=1, X_train=ted, X_tests=test_sets)
    report('Uniform Model', test_names, scores)

    # Run the Unigram model
    scores = run_ngram(n=1, X_train=ted, X_tests=test_sets)
    report('Unigram Model', test_names, scores)


    # Run the Ngram model
    ted_scores = []
    for i in range(1, 8):
        scores = run_ngram(n=i, X_train=ted, X_tests=test_sets)
        batch_report(f'Ted Ngram', i, test_names, scores)
        ted_scores.append(scores)
    
    reddit_scores = []
    for i in range(1, 8):
        scores = run_ngram(n=i, X_train=reddit, X_tests=test_sets)
        batch_report(f'Reddit Ngram', i, test_names, scores)
        reddit_scores.append(scores)

    plot('Perplexity for Ted Data (Ngram)', ted_scores, test_names)
    plot('Perplexity for Reddit Data (Ngram)', reddit_scores, test_names)

    # Run the Laplace model
    ted_scores = []
    for i in range(2, 8):
        if i == 2:
            print("#" * 35)
            print(f"## Results for the Ted Laplace")
            print("  {scores}".format(scores='\t\t'.join(test_names)))
        scores = run_laplace(n=i, X_train=ted, X_tests=test_sets)
        batch_report(f'Ted Laplace', i, test_names, scores)
        ted_scores.append(scores)
    
    reddit_scores = []
    for i in range(2, 8):
        if i == 2:
            print("#" * 35)
            print(f"## Results for the Reddit Laplace")
            print("  {scores}".format(scores='\t\t'.join(test_names)))
        scores = run_laplace(n=i, X_train=reddit, X_tests=test_sets)
        batch_report(f'Ted Laplace', i, test_names, scores)
        reddit_scores.append(scores)

    plot('Perplexity for Ted Data (Laplace)', ted_scores, test_names)
    plot('Perplexity for Reddit Data (Laplace)', reddit_scores, test_names)

    ted_ngram = Ngram(7)
    ted_ngram.fit(ted)
    ted_text = ted_ngram.npredict(500)
    with open(os.path.join('outputs', 'ted.out'), 'w+') as f:
        f.write(' '.join(ted_text))
    print(f"Perplexity for Ted Generated Text: {ted_ngram.perplexity([ted_text])}")
    
    reddit_ngram = Ngram(7)
    reddit_ngram.fit(reddit)
    reddit_text = reddit_ngram.npredict(500)
    with open(os.path.join('outputs', 'reddit.out'), 'w+') as f:
        f.write(' '.join(reddit_text))
    print(f"Perplexity for Reddit Generated Text: {reddit_ngram.perplexity([reddit_text])}")


if __name__ == '__main__':
    main()
