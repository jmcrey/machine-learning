from typing import List, Dict, Tuple
from collections import OrderedDict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Tfidf():

    def __init__(self) -> None:
        self._df = dict()
        self._features = None
        self._feature_map = OrderedDict()
        self._idf = None

    @property
    def features(self) -> np.ndarray:
        if not self._features:
            return np.ndarray([])
        return np.array(self._features)

    @property
    def feature_map(self) -> OrderedDict:
        return self._feature_map
    
    @feature_map.setter
    def feature_map(self, value: List[str]) -> None:
        """ Sets up the feature map based on the frozen set of features """
        for i, v in enumerate(value):
             self._feature_map[v] = i

    def fit(self, samples: List[str]) -> None:
        """ Calculates the IDF for the given corpus of documents """
        N = len(samples)  # Number of "documents"
        self._df = self._doc_frequency(samples)
        self._features = sorted(list(self._df.keys()))
        self.feature_map = self._features
        self._idf = self._inverse_doc_frequency(N)

    def transform(self, samples: List[str]) -> np.ndarray:
        """ Transforms the input samples into a vectorized format using the learned IDF 

            1) Calculate Term Frequency: tf(t, d) = Count of Term in Doc / Number of Words in Doc
            2) Multiply Term Frequency by IDF
        
        """
        tf = self._term_frequency(samples)
        return tf * self._idf

    def fit_transform(self, samples: List[str]) -> np.ndarray:
        """ Runs the fit and transforms functions """
        self.fit(samples)
        return self.transform(samples)

    def _doc_frequency(self, samples: List[str]) -> Dict[str, float]:
        df = dict()
        for sample in samples:
            tokens = set(sample.split())
            for token in tokens:
                if not df.get(token):
                    df[token] = 0.0
                df[token] += 1.0
        return df

    def _inverse_doc_frequency(self, N: int) -> np.ndarray:
        """ Returns a vectorized inverse document frequency """
        vec = np.zeros(len(self._features))
        for i, feature in enumerate(self._features):
            vec[i] = np.log(N / self._df[feature])
        return vec

    def _term_frequency(self, samples: List[str]) -> np.ndarray:
        """ Returns the term frequency

            1) tf(t, d) = Count of Term in Doc / Number of Words in Doc
            2) Multiply Term Frequency by IDF
        """
        N = len(samples)
        vec = np.zeros((N, len(self._features)))
        for i, sample in enumerate(samples):
            tokens = sample.split()
            D = len(tokens)
            for token in set(tokens):
                j = self._feature_map.get(token)
                if j:
                    vec[i, j] = tokens.count(token) / D
        return vec


class LogisticRegression:
    
    def __init__(self, epochs: int = 100, learning_rate: float = 1e-3, tolerance: float = 1e-4, verbose: bool = True):
        self.epochs = epochs
        self.lr = learning_rate
        self.tol = tolerance
        self.verbose = verbose
        self.w_ = None
        self.history_ = []

    def binary_cross_entropy(self, y: int, y_hat: float) -> float:
        """ Implementation of the binary cross entropy loss function 

            Formula: -(y log(y_hat) + (1 - y)log(1 - y_hat))
        """
        return -1.0 * (y * np.log(y_hat)) - ((1 - y) * np.log(1 - y_hat))

    def binary_cross_entropy_gradient(self, x: np.ndarray, y: int, y_hat: float) -> np.ndarray:
        """ Gradient of the binary cross entropy loss function 

            Formula: X^T(y - y_hat)
        """
        return x.T * (y_hat - y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Runs the prediction on all the samples """
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """ Runs the prediction on all the samples and returns the probability for each class """
        X = self._insert_bias(X)
        prediction = self.sigmoid(X)  # Output P(X = 1)
        return np.hstack((1 - prediction, prediction))

    def sigmoid(self, X: np.ndarray) -> np.ndarray:
        """ Runs the logistic function on the samples """
        return 1 / (1 + np.exp(-1.0 * np.dot(X, self.w_)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Fits the model to the data """
        if len(y.shape) > 1 and y.shape[1] != 1:
            raise ValueError(f"This model only supports binary classification. Y should only be one dimensional")
        
        if len(y.shape) == 2:
            y.reshape(y.shape[0])

        X = self._insert_bias(X)  # n x (d + 1) matrix
        n, d = X.shape  # n x d matrix
        if n != y.shape[0]:
            raise ValueError(f"X does not match y: {n} != {y.shape[0]}")

        if self.w_ is None:
            self.w_ = np.random.normal(scale=0.1, size=(d, 1))  # d x 1 vector

        loss = None
        for i in range(self.epochs):
            indices = np.random.permutation(n)
            xi = X[indices, :]  # n x d matrix
            yi = y[indices]  # n x 1 vector
            loss = self._opt(xi, yi)

            if self.verbose:
                print(f"Loss at epoch {i}: {loss}")

            if i > 0 and abs(self.history_[-1] - loss) < self.tol:
                print(f"Model converged at epoch {i}")
                self.history_.append(loss)
                break

            self.history_.append(loss)
            self.lr *= 0.9  # Decay learning rate for stability
    
    def _insert_bias(self, X: np.ndarray) -> np.ndarray:
        """ Inserts the bias into X variable """
        bias = np.ones((X.shape[0], 1))
        return np.hstack((bias, X))

    def _opt(self, xi: np.ndarray, yi: np.ndarray) -> float:
        """ Performs stochastic gradient descent algorithm """
        n, d = xi.shape
        loss = 0.0
        for i in range(n):
            x = xi[i].reshape((1, d))  # 1 x d vector
            y = yi[i]  # Scalar
            y_hat = self.sigmoid(x).item()  # Scalar
            loss += self.binary_cross_entropy(y, y_hat)
            g = self.binary_cross_entropy_gradient(x, y, y_hat)
            self.w_ -= self.lr * g
        return loss / n


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape does not match: {y_true.shape} != {y_pred.shape}")
    return np.sum(y_true == y_pred) / y_true.shape[0]


def plot(loss: List[float], output: str) -> None:
    """ Plots the loss """
    epochs = len(loss)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), loss)
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(output, bbox_inches='tight')


def preprocess(text: str, punc: List[str], stopwords: List[str]) -> str:
    """ Preprocesses the text """
    # Lowercase
    text = text.lower().strip()
    # Replace punctuation with a space before and after it
    for p in punc:
        text = text.replace(p, f" {p} ")
    # Remove stopwords
    for word in stopwords:
        text = ' '.join(filter(lambda x: x != word, text.split()))
    # Add space to the end if last character is in punctuation
    if text[-1] in punc:
      text += ' '
    return text


def prepare(train_df: pd.DataFrame, test_df: pd.DataFrame, punc: List[str], stopwords: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Prepares the training and test data for intake by the logistic regression model """
    tfidf = Tfidf()
    train_tweets = train_df['Tweet'].apply(preprocess, punc=punc, stopwords=stopwords).tolist()
    test_tweets = test_df['Tweet'].apply(preprocess, punc=punc, stopwords=stopwords).tolist()
    X_train = tfidf.fit_transform(train_tweets)
    y_train = train_df['Label'].apply(lambda x: int(x == 'Yes')).to_numpy()
    X_test = tfidf.transform(test_tweets)
    y_test = test_df['Label'].apply(lambda x: int(x == 'Yes')).to_numpy()
    return X_train, y_train, X_test, y_test


def train(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
          epochs: int, learning_rate: float, tol: float, verbose: bool) -> List[float]:
    cls = LogisticRegression(epochs=epochs, learning_rate=learning_rate, tolerance=tol, verbose=verbose)
    cls.fit(X_train, y_train)
    train_pred = cls.predict(X_train)
    train_accuracy = accuracy(y_train, train_pred)
    test_pred = cls.predict(X_test)
    test_accuracy = accuracy(y_test, test_pred)
    print(f'Train Accuracy: {train_accuracy}')
    print(f'Test Accuracy : {test_accuracy}')
    return cls.history_


def main():
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    with open(args.punc, 'r') as f:
        punc = tuple(p.strip() for p in f.read().strip().split('\n'))

    with open(args.stopwords, 'r') as f:
        stopwords = tuple(word.strip() for word in f.read().strip().split('\n'))
    X_train, y_train, X_test, y_test = prepare(train_df, test_df, punc, stopwords)
    history = train(X_train, y_train, X_test, y_test, args.epochs, args.lr, args.tol, args.verbose)
    plot(history, args.output)


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=(
            'Runs the TF-IDF Algorithm on given text input and inputs it into a Logistic Regression algorithm. '
            'Will output the accuracy score on the test set to STDOUT and the loss/epoch to a given output path.'
        )
    )
    parser.add_argument('--train', type=str, dest='train', default='data/swad_train.csv', 
                        help='Path to the training data (CSV File)')
    parser.add_argument('--test', type=str, dest='test', default='data/swad_test.csv', 
                        help='Path to the training data (CSV File)')
    parser.add_argument('--punctuation', type=str, dest='punc', default='data/punctuations.txt',
                        help='Path to punctuation text file')
    parser.add_argument('--stopwords', type=str, dest='stopwords', default='data/stopwords.txt',
                        help="Path to the stopwords text file")
    parser.add_argument('-o', '--output', type=str, dest='output', default='outputs/loss.png',
                        help='Where the elbow diagram should be written')
    parser.add_argument('-lr', '--learning-rate', type=float, dest='lr', default=0.1,
                        help='The learning rate for the logisitc regression model')
    parser.add_argument('--epochs', type=int, dest='epochs', default=100, 
                        help="The number of epochs to run")
    parser.add_argument('--tol', type=float, dest='tol', default=1e-5, 
                        help='Tolerance for model convergence')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False, 
                        help='Whether to print out verbose training message')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    if not os.path.exists(args.train):
        raise ValueError(f"Path '{args.train}' does not exist")

    if not os.path.exists(args.test):
        raise ValueError(f"Path '{args.test}' does not exist")

    if not os.path.exists(args.punc):
        raise ValueError(f"Path '{args.punc}' does not exist")

    if not os.path.exists(args.stopwords):
        raise ValueError(f"Path '{args.stopwords}' does not exist")

    main()
