from typing import Tuple
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KMeansCluster:

    def __init__(self, k: int, strategy: str = '++', distance: str = 'euclidean', tolerance: float = 1e-4,
                 random_state: int = 33):
        """ Initializes the class """
        self.k = k
        self.strategy = strategy
        self.distance_alg = distance
        self.tol = tolerance
        self.random_state = random_state
        self.centroids = None
        self.__bootstrap(strategy, distance)
        np.random.seed(random_state)

    def __bootstrap(self, strategy: str, distance: str) -> None:
        """ Bootstraps the algorithm with the choice provided in the kwargs """
        if distance not in ('euclidean', 'manhattan'):
            raise NotImplementedError(f"'distance' must be a string of either 'manhattan' or 'euclidean', not '{distance}'")
        
        if strategy not in ('random', '++'):
            raise NotImplementedError(f"'strategy' must be one of 'random' or '++', not '{strategy}'")
        
        self.distance = self.euclidean if distance == 'euclidean' else self.manhattan
        self.initialize = self._random if strategy == 'random' else self._plus_plus

    def euclidean(self, x: np.ndarray, y: np.ndarray):
        if x.shape == y.shape:
            return np.sqrt(np.sum(np.power(x - y, 2)))
        return np.sqrt(np.sum(np.power(x - y, 2), axis=1))

    def manhattan(self, x: np.ndarray, y: np.ndarray):
        if x.shape == y.shape:
            return np.sum(np.abs(x - y))
        return np.sum(np.abs(x - y), axis=1)

    def _random(self, X: np.ndarray) -> np.ndarray:
        """ Initializes the centroids by randomly selecting K values from the input array 

            Note: This strategy may lead to misaligned centroids
        """
        centers = np.random.randint(0, X.shape[0], size=self.k)  # Get K random integer in range of [0, n]
        centroids = X[centers]  # k x d matrix
        return centroids

    def _plus_plus(self, X: np.ndarray) -> np.ndarray:
        """ Implements the KMeans++ algorithm for initalization. Basically, try to initialize centers that are far from one another.

            Reference: https://en.wikipedia.org/wiki/K-means%2B%2B
        """
        n, d = X.shape
        centroids = np.zeros((self.k, d))
        centroids[0] = X[np.random.randint(0, high=n, size=1).item()]  # Select a random item in the array
        for i in range(1, self.k):
            total_dist = np.zeros((n, i))  # n x i vector
            for j in range(i):
                dist = self.distance(X, centroids[j])  # Calculates the distance from each point to the centroid
                total_dist[:, j - 1] = dist  # Assigns the distances to the total for each cluster
            minimums = np.min(total_dist, axis=1)  # Get the distance from the nearest center
            p = minimums / np.sum(minimums)  # Calculates the probability that x is the next center
            partitions = np.cumsum(p)  # Partition the probabilities by cumualitively summing them
            r = np.random.rand()  # Randomly generate a probability between [0, 1]
            center = np.argmax(partitions > r)  # Find the first partition instance where the probability is greater than r
            centroids[i] = X[center]  # Assign the center
        return centroids

    def predict(self, X: np.ndarray):
        if self.centroids is None:
            raise ValueError("Must fit call 'fit()'")
        if X.shape[1] != self.centroids.shape[1]:
            raise ValueError(f"Input must have same dimensions as centroids: {X.shape[1]} != {self.centroids.shape[1]}")
        return self._assign(X)

    def fit(self, X: np.ndarray) -> None:
        """ Fits the data to a cluster """
        centroids = self.initialize(X)
        while (not self._stop(centroids)):
            self.centroids = centroids
            clusters = self._assign(X)
            centroids = self._recenter(X, clusters)
        self.centroids = centroids
            
    def _assign(self, X: np.ndarray) -> np.ndarray:
        """ Assigns each data point to a cluster """
        n, _ = X.shape  # n x d matrix
        dist = np.zeros((n, self.k))  # n x k matrix
        for i in range(self.k):
            center = self.centroids[i]  #  1 x d vector
            cluster_dist = self.distance(X, center)  # n x 1 vector
            dist[:, i] = cluster_dist  # assign ith column as n x 1 vector
        return np.argmin(dist, axis=1)  # n x 1 vector with each point's cluster

    def _recenter(self, X: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        """ Recenters the array """
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            indices = np.argwhere(clusters == i).flatten()
            centroids[i] = np.mean(X[indices], axis=0)  # 1 x d vector
        return centroids

    def _stop(self, centroids: np.ndarray) -> bool:
        """ Deteremins whether the fit process should stop """
        if self.centroids is None:
            return False
        return (self.distance(self.centroids, centroids)) < self.tol

    def __str__(self) -> str:
        return (
            f'<KMeansCluster(k={self.k}, strategy={self.strategy}, '
            f'distance={self.distance_alg}, tolerance={self.tol}, '
            f'random_state={self.random_state})>'
        )


def dispersion(X: np.ndarray, kmeans: KMeansCluster) -> float:
    """ Calculates the total dispersion for the KMeansCluster """
    clusters = kmeans.predict(X)
    dispersion = 0
    for i in range(kmeans.k):
        indices = np.argwhere(clusters == i).flatten()
        points = X[indices]
        center = kmeans.centroids[i]
        dispersion += np.sum(kmeans.distance(points, center))
    return dispersion


def standardize(data: pd.Series) -> np.ndarray:
    return (data - np.mean(data)) / np.std(data)


def mean(data: pd.Series) -> float:
    return sum(data) / len(data)


def max_categorical(data: pd.Series):
    classes = dict()
    for item in data:
        if pd.isna(item):
            continue
        if not classes.get(item):
            classes[item] = 0
        classes[item] += 1
    return max(classes.items(), key=lambda x: x[1])[0]


def onehot(data: pd.Series, prefix: str) -> pd.DataFrame:
    """ One hot encodes the data into a DataFrame where the columns are the category """
    classes = list(set(data))
    onehot = np.zeros((data.shape[0], len(classes)))
    for i, val in enumerate(data):
        onehot[i][classes.index(val)] = 1
    return pd.DataFrame(onehot, columns=[f"{prefix}-{k.lower()}" for k in classes])


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    _temp = df.replace('?', np.nan)
    cleaned = []
    categorical = set(_temp.select_dtypes(include=[object]).columns)
    numeric = set(_temp.select_dtypes(include=[np.int64]).columns)
    assert categorical | numeric == set(df.columns), "Missing columns from DF after cleaning"
    for col in categorical:
        _temp[col] = _temp[col].fillna(max_categorical(_temp[col]))
        cleaned.append(onehot(_temp[col], prefix=col))
    for col in numeric:
        _temp[col] = _temp[col].fillna(mean(_temp[col]))
        _temp[col] = standardize(_temp[col])
        cleaned.append(_temp[[col]])
    _df = pd.concat(cleaned, axis=1)
    assert _df.shape[0] == df.shape[0], f"Missing Values {df.shape[0]} != {_df.shape[0]}"
    return _df


def main(df: pd.DataFrame, output: str, search_range: Tuple[int, int], init: str, distance: str,
         random_state: int) -> None:
    train = preprocess(df).to_numpy()
    indices = np.random.permutation(train.shape[0])
    X_train = train[indices, :]
    scores = []
    start, end = search_range
    for i in range(start, end + 1):
        kmeans = KMeansCluster(i, strategy=init, distance=distance, random_state=random_state)
        kmeans.fit(X_train)
        disp = dispersion(X_train, kmeans)
        print(f"[{i} / {end}] {str(kmeans)} Score - {disp}")
        scores.append(dispersion(X_train, kmeans))
    plt.plot(list(range(start, end + 1)), scores)
    plt.savefig(output, bbox_inches='tight')


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=(
            'Runs the KMeans Algorithm over a range of numbers on a given input. '
            'Will output the elbow diagram by calculating the dispersion over the range of K'
        )
    )
    parser.add_argument('-p', '--path', type=str, dest='path', default='data/income.csv',
                        help='Path to the input file (must be a CSV)')
    parser.add_argument('-s', '--start', type=int, dest='start', default=1,
                        help="The start range for the K search (cannot be less than 1)")
    parser.add_argument('-e', '--end', type=int, dest='end', default=8,
                        help='The end range for the K search')
    parser.add_argument('-o', '--output', type=str, dest='output', default='outputs/elbow.png',
                        help='Where the elbow diagram should be written')
    parser.add_argument('-a', '--alg', type=str, dest='alg', choices=['++', 'random'], default='++',
                        help='The algorithm to use for centroid initialization')
    parser.add_argument('-r', '--random-state', type=int, dest='random_state', default=0,
                        help="The random state (used for reproducability)")
    parser.add_argument('-d', '--dist', type=str, dest='distance', choices=['euclidean', 'manhattan'], default='euclidean',
                        help='Which distance algorithm to use for clustering')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    if args.start < 1:
        raise ValueError(f"Start of search range cannot be less than 1")
    if args.random_state < 0:
        raise ValueError(f"Random State cannot be less than 0")
    if not os.path.exists(args.path):
        raise ValueError(f"Path '{args.path}' does not exist")

    df = pd.read_csv(args.path)
    main(df, args.output, (args.start, args.end), args.alg, args.distance, args.random_state)
