from argparse import ArgumentParser, Namespace
from typing import List

import os
import csv
import math


_path = os.path.join(os.path.dirname(__file__), 'data', 'covid.csv')


def get_stddev(data: List[int], mean: float) -> float:
    """ Calculates the standard deviation given the data and mean """
    dist = list()
    for x in data:
        dist.append(math.pow(abs(x - mean), 2))
    stddev = math.sqrt(sum(dist) / len(data))
    return stddev


def get_mean(data: List[int]) -> float:
    """ Calculates the mean of a given list of integers """
    total = 0.0
    count = 0.0
    for x in data:
        total += x
        count += 1
    return total / count


def get_min(data: List[int]) -> int:
    """ Calculates the minimum or maximum """
    min_ = math.inf
    for x in data:
        if x < min_:
            min_ = x
    return min_

def get_max(data: List[int]) -> int:
    """ Calculates the minimum or maximum """
    max_ = -1
    for x in data:
        if x > max_:
            max_ = x
    return max_


def main(target: str, cases: dict) -> None:
    """ Calcs the min, max, mean and stardard deviation of a given target 
    
        Note: this function uses custom built methods to calculate these metrics.
        I could have also calculated the metrics like so:

            min_ = min(data)
            max_ = max(data)
            mean = np.mean(data)
            stddev = np.std(data)
    """
    if not cases.get(target):
        print("Target has no available data!")
        return
    
    data = cases.get(target)
    min_ = get_min(data)
    max_ = get_max(data)
    mean = get_mean(data)
    stddev = get_stddev(data, mean)
    print(f"Stats for {target.upper()}:\n\t- Min: {min_}\n\t- Max: {max_}\n\t- Mean: {mean}\n\t- Standard Deviation: {stddev}")


def _parse_args() -> Namespace:
    """ Parse command line arguments """
    parser = ArgumentParser(
        description="This program computes stats for the number of COVID cases in a given state"
    )
    parser.add_argument('-s', '--state', type=str, required=True, dest='target', help="The name of the US state")
    return parser.parse_args()


if __name__ == '__main__':
    cases = dict()
    with open(_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = row['state'].lower().strip()
            if not cases.get(state):
                cases[state] = list()
            cases[state].append(int(row['cases']))
    args = _parse_args()
    target = args.target.lower().strip()
    main(target, cases)
