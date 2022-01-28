from argparse import ArgumentParser, Namespace
from typing import List

import os
import csv
import math


_path = os.path.join(os.path.dirname(__file__), 'data', 'covid.csv')


def get_std(data: List[int], mean: float) -> float:
    """ Calculates the standard deviation given the data and mean """
    dist = list()
    for x in data:
        dist.append(math.pow(abs(x - mean), 2))
    std = math.sqrt(sum(dist) / (len(data) - 1))
    return std


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

        Can also use pandas:
            data = df.loc[df.state == {target}]
            mean = data.cases.mean()
            std = data.cases.std()
            ...
    """
    if not cases.get(target):
        print("Target has no available data!")
        return
    
    data = cases.get(target)
    min_ = get_min(data)
    max_ = get_max(data)
    mean = get_mean(data)
    stddev = get_std(data, mean)
    print(f"Stats for {target.upper()}:\n\t- Min: {min_}\n\t- Max: {max_}\n\t- Mean: {mean}\n\t- Standard Deviation: {stddev}")


def _get_data(path: str) -> dict:
    """ Reads the data from the CSV """
    cases = dict()
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = row['state'].lower().strip()
            if not cases.get(state):
                cases[state] = list()
            cases[state].append(float(row['cases']))
    return cases


def _parse_args() -> Namespace:
    """ Parse command line arguments """
    parser = ArgumentParser(
        description=(
            "This program computes stats for the number of COVID cases in a given state. "
            "Encapsulate the state in quotes (e.g. \"new york\") for most accurate reading"
        )
    )
    parser.add_argument('-s', '--state', type=str, required=True, dest='target', help="The name of the US state (long form only)")
    return parser.parse_args()


if __name__ == '__main__':
    cases = _get_data(_path)
    args = _parse_args()
    target = args.target.lower().strip()
    main(target, cases)
