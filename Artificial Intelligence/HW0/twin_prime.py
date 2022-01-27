from argparse import ArgumentParser, Namespace
from typing import Iterator, Tuple
import math


def is_prime(n: int) -> bool:
    """ Checks if a number is a prime number """
    # Sanity check
    if n <= 1:
        return False

    for candidate in list(range(2, int(math.sqrt(n) + 1))):
        # Check if the number is divisible
        # If so, return that the number is not a prime
        if n % candidate == 0:
            return False
    return True


def find_max_primes(_range: Iterator) -> Tuple[int, int]:
    """ Finds the maximum primes given a range of numbers """
    x = next(_range, None)
    while x != None:
        # If the next number and the current number doesn't have a difference of two, it is not a candidate
        y = next(_range, None)
        if y == None or x - y != 2:
            x = y
        elif is_prime(y) and is_prime(x):
            break
        else:
            x = y
    return x, y


def main(target: int):
    if target < 5:
        raise ValueError(f"No twin primes exist that are less than 5")
    # Check if number is even and make number odd if it is
    if target % 2 == 0:
        target -= 1
    # Stepping by two because no even number can be prime
    # 2 is a prime number, but it can never be a 'twin' prime
    # Any number that ends with 5 cannot be a prime, so remove them prior to processing
    _range = filter(lambda n: n == 5 or not str(n).endswith('5'), range(target, 2, -2))
    x, y = find_max_primes(_range)
    return x, y


def _parse_args() -> Namespace:
    parser = ArgumentParser(description="This program computes the maxmimum twin prime bewteen 1 and x, given x")
    parser.add_argument('-n', '--number', type=int, required=True, dest='target', help=("The maximum a number"))
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    assert isinstance(args.target, int), "'-n' or must be an integer!"
    x, y = main(args.target)
    print(f"[1 - {args.target}] Max Twin Primes: {x} and {y}")
