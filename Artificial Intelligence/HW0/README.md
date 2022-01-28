## Twin Prime
Calculates the maximum twin prime between 1 and a given number (including the given number)

Usage: `python twin_prime.py -n 5` (run `python twin_prime.py -h` for details)

Returns: "[1 - 5] Max Twin Prime: 5 and 3"

## Matrix Manipulation
Caluclates the grayscale for a particular image. Requires installation of the `requirements.txt` file

Install: `pip -r requirements.txt`

Usage: `python grayscale.py -p /path/to/image -o /path/to/outfile` (run `python grayscale.py -h` for details)

Returns: "Grayscale Image Written To: /path/to/outfile"

## Data Summary
Calculates the min, max, mean and sample standard deviation of the number of COVID cases in a given state.

Usage: `python summary.py -s "florida"` (run `python summary.py -h` for details)

Returns:

```
Stats for FLORIDA:
	- Min: 2.0
	- Max: 5420755.0
	- Mean: 1777642.8149210904
	- Standard Deviation: 1388586.922729319
```