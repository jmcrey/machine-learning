# Homework 3: Ngrams

## ngram.py

Defines the classes for the Ngram models in three formats:

1. Standard relative frequency Ngram
2. Uniform distribution Ngram
3. Laplace smoothing ngram

The `main()` function runs all procedures required for this assignment.

Requirements: `pip3 install -r requirement.txt`
Usage: `python3 ngram.py`

Example Output:

```
###################################
## Results for the Uniform Model
##   - Ted: 62733.00000000012
##   - Reddit: 62733.00000000004
##   - News: 62733.00000000004
###################################
###################################
## Results for the Unigram Model
##   - Ted: 625.0723987438362
##   - Reddit: 1398.5167333773425
##   - News: 1457.1352403815106
###################################
###################################
## Results for the Ted Ngram
  Ted		Reddit		News
1 625.07		1398.52		1457.14
2 78.43		103.36		121.62
3 19.5		21.77		20.98
4 7.6		7.46		5.98
5 4.11		3.58		3.53
6 2.86		2.47		2.47
7 1.92		1.75		1.59
###################################
## Results for the Reddit Ngram
  Ted		Reddit		News
1 735.45		882.46		1345.97
2 73.9		65.07		93.27
3 16.38		11.82		13.54
4 4.88		4.88		3.57
5 2.32		2.78		1.79
6 1.15		1.68		1.0
/home/jmack/.pyenv/versions/hw3/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3474: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/home/jmack/.pyenv/versions/hw3/lib/python3.9/site-packages/numpy/core/_methods.py:189: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
7 nan		1.15		nan
###################################
## Results for the Ted Laplace
  Ted		Reddit		News
2 1473.27		4895.43		5584.48
3 13242.45		28474.37		31921.84
4 38240.24		52700.39		54610.4
5 54336.76		60604.09		61135.74
6 60107.1		62103.3		62178.44
7 61568.49		62058.16		62070.28
###################################
## Results for the Reddit Laplace
  Ted		Reddit		News
2 4478.48		4787.02		9735.99
3 25714.41		22796.62		34496.67
4 39783.64		35429.52		41906.03
5 42335.35		40149.14		42599.42
6 42340.84		41402.17		42357.96
7 41933.0		41335.09		41933.0
Perplexity for Ted Generated Text: 1.06085778432205
Perplexity for Reddit Generated Text: 1.0035202381128157
```

## Outputs

### Scores for Uniform Model

- Ted: 62733.00000000012
- Reddit: 62733.00000000004
- News: 62733.00000000004


### Scores for Unigram Model

- Ted: 625.0723987438362
- Reddit: 1398.5167333773425
- News: 1457.1352403815106

### Perplexity for the Ted Data (Using the Ngram Model)

![](outputs/Perplexity%20for%20Ted%20Data%20(Ngram).png)

### Perplexity for the Reddit Data (Using the Ngram Model)

![](outputs/Perplexity%20for%20Reddit%20Data%20(Ngram).png)

### Perplexity for the Ted Data (Using Laplace Smoothing)

![](outputs/Perplexity%20for%20Ted%20Data%20(Laplace).png)

### Perplexity for the Reddit Data (Using Laplace Smoothing)

![](outputs/Perplexity%20for%20Reddit%20Data%20(Laplace).png)

### Perplexity for Generated Text

- Perplexity for Ted Generated Text: 1.06085778432205
- Perplexity for Reddit Generated Text: 1.0035202381128157