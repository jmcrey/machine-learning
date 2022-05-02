import numpy as np
from nltk.lm import MLE, Laplace
from ngram import Ngram, read


def test_unigram() -> None:
    corpus = [
        ['This', 'is', 'the', 'first', 'sentence', '.'],
        ['This', 'sentence', 'is', 'the', 'second', 'sentence', '.'],
        ['And', 'this', 'is', 'the', 'third', 'one', '.'],
        ['Is', 'this', 'the', 'first', 'sentence', '?']
    ]

    x = Ngram(1)
    x.fit(corpus)
    actual = [x.perplexity([s]) for s in corpus]

    y = MLE(1)
    train_corpus = [list(x._ngrams(s)) for s in corpus]
    y.fit(train_corpus, x.vocab)
    expected = [y.perplexity(s) for s in train_corpus]
    assert all([np.isclose(actual[i], expected[i]) for i in range(len(actual))])


def test_bigram() -> None:
    corpus = [
        ['This', 'is', 'the', 'first', 'sentence', '.'],
        ['This', 'sentence', 'is', 'the', 'second', 'sentence', '.'],
        ['And', 'this', 'is', 'the', 'third', 'one', '.'],
        ['Is', 'this', 'the', 'first', 'sentence', '?']
    ]

    x = Ngram(2)
    x.fit(corpus)
    actual = [x.perplexity([s]) for s in corpus]

    y = MLE(2)
    train_corpus = [list(x._ngrams(s)) for s in corpus]
    y.fit(train_corpus, x.vocab)
    expected = [y.perplexity(s) for s in train_corpus]
    assert all([np.isclose(actual[i], expected[i]) for i in range(len(actual))])


def test_trigram() -> None:
    corpus = [
        ['This', 'is', 'the', 'first', 'sentence', '.'],
        ['This', 'sentence', 'is', 'the', 'second', 'sentence', '.'],
        ['And', 'this', 'is', 'the', 'third', 'one', '.'],
        ['Is', 'this', 'the', 'first', 'sentence', '?']
    ]

    x = Ngram(3)
    x.fit(corpus)
    actual = [x.perplexity([s]) for s in corpus]

    y = MLE(3)
    train_corpus = [list(x._ngrams(s)) for s in corpus]
    y.fit(train_corpus, x.vocab)
    expected = [y.perplexity(s) for s in train_corpus]
    assert all([np.isclose(actual[i], expected[i]) for i in range(len(actual))])


def test_generate() -> None:
    corpus = [
        ['This', 'is', 'the', 'first', 'sentence', '.'],
        ['This', 'sentence', 'is', 'the', 'second', 'sentence', '.'],
        ['And', 'this', 'is', 'the', 'third', 'one', '.'],
        ['Is', 'this', 'the', 'first', 'sentence', '?']
    ]

    x = Ngram(3)
    x.fit(corpus)
    text = x.npredict(25)
    print(text)

# def test_perplexity() -> None:
#     corpus = [
#         ['This', 'is', 'the', 'first', 'sentence', '.'],
#         ['This', 'sentence', 'is', 'the', 'second', 'sentence', '.'],
#         ['And', 'this', 'is', 'the', 'third', 'one', '.'],
#         ['Is', 'this', 'the', 'first', 'sentence', '?']
#     ]
#     model = Ngram(2)
#     model.fit(corpus)
#     actual = model.perplexity(corpus)
#     expected = 
