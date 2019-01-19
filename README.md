# N-Gram-Language-Model

Includes:
- [x] Index words
- [x] Store ngrams in a Trie data structure
- [x] Efficiently extract ngrams and their frequencies
- [x] Compute out-of-vocabulary (OOV) rate
- [x] Compute ngram probabilities with absolute discounting with interpolation smoothing.
- [x] Compute Perplexity

## Introduction

A statistical language model is the development of probabilistic models to predict the probability of a sequence of
words. It is able to predict the next word in a sequence given a history context represented by the preceding words. 

The probability that we want to model can be factorized using the chain rule as follows:

<p align="center">
  <img src="http://latex.codecogs.com/gif.latex?p%28w_1%5EN%29%20%3D%20%5Cprod_%7Bn%3D1%7D%5E%7BN%7D%20p%28w_n%20%7C%20w_%7B0%7D%5E%7Bn-1%7D%29">
</p>

where ![equation](http://latex.codecogs.com/gif.latex?w_0) is a special token to denote the start of the sentence.

In practice, we usually use what is called N-Gram models that use Markov process assumption to limit the history context. Examples of N-Grams are:
<p align="center">
  <img src="http://latex.codecogs.com/gif.latex?%5C%5C%5Ctext%7BUnigram%20LM%7D%3A%20p%28w_1%5EN%29%20%3D%20%5Cprod_%7Bn%3D1%7D%5E%7BN%7D%20p%28w_n%29%20%5C%5C%20%5Ctext%7BBigram%20LM%7D%3A%20p%28w_1%5EN%29%20%3D%20%5Cprod_%7Bn%3D1%7D%5E%7BN%7D%20p%28w_n%20%7C%20w_%7Bn-1%7D%29%5C%5C%20%5Ctext%7BTrigram%20LM%7D%3A%20p%28w_1%5EN%29%20%3D%20%5Cprod_%7Bn%3D1%7D%5E%7BN%7D%20p%28w_n%20%7C%20w_%7Bn-2%7D%2C%20w_%7Bn-1%7D%29"
</p>

## Training

Using Maximum Likelihood criteria, these probabilities can be estimated using counts. For example, for the bigram model, 

![equation](https://latex.codecogs.com/gif.latex?p%28w_n%20%7C%20w_%7Bn-1%7D%29%20%3D%20%5Cdfrac%7BN%28w_%7Bn-1%7D%2C%20w_n%29%7D%7BN%28w_%7Bn-1%7D%29%7D)

![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7Bwhere%20%7D%20N%28w_%7Bn-1%7D%2C%20w_n%29%20%5Ctext%7B%20is%20the%20count%20of%20bigrams%20%7D%20%28w_%7Bn-1%7D%2C%20w_n%29%20%5Ctext%7B%20and%20%7D%20N%28w_%7Bn-1%7D%29%20%5Ctext%7B%20is%20the%20count%20of%20%7D%20w_%7Bn-1%7D)

However, this can be problamatic if we have unseen data because the counts will be 0 and thus the probability is undefined. To solve this problem, we use smoothing techniques. There are different smoothing techniques and the one that we used is called *absolute discounting with interpolation*. 

## Perplexity

To meausre the performance of a language model, we compute the perplexity of the test corpus which is:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?PP%20%3D%20%5Cbigg%5B%5Cprod_%7Bn%3D1%7D%5E%7BN%7D%20p%28w_n%20%7C%20w_%7Bn-1%7D%29%5Cbigg%5D%5E%7B-1/N%7D">
</p>

## Results

Model was tested on europarl dataset (dir `data`):

Test PP with bigrams = 130.09

Test PP with trigrams = 94.82
