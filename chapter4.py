import numpy as np
import pandas as pd

# Topic Vector simple demo
topic = {}
tfidf = dict(list(zip('cat dog apple lion NYC love'.split(), np.random.rand(6))))
# print(tfidf)
# {'cat': 0.3591248619179701, 'dog': 0.8957950407993277, 'apple': 0.0957776660010744,
# 'lion': 0.38395029333039554, 'NYC': 0.4787884467254815, 'love': 0.9016759457609537}

# print(list(zip('cat dog apple lion NYC love'.split(), np.random.rand(6))))

'''LDA Modelling (Linear Discriminant Analysis
Breaks down a document into only one topic
Fast, Simple Dimension Reduction and Classification Algorithms
Train to find the vector line between two centroids for binary class,
'''

# Naive Bayes
'''
In statistics, naive Bayes classifiers are a family of simple "probabilistic
classifiers" based on applying Bayes' theorem with strong (naive) independence
assumptions between the features (see Bayes classifier). They are among the
simplest Bayesian network models, but coupled with kernel density estimation,
they can achieve high accuracy levels.

Naive Bayes classifiers are highly scalable, requiring a number of parameters
linear in the number of variables (features/predictors) in a learning problem.
Maximum-likelihood training can be done by evaluating a closed-form expression,
which takes linear time, rather than by expensive iterative approximation as used
for many other types of classifiers.

Naive Bayes is a simple technique for constructing classifiers: models that
assign class labels to problem instances, represented as vectors of feature
values, where the class labels are drawn from some finite set. There is not a
single algorithm for training such classifiers, but a family of algorithms based
on a common principle: all naive Bayes classifiers assume that the value of a
particular feature is independent of the value of any other feature, given the
class variable. For example, a fruit may be considered to be an apple if it is
red, round, and about 10 cm in diameter. A naive Bayes classifier considers each
of these features to contribute independently to the probability that this fruit
is an apple, regardless of any possible correlations between the color,
roundness, and diameter features.
'''

''' Latent Semantic Analysis
Latent semantic analysis is based on the oldest and most commonly-used technique
for dimension reduction, singular value decomposition. SVD was in widespread use
long before the term “machine learning” even existed.SVD decomposes a matrix
into three square matrices, one of which is diagonal.
'''
