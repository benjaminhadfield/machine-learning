# Gaussian Process

## Table of Contents

 - [Terminology](https://github.com/benjaminhadfield/machine-learning/tree/master/src/gaussian_process#terminology)
  - [Introduction](https://github.com/benjaminhadfield/machine-learning/tree/master/src/gaussian_process#introduction)

## Terminology

Before we begin, it's important to understand the following terms:
 
 - [Gaussian distribution](https://github.com/benjaminhadfield/machine-learning/tree/master/src/gaussian_process#gaussian-distribution)
 - [Multivariate normal distribution](https://github.com/benjaminhadfield/machine-learning/tree/master/src/gaussian_process#multivariate-normal-distribution)

### Gaussian Distribution

The gaussian distribution, also called the normal distribution, is a continuous probability distribution that is common in both theory and practice.

A random variable 𝑋 with a normal distribution is described with two parameters:
 - the mean 𝜇
 - the variance 𝜎²
 
We write 𝑋 ~ 𝑁(𝜇, 𝜎²).
 
The distribution is always symmetric about the mean. Approximately, 95% of the area of the distribution is within two standard deviations of the mean.

As the standard deviation 𝜎 increases, the peak at 𝜇 becomes lower and more area is distributed to the tails.

![Gaussian distributions.](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/700px-Normal_Distribution_PDF.svg.png)

### Multivariate normal distribution

This is a generalisation on the (univariate) normal distribution, applied to higher dimensions.

![Sample points in a multivariate normal distribution.](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/MultivariateNormal.png/600px-MultivariateNormal.png)

The random variable 𝑍 with a standard normal distribution is described by 𝑍 ~ 𝑁(0, 1).

## Introduction

A gaussian process is often referred to as the infinite-dimensional extension of the multi-variate normal distribution. However, in practice we typically don't work with random variables with infinitely many components. When we work with GPs, the thinking is that we observe some finite-dimensional subset of infinite-dimensional data, and that this finite subset follows a multivariate normal distribution. We call this finite subset the **finite dimensional distribution**.

Okay... but what does this actually mean?

Well, imagine if we measure the temperature in a certain shop every day at 10am for 6-weeks. This would yield a (6 × 7 =) **42** dimensional vector:

 - { 𝑥₁, 𝑥₂, ..., 𝑥₄₂ }, _this is the finite dimensional distribution_.

In reality, however, the temperature is continuous, and and our choice to only take a measurement at 10am is completely arbitrary. How would the data change if we took measurements at 10pm instead? If we model this data with a GP, we are making the assumption that each of these possible data collection schemes would yield data from a multivariate normal distribution.

As a result of this, it makes sense to think of gaussian processes as a function.

Formally, we'd say that a function 𝑓 is a gaussian process if any set of finite values 𝑓(𝑥₁), 𝑓(𝑥₂), ..., 𝑓(𝑥ₙ) has a [multivariate normal distribution](https://github.com/benjaminhadfield/machine-learning/tree/master/src/gaussian_process#multivariate-normal-distribution) where our inputs {𝑥₁, 𝑥₂, ..., 𝑥ₙ} correspond to objects (usually vectors) from some arbitrarily sized domain.

So in our temperature example, { 𝑥₁, 𝑥₂, ..., 𝑥₄₂ } corresponds to days in a 6-week period, and 𝑓(𝑥ₙ) indicates the temperature at 10am on day 𝑛.

## Acknowledgements

> Please remember, this repo is intended as a personal revision guide and not an original tutorial, and as such some content is very similar to other sources which I found helpful in understanding this topic. 😇

My introduction to GPs was borrowed heavily from [keyonvafa.com](http://keyonvafa.com/gp-tutorial/), and I thought it would be helpful to repeat many of the same examples here.
