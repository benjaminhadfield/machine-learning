# Gaussian Process

## Table of Contents

 - [Terminology](https://github.com/benjaminhadfield/machine-learning/tree/master/src/gaussian_process#terminology)
  - [Introduction](https://github.com/benjaminhadfield/machine-learning/tree/master/src/gaussian_process#introduction)
  - [Describing a Gaussian Process](https://github.com/benjaminhadfield/machine-learning/tree/master/src/gaussian_process#describing-a-gaussian-process)
  - [Acknowledgements](https://github.com/benjaminhadfield/machine-learning/tree/master/src/gaussian_process#acknowledgements)

## Terminology

Before we begin, it's important to understand the following terms:
 
 - [Gaussian distribution](https://github.com/benjaminhadfield/machine-learning/tree/master/src/gaussian_process#gaussian-distribution)
 - [Multivariate normal distribution](https://github.com/benjaminhadfield/machine-learning/tree/master/src/gaussian_process#multivariate-normal-distribution)

### Gaussian Distribution

The gaussian distribution, also called the normal distribution, is a continuous probability distribution that is common in both theory and practice.

A random variable 𝑋 with a normal distribution is described with two parameters:
 - the mean 𝜇
 - the variance 𝜎²
 
We write 𝑋 ~ 𝒩(𝜇, 𝜎²).
 
The distribution is always symmetric about the mean. Approximately, 95% of the area of the distribution is within two standard deviations of the mean.

As the standard deviation 𝜎 increases, the peak at 𝜇 becomes lower and more area is distributed to the tails.

![Gaussian distributions.](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/700px-Normal_Distribution_PDF.svg.png)

### Multivariate normal distribution

This is a generalisation on the (univariate) normal distribution, applied to higher dimensions.

![Sample points in a multivariate normal distribution.](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/MultivariateNormal.png/600px-MultivariateNormal.png)

The random variable 𝑍 with a standard normal distribution is described by 𝑍 ~ 𝒩(0, 1).

## Introduction

A gaussian process (GP) is often referred to as the infinite-dimensional extension of the multi-variate normal distribution. However, in practice we typically don't work with random variables with infinitely many components. When we work with GPs, the thinking is that we observe some finite-dimensional subset of infinite-dimensional data, and that this finite subset follows a multivariate normal distribution. We call this finite subset the **finite dimensional distribution**.

Okay... but what does this actually mean?

Well, imagine if we measure the temperature in a certain shop every day at 10am for 6-weeks. This would yield a (6 × 7 =) **42** dimensional vector:

 - { 𝑥₁, 𝑥₂, ..., 𝑥₄₂ }, _this is the finite dimensional distribution_.

In reality, however, the temperature is continuous, and and our choice to only take a measurement at 10am is completely arbitrary. How would the data change if we took measurements at 10pm instead? If we model this data with a GP, we are making the assumption that each of these possible data collection schemes would yield data from a multivariate normal distribution.

As a result of this, it makes sense to think of GPs as functions.

Formally, we'd say that a function 𝑓 is a GP if any set of finite values 𝑓(𝑥₁), 𝑓(𝑥₂), ..., 𝑓(𝑥ₙ) has a [multivariate normal distribution](https://github.com/benjaminhadfield/machine-learning/tree/master/src/gaussian_process#multivariate-normal-distribution) where our inputs {𝑥₁, 𝑥₂, ..., 𝑥ₙ} correspond to objects (usually vectors) from some arbitrarily sized domain (in our case, that domain is the temperature of the shop at 10am over the 6-week period, and is the finite subset of the infinite domain of all temperatures in the shop).

So in our temperature example, { 𝑥₁, 𝑥₂, ..., 𝑥₄₂ } corresponds to days in a 6-week period, and 𝑓(𝑥ₙ) indicates the temperature at 10am on day 𝑛.

## Describing a Gaussian Process

### Parameters

A GP is specified by two functions:
 
 - a mean function, 𝑚(𝑥)
 - a covariance function, also called a kernel, 𝑘(𝑥, 𝑥ʹ)
 
### Effect on the output
 
The shape and smoothness of the function is determined by the covariance function, as it controls the correlation between all pairs of output values. That is, for any given 𝑥 and 𝑥ʹ, 𝑚(𝑥) = 𝔼[𝑓(𝑥)], where 𝔼[𝑓(𝑥)] denotes the expectation of 𝑓(𝑥)., and 𝑘(𝑥, 𝑥ʹ) = 𝐶𝑜𝑣(𝑓(𝑥), 𝑓(𝑥)ʹ).

Thus if 𝑘(𝑥, 𝑥ʹ) is large when 𝑥 and 𝑥ʹ are near each other, the function will be more smooth, whilst smaller kernel values imply a more jagged function.

So, given a mean function 𝑓(𝑥) and a kernel 𝑘(𝑥, 𝑥ʹ), we can sample from any GP.

Lets say we want to evaluate the function at 𝑵 inputs, each of which has dimension 𝑫. Firstly, we create a matrix 𝑿 ∊ ℝᴺᴰ, where each row corresponds to an input we would like to sample from. We can then evaluate the mean function 𝑓(𝑥) at all inputs, denoted by 𝒎𝐱, which is a vector of length 𝑵, and the kernel matrix corresponding to 𝑿, denoted by 𝑲𝐱𝐱.

More generally, for any two sets of input data, 𝑿 and 𝑿ʹ, we define 𝑲𝐱𝐱′ to be the matrix where the (𝑖, 𝑗) element is 𝒌(𝑥ᵢ, 𝑥ⱼ). Finally, we can sample a random vector 𝑓 from a multivariate normal distribution: 𝑓 ~ 𝒩(𝒎𝐱, 𝑲𝐱𝐱). By construction, 𝔼(𝑓(𝑥ₙ)) = 𝒎(𝑥ₙ) for all 𝑛 and 𝐶𝑜𝑣(𝑓(𝑥ₙ), 𝑓(𝑥ₘ)ʹ) = 𝒌(𝒙ₙ, 𝒙ₘ) for all pairs 𝑛, 𝑚. Because this vector has a multivariate normal distribution, all subsets also follow a multivariate distribution, thereby fulfilling the definition of a GP.

### Kernels

A kernel must be a positive-definite function that maps two inputs, 𝑥 and 𝑥ʹ, to a scalar, such that 𝑲𝐱𝐱 is a valid covariance matrix.

For example, consider the single-dimensional inputs {𝑥ₙ} with a constant mean function at 0 and the following kernel:

```
𝑘(𝑥, 𝑥ʹ) = ℎ²(1 + ( (𝑥 － 𝑥ʹ)² / 2α𝑙² )) ^ (－α)
```

Where ℎ, α, 𝑙 are positive hyperparameters ∊ ℝ. This is known as the **rational quadratic covariance** function (RQ). Note that it only depends on the inputs via their difference (𝑥 － 𝑥ʹ), meaning the shape of the function is constant throughout the input space. Further, as 𝑥 and 𝑥ʹ get closer together, the covariance is larger, resulting in continuity.

Some examples of the rational quadratic kernel function across various kernel parameters are shown below.

![Rational quadratic kernel across various kernel parameters.](http://keyonvafa.com/assets/images/gp_predictit_blog/gp_samples.png)

## Acknowledgements

> Please remember, this repo is intended as a personal revision guide and not an original tutorial, and as such some content is very similar to other sources which I found helpful in understanding this topic. 😇

My introduction to GPs was borrowed heavily from [keyonvafa.com](http://keyonvafa.com/gp-tutorial/), and I thought it would be helpful to repeat many of the same examples here.
