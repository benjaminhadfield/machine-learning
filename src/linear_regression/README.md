# Linear Regression

Linear Regression is a statistical method used to find a linear relationship between a dependent variable, and one or more independent variables.

There are several approaches to finding a linear relationship - the most common method, and one implemented, is to find a line `L` that minimises the difference between the observed and predicted values (residules). This approach (the [least squares](https://en.wikipedia.org/wiki/Least_squares) approach) works by calculating a regression `R` such that `R` is the sum of squares of all residules. `L` is then iteratively updated (e.g. using [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)) in such a way that `R` is reduced each time until it cannot be reduced further. At that point `L` is the line-of-best-fit for the given data, and can be used to predict unknown values.
