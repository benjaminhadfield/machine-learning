# Linear Regression

Linear Regression is a statistical method used to find a linear relationship between a dependent variable, and one or more independent variables.

There are several approaches to finding a linear relationship - the most common method, and one implemented, is to find a line `L` that minimises the difference between the observed and predicted values (residules). This approach (the [least squares](https://en.wikipedia.org/wiki/Least_squares) approach) works by calculating a regression `R` such that `R` is the sum of squares of all residules. `L` is then iteratively updated (e.g. using [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)) in such a way that `R` is reduced each time until it cannot be reduced further. At that point `L` is the line-of-best-fit for the given data, and can be used to predict unknown values.

## Gradient Descent

When finding a simple linear-regression, there are three important variables to track:

 - `m` the gradient of the line-of-best-fit.
 - `c` the y-intercept of the line-of-best-fit, `m` and `c` together describes the line's equation.
 - `R` the sum of all residules - the cost function and the value, with respect to `m` and `c`, that we are aiming to minimise.

If we plotted these three variables, we would get a three-dimensional surface showing how `R` changes with respect to `m` and `c`. The idea of gradient descent is to follow the slope of this surface down in `R`'s direction so that we arrive at some optimal point where `R` is minimised. Then, the line-of-best-fit is trivially calculated using the corrosponding `m` and `c` values.

![Plot of `m`, `c` and `R`.](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/35389/versions/1/screenshot.png)

To find the direction of travel that reduces the cost function, we take partial derivatives with respect to `m` and `c`. By travelling proportinally to the negative gradient the cost functiontion is iteratively reduced until finally a minimum is reached.
