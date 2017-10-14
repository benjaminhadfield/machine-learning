import numpy as np


class LinearRegression:
    def __init__(self, points_file):
        # (1) Import data.
        self.points = np.genfromtxt(points_file, delimiter=',')
        # (2) Set hyperparams.
        self.learning_rate = 0.0001
        # Using y = mx + c.
        self.initial_c = 0
        self.initial_m = 0
        self.iterations = 1000
        self._regression_for_line(self.points, self.initial_m, self.initial_c)

    @staticmethod
    def _regression_for_line(points, m, c):
        """
        Computes the total regression for the set of points and a given line.
        Eqn: `err := (1/N)(∑[1->N](yi - (mxi + b))^2)`
        """
        total_regression = 0
        for p in points:
            # Get x and y values for point.
            p_x, p_y = p
            # Calc regression (error) using `(yi - (mxi + b))^2`.
            total_regression += (p_y - (m * p_x + c)) ** 2
        # Return the total divided by N.
        return total_regression / len(points)

    @staticmethod
    def _step_gradient(points, m_initial, c_initial, learning_rate):
        m_gradient = c_gradient = 0
        for p in points:
            p_x, p_y = p
            # Get direction with respect to m and c (partial derivatives).
            m_gradient += -(2 / len(points)) \
                * p_x * (p_y - ((m_initial * p_x) + c_initial))
            c_gradient += -(2 / len(points)) \
                * (p_y - ((m_initial * p_x) + c_initial))

        m_new = m_initial - (learning_rate * m_gradient)
        c_new = c_initial - (learning_rate * c_gradient)
        return m_new, c_new

    @staticmethod
    def _gradient_descent_runner(points, initial_m, initial_c, learning_rate, iterations):
        m = initial_m
        c = initial_c
        for i in range(iterations):
            # Update m, c with more accurate values.
            m, c = LinearRegression._step_gradient(points, m, c, learning_rate)
        return m, c

    def learn(self):
        [m, c] = self._gradient_descent_runner(
            self.points,
            self.initial_m,
            self.initial_c,
            self.learning_rate,
            self.iterations)
        return m, c
