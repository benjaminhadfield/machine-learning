import numpy as np


class LinearRegression:
    def __init__(self, points_file):
        # (1) Import data.
        self.points = np.genfromtxt(points_file, delimiter=',')
        # Using `y = mx + c`.
        self.initial_c = 0
        self.initial_m = 0
        self._regression_for_line()

    def _regression_for_line(self):
        """
        Computes the total regression for the set of points and a given line.
        Eqn: `err := (1/N)(∑[1->N](yi - (mxi + b))^2)`
        """
        total_regression = 0
        for p in self.points:
            # Get x and y values for point.
            p_x, p_y = p
            # Calc regression (error) using `(yi - (mxi + b))^2`.
            total_regression += (p_y - (self.initial_m * p_x + self.initial_c)) ** 2
        # Return the total divided by N.
        return total_regression / len(self.points)

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

    def _gradient_descent_runner(self, learning_rate, iterations):
        m = self.initial_m
        c = self.initial_c
        for i in range(iterations):
            # Update m, c with more accurate values.
            m, c = self._step_gradient(self.points, m, c, learning_rate)
        return m, c

    def learn(self, learning_rate=0.001, iterations=1000):
        [m, c] = self._gradient_descent_runner(
            learning_rate,
            iterations)
        return m, c
