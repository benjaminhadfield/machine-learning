from src.linear_regression.model import LinearRegression


def main():
    print('Starting...')
    lr = LinearRegression('src/data/test_scores.csv')
    line_eqn = lr.learn()
    print('y = {0}x + {1}'.format(*line_eqn))


if __name__ == '__main__':
    main()
