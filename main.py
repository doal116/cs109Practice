import matplotlib.pyplot as plt
import mle
import random


def generate_data_point(size: int) -> list[float]:
    data_points = [
        round(random.uniform(1, size), 1)
        for _ in range(1, size)
    ]
    return data_points


def main():
    # y = generate_data_point(10)
    # x = [x for x in range(1, 10)]
    # plt.plot(x, y, alpha=0.7, color="blue")
    # plt.show()

    # data_points = generate_data_point(6)
    data_points = [6.3, 5.5, 5.4, 7.1, 4.6, 6.7, 5.3, 4.8, 5.6,
                   3.4, 5.4, 3.4, 4.8, 7.9, 4.6, 7.0, 2.9, 6.4, 6.0, 4.3]
    estimatedMean, estimatedVar, log_likelihoodMle = mle.normal(
        data_points)
    grad_asc_mean, grad_asc_var, log_likelihood = mle.normal_with_gradient_ascent(
        data_points,0.1)
    print(data_points)
    print(estimatedMean, estimatedVar, log_likelihoodMle)
    print(grad_asc_mean, grad_asc_var, log_likelihood)


if __name__ == "__main__":
    main()
