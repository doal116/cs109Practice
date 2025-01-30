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

    data_points = generate_data_point(20)
    estimatedMean, estimatedVar, log_likelihood = mle.normal_without_tweak(
        data_points)
    print(data_points)
    print(estimatedVar, estimatedMean, log_likelihood)


if __name__ == "__main__":
    main()
