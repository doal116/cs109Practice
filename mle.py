import math


def normal_without_tweak(data: list[float]) -> tuple[float, float, float]:

    estimated_mean: float = sum(data) / len(data)

    estimated_var: float = 0.0
    for x in data:
        estimated_var += (x-estimated_mean)**2
    estimated_var /= len(data)

    log_likelihood: float = 0.0
    for x in data:
        log_likelihood += (
            -(math.log(math.sqrt(2*math.pi * estimated_var))) -
            ((1/(2*estimated_var))*(x-estimated_mean)**2)
        )
    return estimated_mean, estimated_var, log_likelihood
