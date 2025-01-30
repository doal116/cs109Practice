import math


def normal(data: list[float]) -> tuple[float, float, float]:

    estimated_mean: float = sum(data) / len(data)

    estimated_var: float = sum(((x-estimated_mean)**2)
                               for x in data) / (len(data) - 1)

    log_likelihood: float = 0.0
    for x in data:
        log_likelihood += (
            -(math.log(math.sqrt(2*math.pi) * estimated_var)) -
            ((1/(2*estimated_var))*((x-estimated_mean)**2))
        )
    return estimated_mean, estimated_var, log_likelihood


def normal_with_gradient_ascent(data: list[float], step: float) -> tuple[float, float, float]:
    # Estimating mean through gradient ascent
    grad_asc_mean: float = step * (sum(data) / len(data))
    for x in data:
        gradient = sum((x-grad_asc_mean)for x in data)
        grad_asc_mean += step*(gradient/len(data))

    # Estimating variance through gradient ascent
    grad_asc_var: float = sum(
        (x-grad_asc_mean)**2 for x in data) / (len(data)-1)

    for _ in data:
        gradient = (
            -(len(data)/(2*grad_asc_var))
            + (
                (1/(2*(grad_asc_var**2)))
                * (sum((x-grad_asc_mean)**2 for x in data))
            )
        )
        grad_asc_var += (step * gradient)

    # Log Likelihood of estimated parameters
    log_likelihood: float = 0.0
    for x in data:
        log_likelihood += (
            -(math.log(math.sqrt(2*math.pi) * grad_asc_var)) -
            ((1/(2*grad_asc_mean))*(x-grad_asc_mean)**2)
        )
    return grad_asc_mean, grad_asc_var, log_likelihood
