import numpy as np

from scipy.special import gamma
from scipy.stats import gamma as gamma_dist

def shi_probabilities(dim, alpha):
    """Compute the probabilities for the Shi distribution as described in
    Shi, 1995b, "Multivariate extreme value distribution and its fisher information matrix"

    The vector of probabilities p = (p_1, ..., p_d) is computed as follows:
        p_d,1 = gamma(d - alpha) / (gamma(d) * gamma(1 - alpha))
        p_d,j = ((d - 1 - alpha * j) * p_{d-1,j} + alpha * (j - 1) * p_{d-1,j-1}) / (d - 1) for j = 2, ..., d - 1
        p_d,d = alpha^d

    Args:
        dim (int): Dimension of the distribution
        alpha (float): Shape parameter

    Returns:
        np.ndarray: Array of probabilities
    """

    p = np.zeros(dim)
    p[0] = gamma(dim - alpha) / (gamma(dim) * gamma(1 - alpha))

    if dim == 1:  return p
    if dim == 2:
        p[1] = alpha
        return p

    q = shi_probabilities(dim - 1, alpha)
    for j in range(2, dim):
        p[j - 1] = ((dim - 1 - alpha * j) * q[j - 1] + alpha * (j - 1) * q[j - 2]) / (dim - 1)

    p[dim - 1] = alpha ** dim

    return p

def shi_cumulative_probabilities(dim, alpha):
    """Compute the cumulative probabilities for the Shi distribution

    Args:
        dim (int): Dimension of the distribution
        alpha (float): Shape parameter

            Returns:
        np.ndarray: Array of cumulative probabilities"""
    return np.cumsum(shi_probabilities(dim, alpha))



def generate_multivariate_logistic(n_samples, dim, alpha=1.0):
    """Generate samples from the multivariate logistic distribution as described in
    Stephenson, 2003, "Simulation of multivariate extreme value distributions of logistic type"

    Args:
        n_samples (int): Number of samples to generate
        dim (int): Dimension of the distribution
        alpha (float): Shape parameter

    Returns:
        np.ndarray: Array of samples
    """

    W = np.random.exponential(size=(n_samples, dim))
    SW = np.sum(W, axis=1)
    T = W / SW[:, None]

    U = np.random.uniform(size=n_samples)
    P = shi_cumulative_probabilities(dim, alpha)

    Z = np.zeros(n_samples)
    for i, u in enumerate(U):
        for k, p in enumerate(P):
            if u < p:
                break

        Z[i] = gamma_dist.rvs(k, scale=1)

    X = 1 / (Z[:, None] * T ** alpha)

    return X

def generate_bivariate_logistic(n_samples, alpha):
    """Generate samples from the bivariate logistic distribution as described in
    Stephenson, 2003, "Simulation of multivariate extreme value distributions of logistic type"

    Args:
        n_samples (int): Number of samples to generate
        dim (int): Dimension of the distribution
        alpha (float): Shape parameter

    Returns:
        np.ndarray: Array of samples
    """
    W = np.random.exponential(size=(n_samples, 2))
    SW = np.sum(W, axis=1)
    T = W / SW[:, None]

    Z = np.random.gamma(shape=2, scale=1.0, size=n_samples)

    X = np.zeros((n_samples, 2))
    for t, z in zip(T, Z):
        X += 1 / (z * t ** alpha)

    return X


def positive_stable_distribution(n_samples, alpha=1.0):
    """Generate samples from the positive stable distribution
    Args:
        n_samples (int): Number of samples to generate
        alpha (float): Shape parameter

    Returns:
        np.ndarray: Array of samples
    """
    U = np.random.uniform(size=n_samples, low=0, high=np.pi)
    W = np.random.exponential(size=n_samples)

    if alpha == 1.0:
        S = np.sin(alpha*U) / (np.sin(U)**(1/alpha))
    else:
        A = (np.sin((1-alpha)*U) / W)
        S = A ** ((1-alpha)/alpha) * np.sin(alpha*U) / (np.sin(U)**(1/alpha))
    return S


def multivariate_logistic_distribution(n_samples, dim, alpha=1.0):
    """Generate samples from the multivariate logistic distribution as described in
    Stephenson, 2003, "Simulation of multivariate extreme value distributions of logistic type"

    Args:
        n_samples (int): Number of samples to generate
        dim (int): Dimension of the distribution
        alpha (float): Shape parameter

    Returns:
        np.ndarray: Array of samples
    """
    S = positive_stable_distribution(n_samples, alpha=alpha)[:, np.newaxis]#.repeat(dim).reshape(n_samples, dim)
    W = np.random.exponential(size=(n_samples, dim))
    X = (S / W) ** alpha
    return X