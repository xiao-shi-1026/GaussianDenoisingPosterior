import numpy as np
from scipy.stats import multivariate_normal

def generate_mu_list(M: int, D: int) -> list:
    """
    Generate a list of M mean vectors for a D-dimensional Gaussian Mixture Model (GMM).
    
    Parameters:
        M (int): Number of Gaussian components in the mixture.
        D (int): Dimensionality of the data.
    
    Returns:
        list of np.array: List of mean vectors for each Gaussian component.
    """
    return [np.random.uniform(0, 10, size=(D,)) for _ in range(M)]

def generate_sigma_list(M: int, D: int) -> list:
    """
    Generate a list of M sigma vectors for a D-dimensional Gaussian Mixture Model (GMM).

    Parameters:
        M (int): Number of Gaussian components in the mixture.
        D (int): Dimensionality of the data.

    Returns:
        list of np.array: List of covariance matrices for each Gaussian component.
    """

    A = A = np.random.uniform(0, 2, (D, D))
    return [A @ A.T + np.eye(D) for _ in range(M)]


def sample_multivariate_gaussian_mixture(mus: np.array, sigmas: np.array, N: int = 10000, weights: np.array = None) -> np.array:
    """
    Sample data points from a Gaussian Mixture Model (GMM) using the Component Selection Method.
    
    This function first generates samples from all individual multivariate Gaussian distributions 
    and then selects one component per data point based on the given mixture weights.

    Sampling process:
    1. Generate an array of shape (N, d, M), where each column represents samples drawn from 
       a different Gaussian component.
    2. Randomly select one component for each data point according to the specified `weights`.
    3. Return the selected samples, ensuring that the final output has the shape (N, d).

    Parameters:
        mus (list of np.array): List of mean vectors for each Gaussian component, 
                                where each element is a (d,) numpy array.
        sigmas (list of np.array): List of covariance matrices for each Gaussian component, 
                                   where each element is a (d, d) numpy array.
        N (int): Number of data points to sample.
        weights (list or np.array, optional): Probability weights for selecting each Gaussian 
                                              component. If None, all components are assumed 
                                              to be equally weighted (1/M).

    Returns:
        np.array: An array of shape (N, d), where each row is a sampled data point.
    """

    M = len(mus)
    
    samples = np.zeros((N, len(mus[0]), M), dtype=np.float64)
    for i in range(M):
        samples[:, :, i] = np.random.multivariate_normal(mus[i], sigmas[i], size=(N,)).astype(np.float64)

    if weights is None:
        weights = np.ones(M, dtype=np.float64) / M
    random_idx = np.random.choice(np.arange(M), size=(N,), p=weights)
    return samples[np.arange(N), :, random_idx]

import numpy as np

def mu1_real(y: np.array, A: np.array, means: list, sigmas: list, Nsigma: np.array) -> np.array:
    """
    Calculates the theoretical posterior E(x|y) of an M-component GMM with transformation y = Ax + e.
    Parameters:
        y: np.array(d, N)          - Observations (d-dimensional, N samples)
        A: np.array(d, d)          - Transformation matrix in y = Ax + e
        means: list of np.array(d, 1)  - List of mean vectors of M Gaussian components
        sigmas: list of np.array(d, d) - List of covariance matrices for M Gaussian components
        Nsigma: np.array(d, d)     - Covariance matrix of noise (diagonal elements)
        
    Returns:
        E[x | y]: np.array(d, N)  - Posterior expected value of x given y
    """
    y = y.astype(np.float64)
    M = len(means)  # Number of Gaussian components
    dy, N = y.shape
    dx = A.shape[1]
    mu = np.zeros((M, dx, N))  # Store posterior means
    p = np.zeros((M, N))  # Store likelihoods

    for i in range(M):
        inv_term = np.linalg.inv(A @ sigmas[i] @ A.T + Nsigma)  # Inverse covariance
        mu[i] = means[i].reshape(-1, 1) + sigmas[i] @ A.T @ inv_term @ (y - A @ means[i].reshape(-1, 1))  # Posterior mean
        p[i] = np.exp(-0.5 * np.diag((y - A @ means[i].reshape(-1, 1)).T @ inv_term @ (y - A @ means[i].reshape(-1, 1))))  # Likelihood

    p_sum = np.sum(p, axis=0)
    p_sum[p_sum == 0] = 1e-10  # Avoid division by zero

    mu_weighted = mu * p[:, None, :]  # Ensure correct broadcasting
    return np.sum(mu_weighted, axis=0) / p_sum

def px_pdf_real(xs: np.array, means: list, sigmas: list, weights: np.array = None) -> np.array:
    """
    Calculate the pdf of an M-component Gaussian Mixture Model (GMM).
    xs should have the same dimensionality as the means and sigmas.

    Parameters:
        xs: np.array(d, N)         - Data points (d-dimensional, N samples)
        means: list of np.array(d, 1)  - List of mean vectors of M Gaussian components
        sigmas: list of np.array(d, d) - List of covariance matrices for M Gaussian components
        weights: np.array(M,)      - Mixing weights for each component (default: uniform)
    
    Returns:
        np.array(N,) - The probability density function evaluated at each sample point.
    """
    xs = xs.astype(np.float64)
    M = len(means)  # Number of Gaussian components

    pdf = np.zeros((M, xs.shape[1]))  # Store pdfs

    for i in range(M):
        pdf[i] = multivariate_normal.pdf(xs.T, mean=means[i].flatten(), cov=sigmas[i])

    if weights is None: # default weights
        weights = np.ones(M, dtype=np.float64) / M

    total_pdf = np.sum(pdf * weights[:, None], axis=0)

    # Calculate the mean of the pdf
    mu_gmm = np.sum(np.array(means) * weights[:, None], axis=0)

    return total_pdf, mu_gmm

def generate_noise(level: float, D: int, N: int) -> tuple:
    """
    Generate noise with given level of variance.
    
    Parameters:
        level (float): Variance of the noise.
        D (int): Dimensionality of the data.
        N (int): Number of samples to generate.

    Returns:
        Nsigma: np.array. Noise matrix in shape of (D, D).
        noise: np.array. Noise vector in shape of (D,).
    """
    Nsigma = level * np.eye(D)
    noise = np.random.multivariate_normal(np.array([0 for i in range(D)]), Nsigma, (N,)).astype(np.float64).T
    return Nsigma, noise

def moment_calculation(y: np.array, A: np.array, means: list, sigmas: list, Nsigma: np.array, n_ev, iters = 100, c = 1e-5):
    """
    Calculate the projected first and second moments of the posterior distribution of a GMM using the power method.
    
    Params:
        y: np.array(d, N)          - Observations (d-dimensional, N samples)
        A: np.array(d, d)          - Transformation matrix in y = Ax + e
        means: list of np.array(d, 1)  - List of mean vectors of M Gaussian components
        sigmas: list of np.array(d, d) - List of covariance matrices for M Gaussian components
        Nsigma: np.array(d, d)     - Covariance matrix of noise (diagonal elements)
        iters: int                 - Number of power iterations
        n_ev: int                  - Number of eigenvectors to compute
        c: float                   - Scaling factor for power iterations

    returns:
        first_moments: np.array(d, n_ev) - First moments of the posterior distribution
        second_moments: np.array(n_ev,)  - Second moments of the posterior distribution
        eigvecs: np.array(d, n_ev)       - Eigenvectors of the posterior distribution
        eigvals: np.array(n_ev,)         - Eigenvalues of the posterior distribution
        mmse: np.array(d, N)             - Minimum mean square error estimate of x
    """

    dx = A.shape[1]
    # A_invT = np.linalg.pinv(A.T)
    # print(A_invT.shape)
    mmse = mu1_real(y, A, means, sigmas, Nsigma)
    eigvecs = np.random.randn(dx, n_ev) * np.sqrt(np.mean(np.diag(Nsigma)))
    Ab = np.zeros((dx, n_ev))

    for i in range(iters):
        for j in range(n_ev):

            out = mu1_real(y + A @ eigvecs[:,j].reshape(-1, 1), A, means, sigmas, Nsigma)

            Ab[:,j] = (out - mmse).T

        if n_ev > 1:
            norm_of_Ab = np.linalg.norm(Ab, axis=0)
            eigvecs = Ab / norm_of_Ab

            Q, _ = np.linalg.qr(eigvecs.astype(np.float64), mode='reduced')
            eigvecs = Q
        else:
            norm_of_Ab = np.linalg.norm(Ab.ravel())
            eigvecs = Ab / norm_of_Ab
        
        eigvecs *= c

    eigvals = (norm_of_Ab / c * np.mean(np.diag(Nsigma))).reshape(n_ev, )
    eigvecs /= c

    # QR decomposition does not gaurantee that eigenvectors are sorted.
    sort_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    first_moments = eigvecs.T @ mmse

    second_moments = np.abs(eigvals)

    return first_moments, second_moments, eigvecs, eigvals, mmse

def x_given_y_v(alpha, Yt, means, sigmas, sigmaN, v, weights = None):
    """
    计算 M 组件 GMM 在给定观测 Yt 后，在方向 v 上的投影概率密度。

    参数:
        alpha: np.array (N,) - 需要计算的投影点 (1D)
        Yt: np.array (d, N) - 观测数据 (d维, N个样本)
        means: list[np.array(d,)] - GMM 每个分量的均值 (M 个 d 维向量)
        sigmas: list[np.array(d, d)] - GMM 每个分量的协方差 (M 个 d×d 矩阵)
        sigmaN: np.array (d, d) - 观测噪声协方差矩阵
        v: np.array (d,) - 方向向量 (d 维)
        weights: np.array (M,) - GMM 每个分量的混合权重 (M 个值，且 sum(weights) = 1)

    返回:
        out: np.array (N,) - 在方向 v 上的投影概率密度
    """
    Y = Yt.T  # 转置以适配 NumPy 计算
    M = len(means)  # GMM 组件数
    d = Y.shape[1]  # 维度
    N = alpha.shape[0]  # 需要计算的投影点个数
    if weights is None:
        weights = np.ones(len(means)) / len(means)  # 设为均匀分布

    # 计算 p(Y | C_i) (先验观测概率)
    p_y_given_C = np.zeros((M, Y.shape[0]))
    for i in range(M):
        cov_i = sigmas[i] + sigmaN
        p_y_given_C[i] = multivariate_normal.pdf(Y, mean=means[i], cov=cov_i)

    # 计算 p(C_i | Y) (后验概率)
    weighted_p_y_given_C = p_y_given_C * weights[:, None]  # 乘以 GMM 先验权重
    p_C_given_Y = weighted_p_y_given_C / np.sum(weighted_p_y_given_C, axis=0, keepdims=True)

    # 计算投影后的 1D 高斯参数 (均值 & 方差)
    mu_x_given_y = np.zeros((M, Y.shape[0]))  # 存储投影均值
    sigma_x_given_y = np.zeros(M)  # 存储投影方差

    for i in range(M):
        inv_cov_i = np.linalg.inv(sigmas[i] + sigmaN)
        mu_x_given_y[i] = np.dot(v, means[i].reshape(-1, 1) + sigmas[i] @ inv_cov_i @ (Y.T - means[i].reshape(-1, 1)))
        sigma_x_given_y[i] = np.dot(v, (sigmas[i] - sigmas[i] @ inv_cov_i @ sigmas[i]) @ v)

    # 计算投影后的 1D GMM 概率密度
    p_x_given_y = np.zeros((M, Y.shape[0], N))
    for i in range(M):
        p_x_given_y[i] = multivariate_normal.pdf(alpha[:, None], mean=mu_x_given_y[i], cov=sigma_x_given_y[i])

    # 计算最终的加权后验概率密度
    out = np.sum(p_x_given_y * p_C_given_Y[:, :, None], axis=0)  # (N,)

    return out