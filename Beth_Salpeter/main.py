# TITLE: Loading libraries

import numpy as np
import scipy as sp
import time
import numba as nb


# TITLE: Defining functions

# kronecker_delta is python implementation of the Kronecker delta
@nb.jit(nopython=True)
def kronecker_delta(i, j):
    return 1 if i == j else 0


# The rank_reducer function reduces the rank of a rank 4 tensor to a rank 2 tensor mimicking
#  Flatten[a,{{1,3},{2,4}}] in Mathematica. It is used in the vectorizer function
def rank_reducer(matrix_of_matrices):
    # Convert the input to a NumPy array if it's not already
    if isinstance(matrix_of_matrices, list):
        matrix_of_matrices = np.array(matrix_of_matrices)

    # Get the dimensions of the input array
    num_rows = matrix_of_matrices.shape[0]
    num_cols = matrix_of_matrices.shape[1]

    # Get the dimensions of each matrix block
    block_rows, block_cols = matrix_of_matrices[0, 0].shape

    # Construct the flattened matrix
    flattened_matrix = np.zeros((num_rows * block_rows, num_cols * block_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            flattened_matrix[i * block_rows:(i + 1) * block_rows, j * block_cols:(j + 1) * block_cols] = \
                matrix_of_matrices[i, j]
    flattened_matrix = flattened_matrix.tolist()
    return flattened_matrix


# The mediator propagator
@nb.jit(nopython=True)
def deltaphi(p, q, z1, z2, m1, m2, MB, mPhi):
    term1 = (2 - ((2 * (m2 ** 2 + m1 ** 2 - MB ** 2) + mPhi ** 2) * np.arctanh(
        (2 * p * q * np.sqrt((1 - z1 ** 2) * (1 - z2 ** 2)))
        / (mPhi ** 2 + p ** 2 + q ** 2 - 2 * p * q * z1 * z2))) / (
                     p * q * np.sqrt((1 - z1 ** 2) * (1 - z2 ** 2))))
    return term1


# The outer leg propagator of particle 1.
@nb.jit(nopython=True)
def g1(q, z2, MB, m1, eta):
    return 1 / (-eta ** 2 * MB ** 2 + q ** 2 + m1 ** 2 + 2j * eta * MB * q * z2)


# The outer leg propagator of particle 2.
@nb.jit(nopython=True)
def g2(q, z2, MB, m2, eta):
    return 1 / (-(1 - eta) ** 2 * MB ** 2 + q ** 2 + m2 ** 2 - 2j * (1 - eta) * MB * q * z2)

@nb.jit(nopython=True)
def mycheb(a, z):
    if a == 0:
        return 1.0
    elif a == 1:
        return z
    else:
        return 2.0 * z * mycheb(a - 1, z) - mycheb(a - 2, z)

# TITLE: Numerical integrations

# The integrand of the kernel integral
def integrand(a, b, p, q, e, mPhi, m1, m2, MB, z1, z2):
    term1 = (e ** 2 * q ** 3) / (4 * (1 + kronecker_delta(a, 0)) * np.pi ** 4)
    term2 = np.sqrt(1 - z2 ** 2)
    term3 = g1(q, z2, MB, m1, m1 / (m1 + m2))
    term4 = g2(q, z2, MB, m2, m1 / (m1 + m2))
    term5 = mycheb(b, z2)
    term6 = mycheb(a, z1)
    term7 = np.sqrt(1 - z1 ** 2)
    term8 = deltaphi(p, q, z1, z2, m1, m2, MB, mPhi)
    return term1 * term2 * term3 * term4 * term5 * term6 / term7 * term8


# Here we execute the integral over the z1,z2 variables.
# Note that the syntax of {},{},... in the options refer to the options of z1, z2 ,... and that accuracy
# precision are defined as abs(i-result) <= max(epsabs, epsrel*abs(i)) !UNLIKE MATHEMATICA!
def kappa(a, b, p, q, mPhi, MB, m1, m2, e, precision, accuracy):
    def integrand_func(z1, z2):
        return integrand(a, b, p, q, e, mPhi, m1, m2, MB, z1, z2)

    opts = [{'points': [0], 'epsabs': accuracy, 'epsrel': precision}, {'epsabs': accuracy, 'epsrel': precision}]
    result1, _ = sp.integrate.nquad(integrand_func, [(-1, 1), (0, 1)], opts=opts)

    return 2 * np.real(result1)


# gaussian_quadrature_weights(min, maxval, n) works similarly to GaussianQuadratureWeights in Mathematica
def gaussian_quadrature_weights(minval, maxval, n):
    # Gauss-Legendre nodes and weights
    nodes, weights = np.polynomial.legendre.leggauss(n)

    # Rescale nodes and weights to the desired interval
    scaled_nodes = 0.5 * (maxval - minval) * nodes + 0.5 * (maxval + minval)
    scaled_weights = 0.5 * (maxval - minval) * weights

    return scaled_nodes, scaled_weights

# The vectorizer function creates the rank-reduced matrix representation of the kernel.
def vectorizer(mPhi, m1, m2, eb, pg, ag, min, max, nGauss, abmax):
    mu = (m1 * m2) / (m1 + m2)

    gauss_list = np.transpose(
        gaussian_quadrature_weights(min * np.sqrt(2 * mu * eb), max * np.sqrt(2 * mu * eb), nGauss))

    results1 = []
    for a in range(0, abmax + 2, 2):
        results2 = []
        for b in range(0, abmax + 2, 2):
            results3 = []
            for p in gauss_list:
                results4 = []
                for q in gauss_list:
                    results4.append(kappa(a, b, p[0], q[0], mPhi, m1 + m2 - eb, m1, m2, 1, pg, ag) * q[1])
                results3.append(results4)
            results2.append(results3)
        results1.append(results2)

    return rank_reducer(results1)





# TITLE: The analytical Coulomb energy levels

# Coulomb energy levels
def en(n, e, m1, m2):
    alpha = e ** 2 / (4 * np.pi)
    mu = (m1 * m2) / (m1 + m2)
    return -mu * alpha ** 2 / (2 * n ** 2)


# The energy finder finds the effective e coupling given a specific kernel and an input binding energy.
def energy_finder(abmax, nGauss, m1, m2, mPhi, eb, pg, ag, level, min, max):
    kappa_matrix = vectorizer(mPhi, m1, m2, eb, pg, ag, min, max, nGauss, abmax)
    etmp =1/np.sqrt(np.linalg.eigvals(kappa_matrix))[level-1]
    return etmp


start_time = time.time()
value = vectorizer(0.1, 1, 1, 0.1, 1e-2, 1e-2, 0, 1, 20, 2)

end_time = time.time()
duration = end_time - start_time
print("value:", value)
print("Duration:", duration, "seconds")  # takes about 40 secs. about 30 times slower than in Mathematica


