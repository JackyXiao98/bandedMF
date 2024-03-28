import numpy as np


def rotation_matrix(n):
    """Change a upper triangular matrix to lower triangular matrix."""
    mat = np.zeros([n, n])
    for i in range(n):
        mat[i, n-1-i] = 1
    return mat


def moving_average(arr, window_size=5):
    padded_arr = np.pad(arr, (window_size // 2, window_size // 2), mode='edge')
    result = np.convolve(padded_arr, np.ones(window_size) / window_size, mode='valid')
    return result


def average_matrix(mat, b, k):
    """For each diagonal, use moving average to get a new array."""
    res = np.copy(mat)
    for i in range(b):
        vec = np.diag(mat, k=-i)
        avg_vec = moving_average(vec, k)
        m = len(vec)
        # print("i = ", i, vec)
        for j in range(m):
            res[j+i, j] = avg_vec[j]
    return res


def prefix_sum_matrix(n):
    """
    Create the lower triangular matrix with elements of 1.

    Return as the basis matrix.
    """
    mat_w = np.tril(np.ones([n, n]))
    return mat_w


if __name__ == "__main__":
    np.random.seed(0)
    mat = np.random.random([5, 5])
    res = average_matrix(mat, 3, 3)
    print("mat: \n", mat)
    print("res: \n", res)
