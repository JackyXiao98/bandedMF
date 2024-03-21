import numpy as np


def rotation_matrix(n):
    """Change a upper triangular matrix to lower triangular matrix."""
    mat = np.zeros([n, n])
    for i in range(n):
        mat[i, n-1-i] = 1
    return mat


if __name__ == "__main__":
    n = 4
    mat = rotation_matrix(n)
    print(mat)
