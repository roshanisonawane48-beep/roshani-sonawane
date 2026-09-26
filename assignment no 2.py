import numpy as np

# Matrix creation
A = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

B = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

# Matrix Addition
print("\nMatrix Addition (A + B):")
print(A + B)

# Matrix Subtraction
print("\nMatrix Subtraction (A - B):")
print(A - B)

# Matrix Multiplication
print("\nMatrix Multiplication (A @ B):")
print(A @ B)

# Transpose
print("\nTranspose of A:")
print(A.T)

# Determinant
print("\nDeterminant of A:")
print(np.linalg.det(A))

# Mean
print("\nMean of A:")
print(np.mean(A))

# Median
print("\nMedian of A:")
print(np.median(A))

# Standard Deviation
print("\nStandard Deviation of A:")
print(np.std(A))

# Variance
print("\nVariance of A:")
print(np.var(A))

# Minimum and Maximum
print("\nMinimum value:")
print(np.min(A))

print("\nMaximum value:")
print(np.max(A))

# Sum
print("\nSum of all elements:")
print(np.sum(A))

# Row-wise mean
print("\nRow-wise Mean:")
print(np.mean(A, axis=1))

# Column-wise mean
print("\nColumn-wise Mean:")
print(np.mean(A, axis=0))
