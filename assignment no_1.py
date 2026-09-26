import numpy as np

# Create NumPy array
arr = np.array([10, 20, 30, 40, 50, 60])

print("Original Array:")
print(arr)

# Indexing
print("\nIndexing:")
print("First element:", arr[0])
print("Third element:", arr[2])

# Slicing
print("\nSlicing:")
print("Elements from index 1 to 4:", arr[1:5])

# Reshaping
print("\nReshaped Array:")
new_arr = arr.reshape(2, 3)
print(new_arr)

# Vectorized operations
print("\nVectorized Operations:")
print("Addition:", arr + 5)
print("Multiplication:", arr * 2)
print("Square:", arr ** 2)

# Numerical computations
print("\nNumerical Computations:")
print("Sum:", np.sum(arr))
print("Mean:", np.mean(arr))
print("Maximum:", np.max(arr))
print("Minimum:", np.min(arr))
