import time
import numpy as np
from algorithmen import determinant, inverse, multiplyMatrix, initRandArray, transpose, gaussJordanInverse

# start timer for execution time
start = time.time()

# Create a 3x3 random matrix
matrix = np.array(initRandArray(3))
print("Random Matrix:")
print(matrix)

# Calculate determinant
det = determinant(matrix)
print("\nDeterminant:")
print(det)

# Calculate inverse using the original inverse function
inv = inverse(matrix)
print("\nInverse (using original inverse function):")
print(inv)

# Calculate inverse using Gauss-Jordan method
gauss_jordan_inv = gaussJordanInverse(matrix)
print("\nInverse (using Gauss-Jordan method):")
print(gauss_jordan_inv)

# Transpose the matrix
trans = transpose(matrix)
print("\nTransposed Matrix:")
print(trans)

# Create another random matrix for multiplication
matrix2 = np.array(initRandArray(3))
print("\nSecond Random Matrix:")
print(matrix2)

# Multiply matrices
result = multiplyMatrix(matrix, matrix2)
print("\nMatrix multiplication result:")
print(result)

# Verify the inverse by multiplying the original matrix with its inverse
identity_check = multiplyMatrix(matrix, inv)
print("\nOriginal matrix multiplied by its inverse (should be close to identity matrix):")
print(identity_check)

# Compare the results of the two inverse methods
inv_difference = np.abs(inv - gauss_jordan_inv)
print("\nDifference between the two inverse methods:")
print(inv_difference)
print("Maximum difference:", np.max(inv_difference))

# end timer for execution time
end = time.time()
print(f"Execution time: {end - start} seconds")