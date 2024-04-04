import numpy as np

# from utils import plot_lines

A = np.array(
    object=[
        [-1, 3],
        [3, 2]
    ], dtype=np.dtype("float")
)

b = np.array(object=[7, 1], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)

print(f"Shape of A: {A.shape}")
print(f"Shape of b: {b.shape}")

x = np.linalg.solve(A, b)
print(f"Solution: {x}")

d = np.linalg.det(A)
print(f"Determinant of matrix A: {d:.2f}")

A_system = np.hstack((A, b.reshape(2, 1)))
print(A_system)
print(A_system[0][1])

A_2 = np.array(
    object=[
        [-1, 3],
        [3, -9]
    ],
    dtype=np.dtype(float),
)
b_2 = np.array(
    object=[[7, 1]],
    dtype=np.dtype(float),
)
d_2 = np.linalg.det(A_2)
print(f"Determinant of matrix A_2: {d_2:.2f}")

# try:
#     x_2 = np.linalg.solve(A_2, b_2)
# except np.linalg.LinAlgError as err:
#     print(err)

A_2_system = np.hstack((A_2, b_2.reshape((2, 1))))
print(A_2_system)

A_3 = np.array(
    object=[
        [1,2,3],
        [0,2,2],
        [1,4,5]
    ],
    dtype=np.dtype(float)
)
d_3 = np.linalg.det(A_3)
print(f"Determinant of matrix A_2: {d_3:.2f}")