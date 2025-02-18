import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Hilfsfunktion, die eine Zeile mit Zufallswerten erzeugt
def generate_line_sequential(sizeY):
    line = []
    for _ in range(sizeY):
        zahl = random.randint(0, 100) / 10.0
        if zahl == 0:
            zahl = 0.1
        line.append(zahl)
    return line

# Erzeugt eine Zufallsmatrix der Größe sizeX x sizeY
def init_rand_matrix_sequential(sizeX, sizeY=None):
    if sizeY is None:
        sizeY = sizeX
    matrix = []
    for _ in range(sizeX):
        row = generate_line_sequential(sizeY)
        matrix.append(row)
    return matrix

# Matrixmultiplikation in sequentieller Form
def sequential_matmul(A, B):
    m = len(A)
    if m == 0 or len(B) == 0:
        return []

    n = len(A[0])
    p = len(B[0])
    C = [[0.0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            val = 0.0
            for k in range(n):
                val += A[i][k] * B[k][j]
            C[i][j] = val
    return C

# Matrixaddition
def sequential_matadd(A, B):
    m = len(A)
    n = len(A[0])
    C = []
    for i in range(m):
        row = [A[i][j] + B[i][j] for j in range(n)]
        C.append(row)
    return C

# Matrixsubtraktion
def sequential_matsub(A, B):
    m = len(A)
    n = len(A[0])
    C = []
    for i in range(m):
        row = [A[i][j] - B[i][j] for j in range(n)]
        C.append(row)
    return C

# Transponieren einer Matrix
def sequential_transpose(A):
    if not A:
        return []
    m = len(A)
    n = len(A[0])
    AT = []
    for j in range(n):
        col = []
        for i in range(m):
            col.append(A[i][j])
        AT.append(col)
    return AT

# Hilfsfunktion, aktualiisiert die Matrix aug, indem alle Zeilen außer pivot_row "bereinigt" werden
def _gauss_jordan_inversion_worker_seq(aug, pivot_index, pivot_row):
    pivot_val = pivot_row[pivot_index]
    new_aug = []
    for row in aug:
        # check ob es die pivot-row selbst ist:
        if row is pivot_row:
            new_aug.append(row)
        else:
            factor = row[pivot_index] / pivot_val
            new_row = [r - factor * p for r, p in zip(row, pivot_row)]
            new_aug.append(new_row)
    return new_aug

# Invertiert eine Matrix, mit Gauss-Jordan
def sequential_matinv(A):
    n = len(A)
    if n == 0 or len(A[0]) != n:
        raise ValueError("Matrix muss quadratisch sein.")
    
    aug = []
    for i in range(n):
        row = A[i][:] + [0.0]*n
        row[n + i] = 1.0
        aug.append(row)

    # Gauss-Jordan
    for pivot_index in range(n):
        pivot_val = aug[pivot_index][pivot_index]
        if abs(pivot_val) < 1e-12:
            raise ValueError("ERROR: Matrix scheinbar singulär (Pivot ~ 0).")
        # Normiere Pivot-Zeile
        aug[pivot_index] = [x / pivot_val for x in aug[pivot_index]]
        pivot_row = aug[pivot_index]

        aug = _gauss_jordan_inversion_worker_seq(aug, pivot_index, pivot_row)

    invA = []
    for i in range(n):
        invA.append(aug[i][n:])
    return invA

# Erzeugt eine Einheitsmatrix der Größe d
def identity_matrix(d):
    return [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]


def plot_estimated_state(state_history):
    arr = np.array(state_history)  
    num_states = arr.shape[1]

    plt.figure(figsize=(12, 6))
    for i in range(num_states):
        plt.plot(arr[:, i], label=f'State {i+1}')
    
    plt.title("Estimated State Over Time (Sequential)")
    plt.xlabel("Iteration")
    plt.ylabel("State Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("./sequentialPlot/estimated_state.png")

def plot_innovation(innovation_history):
    plt.figure(figsize=(10, 4))
    plt.plot(innovation_history, marker='o', linestyle='-', color='r')
    plt.title("Innovation (α²) Over Time (Sequential)")
    plt.xlabel("Iteration")
    plt.ylabel("α²")
    plt.grid(True)
    plt.savefig("./sequentialPlot/innovation.png")

def plot_covariance_diagonal(covariance_history):
    arr = np.array(covariance_history)
    num_states = arr.shape[1]

    plt.figure(figsize=(12, 6))
    for i in range(num_states):
        plt.plot(arr[:, i], label=f'Variance of State {i+1}')
    plt.title("Covariance Diagonal Over Time (Sequential)")
    plt.xlabel("Iteration")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)
    plt.savefig("./sequentialPlot/covariance.png")
    
# Filter SEQUENTIELL (Rückgabe: x, P, alpha_sq_last)
def filter_sequential(dimension, iterations):

    # Startwerte
    x = [[0.0] for _ in range(dimension)]
    P = [[0.0]*dimension for _ in range(dimension)]
    alpha_sq_prev = 0.0

    # Generierung der "Zufallsmatrizen" (hier sequentiell)
    A = init_rand_matrix_sequential(dimension, dimension)
    G = init_rand_matrix_sequential(dimension, dimension)
    Q = init_rand_matrix_sequential(dimension, dimension)
    R = init_rand_matrix_sequential(dimension, dimension)

    # Verlauf für Visualisierung
    state_history = []
    alpha_history = []
    covariance_history = []

    for k in range(1, iterations + 1):
        y = init_rand_matrix_sequential(dimension, 1)

        # (1.2.5)
        x_pred = sequential_matmul(A, x)

        # (1.2.6)
        AP = sequential_matmul(A, P)
        A_T = sequential_transpose(A)
        APA_T = sequential_matmul(AP, A_T)
        P_pred = sequential_matadd(APA_T, Q)

        # (1.2.7)
        y_pred = sequential_matmul(G, x_pred)
        innovation = sequential_matsub(y, y_pred)

        # (1.2.9)
        GP = sequential_matmul(G, P_pred)
        G_T = sequential_transpose(G)
        GPGT = sequential_matmul(GP, G_T)
        S = sequential_matadd(GPGT, R)
        U = sequential_matinv(S)

        # (1.2.8)
        P_pred_GT = sequential_matmul(P_pred, G_T)
        K = sequential_matmul(P_pred_GT, U)

        # (1.2.2)
        K_innov = sequential_matmul(K, innovation)
        x = sequential_matadd(x_pred, K_innov)

        # (1.2.3)
        KG = sequential_matmul(K, G)
        I = identity_matrix(dimension)
        I_minus_KG = []
        for i_row in range(dimension):
            row_ = [I[i_row][j] - KG[i_row][j] for j in range(dimension)]
            I_minus_KG.append(row_)
        P = sequential_matmul(I_minus_KG, P_pred)

        # (1.2.4)
        innov_T = sequential_transpose(innovation)
        tmp = sequential_matmul(innov_T, U)
        alpha_mat = sequential_matmul(tmp, innovation)
        alpha_sq_current = alpha_sq_prev + alpha_mat[0][0]
        alpha_sq_prev = alpha_sq_current

        state_history.append([val[0] for val in x])
        alpha_history.append(alpha_sq_current)

        diagP = [P[i][i] for i in range(dimension)]
        covariance_history.append(diagP)

        print(f"Iteration {k}")
        print(f"  x(k|k) = {x}")
        print(f"  P(k|k) diag = {[P[i][i] for i in range(dimension)]}")
        print(f"  alpha^2(k|k) = {alpha_sq_current}")
        print("------------")

    # Ausgabe der Visualisierungen
    # plot_estimated_state(state_history)
    # plot_innovation(alpha_history)
    # plot_covariance_diagonal(covariance_history)

    return x, P, alpha_sq_current

# if __name__ == "__main__":
#     final_x, final_P, final_alpha = filter_sequential(dimension=8, iterations=100)
#     print("\n=== SEQUENTIELLES ENDERGEBNIS ===")
#     print("Finaler Zustand x:", final_x)
#     print("Diag von P:", [final_P[i][i] for i in range(len(final_P))])
#     print("Letztes alpha^2:", final_alpha)