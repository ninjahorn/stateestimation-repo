import random
from multiprocessing import Pool, cpu_count
import math
import numpy as np
import matplotlib.pyplot as plt

# Hilfsfunktion, die eine Zeile der Größe sizeY mit Zufallswerten erzeugt
def generate_line(sizeY):
    line = []
    for _ in range(sizeY):
        zahl = random.randint(0, 100) / 10.0
        if zahl == 0:
            zahl = 0.1
        line.append(zahl)
    return line

# Erzeugt eine Matrix der Größe sizeX x sizeY mit Zufallswerten
def init_rand_matrix_parallel(sizeX, sizeY=None):
    if sizeY is None:
        sizeY = sizeX
    with Pool(processes=cpu_count()) as pool:
        matrix = pool.starmap(generate_line, [(sizeY,)] * sizeX)
    return matrix

# Hilfsfunktion für die Multiplikation von Matrizen
def _matmul_chunk(A_rows, B, start_idx, end_idx):
    colsB = len(B[0]) if B else 0
    partial_result = []
    for row in A_rows:
        new_row = []
        for col_index in range(colsB):
            val = 0.0
            for k in range(len(row)):
                val += row[k] * B[k][col_index]
            new_row.append(val)
        partial_result.append(new_row)
    return (start_idx, end_idx, partial_result)

# Multipliziert die Matrizen A und B
def parallel_matmul(A, B):
    m = len(A)
    if m == 0 or len(B) == 0:
        return []

    num_proc = cpu_count()
    chunk_size = max(1, math.ceil(m / num_proc))
    tasks = []
    start_idx = 0
    while start_idx < m:
        end_idx = min(start_idx + chunk_size, m)
        A_chunk = A[start_idx:end_idx]
        tasks.append((A_chunk, B, start_idx, end_idx))
        start_idx = end_idx

    with Pool(processes=num_proc) as pool:
        results = pool.starmap(_matmul_chunk, tasks)

    results.sort(key=lambda x: x[0])
    C = []
    for (s_idx, e_idx, partial_result) in results:
        C.extend(partial_result)

    return C

# Hilfsfunktionen für die Addition und Subtraktion von Matrizen
def _add_rows(rowA, rowB):
    return [a + b for a, b in zip(rowA, rowB)]

def _sub_rows(rowA, rowB):
    return [a - b for a, b in zip(rowA, rowB)]

# Addiert die Matrix B zur Matrix A
def parallel_matadd(A, B):
    assert len(A) == len(B), "A und B müssen gleiche Anzahl Zeilen haben"
    assert len(A[0]) == len(B[0]), "A und B müssen gleiche Anzahl Spalten haben"
    with Pool(processes=cpu_count()) as pool:
        C = pool.starmap(_add_rows, zip(A, B))
    return C

# Subtrahiert die Matrix B von der Matrix A
def parallel_matsub(A, B):
    assert len(A) == len(B), "A und B müssen gleiche Anzahl Zeilen haben"
    assert len(A[0]) == len(B[0]), "A und B müssen gleiche Anzahl Spalten haben"
    with Pool(processes=cpu_count()) as pool:
        C = pool.starmap(_sub_rows, zip(A, B))
    return C

# Liest die col_index-te Spalte von A heraus und gibt sie als Liste zurück (wird für die Transponierung benötigt)
def _transpose_extract_col(A, col_index):
    return [row[col_index] for row in A]

# Transponiert die Matrix A
def parallel_transpose(A):
    if not A:
        return []
    m = len(A)
    n = len(A[0])
    with Pool(processes=cpu_count()) as pool:
        columns = pool.starmap(_transpose_extract_col, [(A, i) for i in range(n)])
    return columns

# Hilfsfunktion für die Inversion: Aktualisiert eine Zeile der augmentierten Matrix
def _gauss_jordan_inversion_worker(row, pivot_row, pivot_val, pivot_index):
    if row is pivot_row:
        return row
    factor = row[pivot_index] / pivot_val
    new_row = [r - factor * p for r, p in zip(row, pivot_row)]
    return new_row

# Berechnet die Inverse der Matrix A mittels Gauss-Jordan
def parallel_matinv(A):
    n = len(A)
    if n == 0 or len(A[0]) != n:
        raise ValueError("Matrix muss quadratisch sein.")
    aug = []
    for i in range(n):
        row = A[i][:] + [0.0]*n
        row[n+i] = 1.0
        aug.append(row)
    # Gauss-Jordan
    for pivot_index in range(n):
        pivot_val = aug[pivot_index][pivot_index]
        if abs(pivot_val) < 1e-12:
            raise ValueError("ERROR: Matrix scheinbar singulär (Pivot ~ 0).")
        # Pivot-Zeile auf 1 normieren
        aug[pivot_index] = [x / pivot_val for x in aug[pivot_index]]
        pivot_row = aug[pivot_index]
        # Alle anderen Zeilen parallel updaten
        with Pool(processes=cpu_count()) as pool:
            updated_rows = pool.starmap(
                _gauss_jordan_inversion_worker,
                [(row, pivot_row, 1.0, pivot_index) for row in aug]
            )
        aug = updated_rows
    invA = []
    for i in range(n):
        invA.append(aug[i][n:])
    return invA

# Erzeugt eine Einheitsmatrix der Größe d
def identity_matrix(d):
    return [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]

# Plotfunktion, um den Verlauf des geschätzten Zustands über die Iterationen zu visualisieren
def plot_estimated_state(state_history):
    arr = np.array(state_history)  # => shape: (iterations, dimension)
    num_states = arr.shape[1]

    plt.figure(figsize=(12, 6))
    for i in range(num_states):
        plt.plot(arr[:, i], label=f'State {i+1}')
    
    plt.title("Estimated State Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("State Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("./parallelPlot/estimated_state.png")

# Plotfunktion, um den Verlauf der Innovationsgröße α² zu visualisieren
def plot_innovation(innovation_history):
    plt.figure(figsize=(10, 4))
    plt.plot(innovation_history, marker='o', linestyle='-', color='r')
    plt.title("Innovation (α²) Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("α²")
    plt.grid(True)
    plt.savefig("./parallelPlot/innovation.png")

# Plotfunktion, um den Verlauf der Diagonale der Kovarianzmatrix zu visualisieren
def plot_covariance_diagonal(covariance_history):
    arr = np.array(covariance_history)
    num_states = arr.shape[1]

    plt.figure(figsize=(12, 6))
    for i in range(num_states):
        plt.plot(arr[:, i], label=f'Variance of State {i+1}')
    plt.title("Covariance Diagonal Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)
    plt.savefig("./parallelPlot/covariance.png")


# =========== Filter Parallel ===========
# - Führt den Filter über 'iterations' Zeitschritte aus.
# - Alle Operationen werden parallel via multiprocessing durchgeführt.
# - 'dimension' definiert die Größe von Zustand und Messung (in diesem Beispiel sind beide gleich).

def filter_parallel(dimension, iterations):
    # -------------------------------------------------
    # Anfangswerte:
    #  x(0|0) = 0,  P(0|0) = P0,  α²(0|0) = 0 (Formel 1.2.10)
    # -------------------------------------------------
    x = [[0.0] for _ in range(dimension)]  # Vektor (d x 1)
    P = [[0.0]*dimension for _ in range(dimension)]  # (d x d), hier als 0-Matrix
    alpha_sq_prev = 0.0  # Entspricht α²(0|0)=0
    
    # Zufällige Matrizen A, G, Q, R
    A = init_rand_matrix_parallel(dimension, dimension)
    G = init_rand_matrix_parallel(dimension, dimension)
    Q = init_rand_matrix_parallel(dimension, dimension)
    R = init_rand_matrix_parallel(dimension, dimension)
    
    # Verlaufsspeicher für Debugging und Visualisierung
    state_history = []
    alpha_history = []
    covariance_history = []

    for k in range(1, iterations + 1):

        # Messvektor y
        y = init_rand_matrix_parallel(dimension, 1)

        # (1.2.5)
        x_pred = parallel_matmul(A, x)

        # (1.2.6)
        AP = parallel_matmul(A, P)
        A_T = parallel_transpose(A)
        APA_T = parallel_matmul(AP, A_T)
        P_pred = parallel_matadd(APA_T, Q)

        # (1.2.7)
        y_pred = parallel_matmul(G, x_pred)
        innovation = parallel_matsub(y, y_pred)

        # (1.2.9)
        GP = parallel_matmul(G, P_pred)
        G_T = parallel_transpose(G)
        GPGT = parallel_matmul(GP, G_T)
        S = parallel_matadd(GPGT, R)
        U = parallel_matinv(S)

        # (1.2.8)
        P_pred_GT = parallel_matmul(P_pred, G_T)
        K = parallel_matmul(P_pred_GT, U)

        # (1.2.2)
        K_innov = parallel_matmul(K, innovation)
        x = parallel_matadd(x_pred, K_innov)

        # (1.2.3)
        KG = parallel_matmul(K, G)
        I = identity_matrix(dimension)
        I_minus_KG = []
        for i_row in range(dimension):
            row_ = [I[i_row][j] - KG[i_row][j] for j in range(dimension)]
            I_minus_KG.append(row_)
        P = parallel_matmul(I_minus_KG, P_pred)

        # (1.2.4)
        innov_T = parallel_transpose(innovation)
        tmp = parallel_matmul(innov_T, U)
        alpha_mat = parallel_matmul(tmp, innovation)
        alpha_sq_current = alpha_sq_prev + alpha_mat[0][0]

        # Speichern für nächsten Schritt
        alpha_sq_prev = alpha_sq_current

        # Debug/Logging
        state_history.append([val[0] for val in x])
        alpha_history.append(alpha_sq_current)
        
        # Diagonale von P anhängen (für Kovarianz-Plot)
        diag_P = [P[i][i] for i in range(dimension)]
        covariance_history.append(diag_P)

        print(f"Iteration {k}")
        print(f"x(k|k): {x}")
        print(f"P(k|k) diag: {[P[i][i] for i in range(dimension)]}")
        print(f"alpha^2(k|k): {alpha_sq_current}")
        print("------------------------------")

    # Plotten (Ende der Iterationen)
    # plot_estimated_state(state_history)
    # plot_innovation(alpha_history)
    # plot_covariance_diagonal(covariance_history)

    # Zurückgeben vom letzten Zustand dem letzten alpha^2
    return x, P, alpha_sq_current

# if __name__ == "__main__":
#     final_x, final_P, final_alpha = filter_parallel(dimension=8, iterations=50)
#     print("\n=== ENDERGEBNIS ===")
#     print("Finaler Zustand x:", final_x)
#     print("Diag von P:", [final_P[i][i] for i in range(4)])
#     print("Letztes alpha^2:", final_alpha)