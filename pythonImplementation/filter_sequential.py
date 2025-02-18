import numpy as np
import matplotlib.pyplot as plt


# Generieren einer zufälligen Matrix
def init_random_matrix(size_x, size_y=None):
    if size_y is None:
        size_y = size_x
    return np.random.rand(size_x, size_y)

# Darstellung der Zustände (Schätzung)
def plot_estimated_state(state_history):
    state_history = np.array(state_history)  # Convert to a numpy array for easier indexing
    num_states = state_history.shape[1]
    
    plt.figure(figsize=(12, 8))
    for i in range(num_states):
        plt.plot(state_history[:, i], label=f'State {i+1}')
    
    plt.title('Estimated State Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('State Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Darstellung der Innovationen (Messabweichungen)
def plot_innovation(innovation_history):
    plt.figure(figsize=(10, 6))
    plt.plot(innovation_history, marker='o', linestyle='-', color='r')
    
    plt.title('Innovation Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Innovation (α²)')
    plt.grid(True)
    plt.show()

# Darstellung der Kovarianzmatrix (Messunsicherheit)
def plot_covariance_diagonal(covariance_history):
    covariance_history = np.array(covariance_history)
    num_states = covariance_history.shape[1]
    
    plt.figure(figsize=(12, 8))
    for i in range(num_states):
        plt.plot(covariance_history[:, i], label=f'Variance of State {i+1}')
    
    plt.title('State Estimation Uncertainty Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)
    plt.show()

# Implementierung des der Formeln für den Filter
def filter_sequential(state_dim, measurement_dim, iterations):
    
    # Initialisierung der Matrizen
    A = init_random_matrix(state_dim, state_dim)
    G = init_random_matrix(measurement_dim, state_dim)
    Q = init_random_matrix(state_dim, state_dim)
    R = init_random_matrix(measurement_dim, measurement_dim)

    # Initialisierung der Zustands- und Kovarianzmatrix (setze auf 0)
    x = np.zeros((state_dim, 1))
    P = np.zeros((state_dim, state_dim))

    # Initialisierung der Listen für die Visualisierung
    state_history = []
    innovation_history = []
    covariance_history = []

    # Iterationen des Filters
    for k in range(1, iterations + 1):
        # Generiere zufällige Messung
        y = init_random_matrix(measurement_dim, 1)

        # Prediction
        x_pred = A @ x  # (1.2.5)
        P_pred = A @ P @ A.T + Q  # (1.2.6)

        # Update
        y_pred = G @ x_pred
        S = G @ P_pred @ G.T + R
        U = np.linalg.inv(S)  # (1.2.9)
        K = P_pred @ G.T @ U  # (1.2.8)
        
        # Innovation
        innovation = y - y_pred  # (1.2.7)
        x = x_pred + K @ innovation  # (1.2.2)
        P = (np.eye(state_dim) - K @ G) @ P_pred  # (1.2.3)
        alpha_squared = innovation.T @ U @ innovation  # (1.2.4)

        # Speichern der Werte für die Visualisierung
        state_history.append(x.flatten())
        innovation_history.append(alpha_squared[0][0])
        covariance_history.append(np.diag(P))

        # Ausgabe der Ergebnisse
        print(f"Iteration {k}:")
        print(f"Estimated state (x̂): \n{x}")
        print(f"Estimated covariance (P̂): \n{P}")
        print(f"Innovation (α²): \n{alpha_squared[0][0]}")
        print("--------------------")

    # Visualisierung der Ergebnisse
    plot_estimated_state(state_history)
    plot_innovation(innovation_history)
    plot_covariance_diagonal(covariance_history)

    # Rückgabe der finalen Werte
    return x, P, alpha_squared[0][0]

# Test der Implementierung mit 8-dimensionalen Zuständen und Messungen bei 10 Iterationen (Variablen anpassbar)
state_dim = 8
measurement_dim = 8
num_iterations = 10
filter_sequential(state_dim, measurement_dim, num_iterations)