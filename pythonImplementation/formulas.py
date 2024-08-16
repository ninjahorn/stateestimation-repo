import numpy as np
import matplotlib.pyplot as plt


def init_random_matrix(size_x, size_y=None):
    if size_y is None:
        size_y = size_x
    return np.random.rand(size_x, size_y)

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

def plot_innovation(innovation_history):
    plt.figure(figsize=(10, 6))
    plt.plot(innovation_history, marker='o', linestyle='-', color='r')
    
    plt.title('Innovation Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Innovation (α²)')
    plt.grid(True)
    plt.show()


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

def kalman_filter_sequential(state_dim, measurement_dim, iterations):
    # Initialize known matrices
    A = init_random_matrix(state_dim, state_dim)  # State transition matrix
    G = init_random_matrix(measurement_dim, state_dim)  # Measurement matrix
    Q = init_random_matrix(state_dim, state_dim)  # Process noise covariance
    R = init_random_matrix(measurement_dim, measurement_dim)  # Measurement noise covariance

    # Initialize state and covariance
    x = np.zeros((state_dim, 1))  # Initial state estimate (close to zero)
    P = np.zeros((state_dim, state_dim))  # Initial error covariance

    # Lists to store history for visualization
    state_history = []
    innovation_history = []
    covariance_history = []

    for k in range(1, iterations + 1):
        # Generate simulated measurement
        y = init_random_matrix(measurement_dim, 1)

        # Prediction step
        x_pred = A @ x  # Eq. (1.2.5)
        P_pred = A @ P @ A.T + Q  # Eq. (1.2.6)

        # Update step
        y_pred = G @ x_pred
        S = G @ P_pred @ G.T + R
        U = np.linalg.inv(S)  # Eq. (1.2.9)
        K = P_pred @ G.T @ U  # Eq. (1.2.8)
        
        innovation = y - y_pred  # Eq. (1.2.7)
        x = x_pred + K @ innovation  # Eq. (1.2.2)
        P = (np.eye(state_dim) - K @ G) @ P_pred  # Eq. (1.2.3)

        # Calculate α² (innovation)
        alpha_squared = innovation.T @ U @ innovation  # Eq. (1.2.4)


        # Store results for visualization
        state_history.append(x.flatten())  # Flatten to 1D array for easier handling
        innovation_history.append(alpha_squared[0][0])
        covariance_history.append(np.diag(P))  # Store only diagonal elements

        print(f"Iteration {k}:")
        print(f"Estimated state (x̂): \n{x}")
        print(f"Estimated covariance (P̂): \n{P}")
        print(f"Innovation (α²): \n{alpha_squared[0][0]}")
        print("--------------------")

        

    # Plot the results after the loop
    plot_estimated_state(state_history)
    plot_innovation(innovation_history)
    plot_covariance_diagonal(covariance_history)

    return x, P, alpha_squared[0][0]

# Run the Kalman filter simulation
state_dim = 8  # As mentioned in the notes, 8-dimensional state vector
measurement_dim = 8  # Assuming a smaller dimension for measurements
num_iterations = 10

kalman_filter_sequential(state_dim, measurement_dim, num_iterations)