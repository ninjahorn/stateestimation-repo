import time
import matplotlib.pyplot as plt

# Import filter implementations
from filter_sequential import filter_sequential
from filter_parallel import filter_parallel

def scenario1_fixed_dimension():
    fixed_dim = 10
    iteration_values = [10, 50, 200, 1000]  # 4 iteration values

    # Create a figure with 2x2 subplots => 4 subplots total
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Scenario 1: FIXED dimension={fixed_dim}, varying iterations", fontsize=14)

    # Iterate over iteration_values and place each result into a subplot
    subplot_positions = [(0,0), (0,1), (1,0), (1,1)]

    for i, iters in enumerate(iteration_values):
        row, col = subplot_positions[i]
        ax = axes[row][col]

        # Measure SEQUENTIAL
        start_seq = time.time()
        _, _, alpha_seq = filter_sequential(fixed_dim, iters)
        seq_time = time.time() - start_seq

        # Measure PARALLEL
        start_par = time.time()
        _, _, alpha_par = filter_parallel(fixed_dim, iters)
        par_time = time.time() - start_par

        # We'll do a simple bar plot with 2 bars
        labels = ["Sequential", "Parallel"]
        times = [seq_time, par_time]

        ax.bar(labels, times, color=["blue", "orange"])

        # Title referencing iteration
        speedup = seq_time / par_time if par_time != 0 else float('inf')
        ax.set_title(f"iters={iters}\nSeq={seq_time:.3f}s, Par={par_time:.3f}s\nSpeedup={speedup:.2f}")

        ax.set_ylabel("Time (s)")
        ax.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig("./comparePlot/scenario1.png", dpi=150)


def scenario2_fixed_iterations():
    fixed_iters = 50
    dimension_values = [10, 50, 100, 500]  # 4 dimension values

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Scenario 2: FIXED iterations={fixed_iters}, varying dimensions", fontsize=14)

    subplot_positions = [(0,0), (0,1), (1,0), (1,1)]

    for i, dim in enumerate(dimension_values):
        row, col = subplot_positions[i]
        ax = axes[row][col]

        # SEQ
        start_seq = time.time()
        _, _, alpha_seq = filter_sequential(dim, fixed_iters)
        seq_time = time.time() - start_seq

        # PAR
        start_par = time.time()
        _, _, alpha_par = filter_parallel(dim, fixed_iters)
        par_time = time.time() - start_par

        labels = ["Sequential", "Parallel"]
        times = [seq_time, par_time]

        ax.bar(labels, times, color=["blue", "orange"])

        speedup = seq_time / par_time if par_time != 0 else float('inf')
        ax.set_title(f"dim={dim}\nSeq={seq_time:.3f}s, Par={par_time:.3f}s\nSpeedup={speedup:.2f}")

        ax.set_ylabel("Time (s)")
        ax.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig("./comparePlot/scenario2.png", dpi=150)


if __name__ == "__main__":
    scenario1_fixed_dimension()
    scenario2_fixed_iterations()
