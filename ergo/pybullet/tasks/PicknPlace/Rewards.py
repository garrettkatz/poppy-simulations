import csv
import matplotlib.pyplot as plt

import csv
import matplotlib.pyplot as plt

def plot_reward_curve(filename, smooth_window=10):
    rewards = []
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # skip empty lines
                rewards.append(float(row[0]))  # scalar per row

    # Optional smoothing (moving average)
    if smooth_window > 1:
        smoothed = []
        for i in range(len(rewards)):
            start = max(0, i - smooth_window + 1)
            smoothed.append(sum(rewards[start:i+1]) / (i - start + 1))
        rewards = smoothed

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(rewards, label='Reward', color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.grid(True)
    plt.legend()
    plt.show()

import csv
import re
import matplotlib.pyplot as plt

def plot_stddev_curves(filename, smooth_window=10):
    def clean_tensor_string(s):
        # Extract the first number in the string (handles negatives and decimals)
        match = re.search(r"-?\d+\.\d+", s)
        return float(match.group()) if match else float(s)

    data = []
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # skip empty lines
                data.append([clean_tensor_string(x) for x in row])

    # Transpose so each DOF is its own list
    dof_values = list(zip(*data))  # shape: (7, num_steps)

    # Moving average smoothing
    def smooth(series):
        if smooth_window <= 1:
            return series
        smoothed = []
        for i in range(len(series)):
            start = max(0, i - smooth_window + 1)
            smoothed.append(sum(series[start:i+1]) / (i - start + 1))
        return smoothed

    plt.figure(figsize=(10, 6))
    for i, dof_series in enumerate(dof_values):
        plt.plot(smooth(dof_series), label=f"DOF {i+1}")

    plt.xlabel("Step / Episode")
    plt.ylabel("Std. Deviation")
    plt.title("7-DOF Standard Deviations Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()
plot_stddev_curves("std_dev2_c.csv", smooth_window=100)
plot_reward_curve("rewards2_c.csv",smooth_window=100)