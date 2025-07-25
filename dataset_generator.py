import pickle

import numpy as np
import matplotlib.pyplot as plt

def generate_inhomogeneous_poisson_spikes(rate_func, T, dt=0.001):
    """
    Generate spike times for a neuron with a time-dependent firing rate using an inhomogeneous Poisson process.
    Code from: https://medium.com/@baxterbarlow/poisson-spike-generators-stochastic-theory-to-python-code-a76f8cc7cc32

    Parameters:
    rate_func (function): Function that gives the firing rate at time t (spikes per second).
    T (float): Total duration of the simulation (seconds).
    dt (float): Time step for simulation (seconds).

    Returns:
    spike_times (list): List of spike times.
    """
    spike_times = []
    t = 0

    while t < T:
        rate = rate_func.rate(t)
        if rate * dt > np.random.rand():
            spike_times.append(t)
        t += dt

    return np.array(spike_times)

class GaussianRate:
    def __init__(self, max_rate, peaks, sigma):
        self.max_rate = max_rate
        self.peaks = peaks
        self.sigma = sigma

    def rate(self, t):
        rate = 0.0
        for peak in self.peaks:
            rate += self.max_rate * np.exp(-0.5 * ((t - peak) / self.sigma) ** 2)
        return rate

def jitter(original_data, variance):
    noise = np.random.normal(loc=0, scale=variance, size=original_data.shape)
    noisy_data = original_data + noise
    filtered_data = noisy_data[(noisy_data >= 0) & (noisy_data <= T)]
    return filtered_data

def deletion(original_data, prob):
    deletion_prob = np.random.uniform(low=0.0, high=1.0, size=original_data.shape)
    filtered_data = original_data[deletion_prob > prob]
    return filtered_data

def insertion(spike_times, T, frac=0.3):
    expected_noise_spikes = int(spike_times.size * frac)

    # Sample random times uniformly in the window
    noise_spike_times = np.random.uniform(0, T, expected_noise_spikes)

    # Combine and sort
    all_spike_times = np.concatenate([spike_times, noise_spike_times])
    all_spike_times.sort()

    return all_spike_times

def variance_noise(original_data, means, scale):
    signed_differences = original_data[:, np.newaxis] - means[np.newaxis, :]

    # Find the index of the mean with the smallest absolute difference
    min_indices = np.argmin(np.abs(signed_differences), axis=1)

    # Use the indices to extract the signed distance
    signed_min_distances = signed_differences[np.arange(len(original_data)), min_indices]

    noisy_data = original_data + signed_min_distances * scale

    filtered_data = noisy_data[(noisy_data >= 0) & (noisy_data <= T)]

    return filtered_data

def shift(original_data, dt):
    noisy_data = original_data + dt

    filtered_data = noisy_data[(noisy_data >= 0) & (noisy_data <= T)]

    return filtered_data

def plotting(data, titles, path, row=1, column=3):
    fig, axs = plt.subplots(row, column, figsize=(15, 4))

    for i in range(column):
        axs[i].eventplot(data[i], orientation='horizontal', colors='black')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_xlim([0, T])
        axs[i].set_ylabel('Spike')
        axs[i].set_title(titles[i])

    plt.tight_layout()
    plt.savefig(f"visualization/{path}.png")
    plt.show()


N = 100 # Number of instance per class
T = 10  # Total duration in seconds

np.random.seed(42)

class_A_instances = []
class_B_instances = []
class_C_instances = []

peak_A = [T / 2.0]
peak_B = [2.0, T - 2.0]
peak_C = [2.0, T / 2.0, T - 2.0]

rate_func_A = GaussianRate(15, peak_A, 1.5)
rate_func_B = GaussianRate(15, peak_B, 1)
rate_func_C = GaussianRate(15, peak_C, 0.5)


for i in range(N):
    class_A_instances.append(generate_inhomogeneous_poisson_spikes(rate_func_A, T))
    class_B_instances.append(generate_inhomogeneous_poisson_spikes(rate_func_B, T))
    class_C_instances.append(generate_inhomogeneous_poisson_spikes(rate_func_C, T))

do_plotting = False

plotting_idx = 0
if do_plotting:
    plotting([class_A_instances[plotting_idx], class_B_instances[plotting_idx], class_C_instances[plotting_idx]],
             [f'Poisson Spike Train with Gaussian Peak at {peak_A[0]:.0f}s',
              f'Poisson Spike Train with Gaussian Peak at {[int(x) for x in peak_B]}s',
              f'Poisson Spike Train with Gaussian Peak at {[int(x) for x in peak_C]}s'], "X_original")

X_original = class_A_instances + class_B_instances + class_C_instances
Y = np.repeat(np.array([0, 1, 2]), N)

with open('data/X_original.pkl', 'wb') as f:
    pickle.dump(X_original, f)

with open('data/Y.pkl', 'wb') as f:
    pickle.dump(Y, f)

# Jitter the original data
jitter_variances = np.arange(0.5, 4.5, 0.5)
for var in jitter_variances:
    X_jitter = [jitter(instance, var) for instance in X_original]
    if do_plotting:
        plotting([X_jitter[plotting_idx], X_jitter[N + plotting_idx], X_jitter[2 * N + plotting_idx]],
                 [f'Gaussian Peak at {peak_A[0]:.0f}s, jitter with variance {var}',
                  f'Gaussian Peak at {[int(x) for x in peak_B]}s, jitter with variance {var}',
                  f'Gaussian Peak at {[int(x) for x in peak_C]}s, jitter with variance {var}'],
                 f"jitter/X_jitter_{var}_variance")
    with open(f'data/jitter/X_jitter_{var}_variance.pkl', 'wb') as f:
        pickle.dump(X_jitter, f)

# Deletion of random spikes
deletion_probs = np.arange(0.1, 1.0, 0.1)
for prob in deletion_probs:
    X_deleted = [deletion(instance, prob) for instance in X_original]
    if do_plotting:
        plotting([X_deleted[plotting_idx], X_deleted[N + plotting_idx], X_deleted[2 * N + plotting_idx]],
                 [f'Gaussian Peak at {peak_A[0]:.0f}s, deletion with probability {prob:.1f}',
                  f'Gaussian Peak at {[int(x) for x in peak_B]}s, deletion with probability {prob:.1f}',
                  f'Gaussian Peak at {[int(x) for x in peak_C]}s, deletion with probability {prob:.1f}'],
                 f"deletion/X_deleted_{prob:.1f}_prob")
    with open(f'data/deletion/X_deleted_{prob:.1f}_prob.pkl', 'wb') as f:
        pickle.dump(X_deleted, f)

# Insertion of random spikes
insertion_frac = np.arange(0.2, 1.1, 0.2)
for frac in insertion_frac:
    X_inserted = [insertion(instance, T, frac) for instance in X_original]
    if do_plotting:
        plotting([X_inserted[plotting_idx], X_inserted[N + plotting_idx], X_inserted[2 * N + plotting_idx]],
                 [f'Gaussian Peak at {peak_A[0]:.0f}s, inserting {int(X_inserted[plotting_idx].size * frac)} spikes',
                  f'Gaussian Peak at {[int(x) for x in peak_B]}s, inserting {int(X_inserted[N + plotting_idx].size * frac)} spikes',
                  f'Gaussian Peak at {[int(x) for x in peak_C]}s, inserting {int(X_inserted[2 * N + plotting_idx].size * frac)} spikes'],
                 f"insertion/X_inserted_{frac:.1f}_frac")
    with open(f'data/insertion/X_inserted_{frac:.1f}_frac.pkl', 'wb') as f:
        pickle.dump(X_inserted, f)

# Noise in the variance of the Gaussian distribution (stretching/compressing the Gaussian peaks)
variance_scales = np.arange(-0.6, 0.7, 0.2)
variance_scales = np.delete(variance_scales, 3)
for scale in variance_scales:
    X_variance_noise_A = [variance_noise(instance, np.array(peak_A), scale) for instance in X_original[:N]]
    X_variance_noise_B = [variance_noise(instance, np.array(peak_B), scale) for instance in X_original[N:2 * N]]
    X_variance_noise_C = [variance_noise(instance, np.array(peak_C), scale) for instance in
                          X_original[2 * N:]]
    X_variance_noise = X_variance_noise_A + X_variance_noise_B + X_variance_noise_C
    if do_plotting:
        plotting([X_variance_noise[plotting_idx], X_variance_noise[N + plotting_idx], X_variance_noise[2 * N + plotting_idx]],
                 [f'Gaussian Peak at {peak_A[0]:.0f}s, variance scaled by {scale:.1f}',
                  f'Gaussian Peak at {[int(x) for x in peak_B]}s, variance scaled by {scale:.1f}',
                  f'Gaussian Peak at {[int(x) for x in peak_C]}s, variance scaled by {scale:.1f}'],
                 f"variance_noise/X_variance_noise_{scale:.1f}_scale")
    with open(f'data/variance_noise/X_variance_noise_{scale:.1f}_scale.pkl', 'wb') as f:
        pickle.dump(X_variance_noise, f)

# Noise in the Gaussian mean -> shifting the spike trains
shifts = np.arange(-2.0, 2.1, 0.4)
shifts = np.delete(shifts, 5)
for s in shifts:
    X_shifted = [shift(instance, s) for instance in X_original]
    if do_plotting:
        plotting([X_shifted[plotting_idx], X_shifted[N + plotting_idx], X_shifted[2 * N + plotting_idx]],
                 [f'Gaussian Peak at {peak_A[0]:.0f}s, shift with {s:.1f}s',
                  f'Gaussian Peak at {[int(x) for x in peak_B]}s, shift with {s:.1f}s',
                  f'Gaussian Peak at {[int(x) for x in peak_C]}s, shift with {s:.1f}s'],
                 f"shift/X_shifted_{s:.1f}_shift")
    with open(f'data/shift/X_shifted_{s:.1f}_shift.pkl', 'wb') as f:
        pickle.dump(X_shifted, f)