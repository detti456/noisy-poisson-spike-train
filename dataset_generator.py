import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
T = 10  # total duration in seconds

def generate_inhomogeneous_poisson_spikes(rate_func, T, dt=0.001):
    """
    Generate spike times for a neuron with a time-dependent firing rate using an inhomogeneous Poisson process.

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
        rate = rate_func(t)
        if rate * dt > np.random.rand():
            spike_times.append(t)
        t += dt

    return np.array(spike_times)

def gaussian_rate(t, peak_time=T/2, sigma=1.5, max_rate=20):
    return max_rate * np.exp(-0.5 * ((t - peak_time) / sigma) ** 2)

def double_gaussian_rate(t, peaks=(1.0, T-1), sigma=1, max_rate=20):
    rate = 0.0
    for peak in peaks:
        rate += max_rate * np.exp(-0.5 * ((t - peak) / sigma) ** 2)
    return rate

def r_constant(t, rate = 10.0):
    return rate

spike_timesA = generate_inhomogeneous_poisson_spikes(gaussian_rate, T)
spike_timesB = generate_inhomogeneous_poisson_spikes(double_gaussian_rate, T)
spike_timesC = generate_inhomogeneous_poisson_spikes(r_constant, T)

# Plotting the spike train
plt.eventplot(spike_timesA, orientation='horizontal', colors='black')
plt.xlabel('Time (s)')
plt.xlim([0, T])
plt.ylabel('Spike')
plt.title(f'Poisson Spike Train Gaussian Peak at {T/2}s')
plt.show()

# plt.eventplot(spike_timesB, orientation='horizontal', colors='black')
# plt.xlabel('Time (s)')
# plt.ylabel('Spike')
# plt.xlim([0, T])
# plt.title(f'Poisson Spike Train Gaussian Peak at {1}s, {T-1}s')
# plt.show()

# plt.eventplot(spike_timesC, orientation='horizontal', colors='black')
# plt.xlabel('Time (s)')
# plt.ylabel('Spike')
# plt.xlim([0, T])
# plt.title(f'Poisson Spike Train constant 10Hz rate')
# plt.show()

def jitter(original_data, variance):
    noise = np.random.normal(loc=0, scale=variance, size=original_data.shape)
    noisy_data = original_data + noise
    filtered_data = noisy_data[(noisy_data >= 0) & (noisy_data <= T)]
    return filtered_data

spike_timesA_jitter = jitter(spike_timesA, variance=1)
spike_timesB_jitter = jitter(spike_timesA, variance=4.0)

# plt.eventplot(spike_timesA_jitter, orientation='horizontal', colors='black')
# plt.xlabel('Time (s)')
# plt.ylabel('Spike')
# plt.xlim([0, T])
# plt.title(f'Poisson Spike Train constant 10Hz rate')
# plt.show()
#
# plt.eventplot(spike_timesB_jitter, orientation='horizontal', colors='black')
# plt.xlabel('Time (s)')
# plt.ylabel('Spike')
# plt.xlim([0, T])
# plt.title(f'Poisson Spike Train constant 10Hz rate')
# plt.show()

def deletion(original_data, prob):
    deletion_prob = np.random.uniform(low=0.0, high=1.0, size=original_data.shape)
    filtered_data = original_data[deletion_prob > prob]
    return filtered_data

# spike_timesC_deletion = deletion(spike_timesC, prob=0.5)
# plt.eventplot(spike_timesC_deletion, orientation='horizontal', colors='black')
# plt.xlabel('Time (s)')
# plt.ylabel('Spike')
# plt.xlim([0, T])
# plt.title(f'Poisson Spike Train constant 10Hz rate')
# plt.show()


def add_random_spike_times(spike_times, T, noise_rate_hz=1):
    # Estimate number of noise spikes
    expected_noise_spikes = np.random.poisson(noise_rate_hz * T )

    # Sample random times uniformly in the window
    noise_spike_times = np.random.uniform(0, T, expected_noise_spikes)

    # Combine and sort
    all_spike_times = np.concatenate([spike_times, noise_spike_times])
    all_spike_times.sort()

    return all_spike_times

# spike_timesA_insertion = add_random_spike_times(spike_timesA, T)
# plt.eventplot(spike_timesA_insertion, orientation='horizontal', colors='black')
# plt.xlabel('Time (s)')
# plt.ylabel('Spike')
# plt.xlim([0, T])
# plt.title(f'Poisson Spike Train constant 10Hz rate')
# plt.show()

# Noise in the Gaussian variance
def time_warping(original_data, means, scale):
    # return original_data + (original_data - mean) * scale

    signed_differences = original_data[:, np.newaxis] - means[np.newaxis, :]

    # Find the index of the mean with the smallest absolute difference
    min_indices = np.argmin(np.abs(signed_differences), axis=1)

    # Use the indices to extract the signed distance
    signed_min_distances = signed_differences[np.arange(len(original_data)), min_indices]

    noisy_data = original_data + signed_min_distances * scale

    filtered_data = noisy_data[(noisy_data >= 0) & (noisy_data <= T)]

    return filtered_data

# spike_timesA_insertion = time_warping(spike_timesC, np.array([1,5,9]), 0.3)
# plt.eventplot(spike_timesA_insertion, orientation='horizontal', colors='black')
# plt.xlabel('Time (s)')
# plt.ylabel('Spike')
# plt.xlim([0, T])
# plt.title(f'Poisson Spike Train constant 10Hz rate')
# plt.show()

def shift(original_data, dt):
    noisy_data = original_data + dt

    filtered_data = noisy_data[(noisy_data >= 0) & (noisy_data <= T)]

    return filtered_data

spike_timesA_insertion = shift(spike_timesB, -2.0)
plt.eventplot(spike_timesA_insertion, orientation='horizontal', colors='black')
plt.xlabel('Time (s)')
plt.ylabel('Spike')
plt.xlim([0, T])
plt.title(f'Poisson Spike Train constant 10Hz rate')
plt.show()