import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

def compute_energy(adjacency_matrix, state, J = 1.0, h = 0.0):
    energy = -h * np.sum(state)
    energy -= (J / 2.0) * np.dot(state, np.dot(adjacency_matrix, state))
    return energy

def simulate_ising_fixed_temp(adjacency_matrix, T, num_steps, n_snapshots = 0, J = 1.0, h = 0.0):
    if n_snapshots > 0:
        snapshot_steps = np.linspace(0, num_steps - 1, n_snapshots, dtype=int)
        snapshot_data = {}
    num_nodes = len(adjacency_matrix[0])
    initial_state = np.random.choice([-1, 1], size = num_nodes)
    data = np.zeros((num_steps, 3))
    state = initial_state.copy()
    for step in range(num_steps):
        if step % 1000 == 0:
            print(f"temperature {T}, step number {step}")
        """
        metropolis update 
        consists in N node updates (N == network size)
        so that, on average, each node is updated once
        """
        for _ in range(num_nodes):
            i = np.random.randint(num_nodes)
            delta_E = 2 * state[i] * (h + J * sum(adjacency_matrix[i, j] * state[j] for j in range(len(state)) if adjacency_matrix[i, j] != 0))
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                state[i] *= -1  # Flip the spin
        energy = compute_energy(adjacency_matrix, state, J, h)
        average_magnetization = np.sum(state) / len(state)
        data[step] = [step, energy, average_magnetization]
        if n_snapshots > 0 and step in snapshot_steps:
            snapshot_data[f'step_{step}'] = state.copy()
    """"
    final ouputs
    """
    df = pd.DataFrame(data, columns=['n_step', 'energy', 'average_magnetization'])
    if n_snapshots > 0:
        snapshot_df= pd.DataFrame(snapshot_data)
        return initial_state, state, df, snapshot_df
    else:
        return initial_state, state, df

def simulate_ising(adjacency_matrix, T_i, T_f, t_points, equilibration_steps, sweep_steps, J = 1.0, h = 0.0):
    if T_i > T_f:
        print("Error: T_i > T_f")
        return
    num_nodes = len(adjacency_matrix[0])
    temperatures = np.linspace(T_i, T_f, t_points)
    data = np.zeros((t_points, 9))
    for t, T in enumerate(temperatures):
        if t % 10 == 0:
            print(f"temperature {T}, point {t + 1}/{t_points}")
        energies = np.zeros(sweep_steps)
        magnetizations = np.zeros(sweep_steps)
        initial_state = np.random.choice([-1, 1], size = num_nodes)
        state = initial_state.copy()
        for step in range(equilibration_steps):
            for _ in range(num_nodes):
                i = np.random.randint(num_nodes)
                delta_E = 2 * state[i] * (h + J * sum(adjacency_matrix[i, j] * state[j] for j in range(len(state)) if adjacency_matrix[i, j] != 0))
                if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                    state[i] *= -1
        for s in range(sweep_steps):
            for _ in range(num_nodes):
                i = np.random.randint(num_nodes)
                delta_E = 2 * state[i] * (h + J * sum(adjacency_matrix[i, j] * state[j] for j in range(len(state)) if adjacency_matrix[i, j] != 0))
                if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                    state[i] *= -1
            energies[s] = compute_energy(adjacency_matrix, state)
            magnetizations[s] = np.sum(state)/num_nodes
        energy = np.mean(energies)
        std_energy = np.std(energies)
        magnetization = np.mean(magnetizations)
        std_magnetization = np.std(magnetizations)
        specific_heat = np.var(energies) / (T**2)
        std_specific_heat = specific_heat * np.sqrt(2 / sweep_steps)
        susceptibility = np.var(magnetizations) / T
        std_susceptibility = susceptibility * np.sqrt(2 / sweep_steps)
        data[t] = [T, energy, std_energy, np.abs(magnetization), std_magnetization,  specific_heat, std_specific_heat, susceptibility, std_susceptibility]
    df = pd.DataFrame(data, columns=['temperature', 'energy', 'std_energy' ,'abs_magnetization', 'std_magnetization', 'heat', 'std_heat', 'susceptibility', 'std_susceptibility'])
    return df


def estimate_temperature(df):

    # METHOD 1: maximum derivative of magnetization and energy
    dM_dT = np.gradient(df['abs_magnetization'], df['temperature'])
    Tc_magnetization = df['temperature'].iloc[np.argmax(np.abs(dM_dT))]

    dE_dT = np.gradient(df['energy'], df['temperature'])
    Tc_energy = df['temperature'].iloc[np.argmax(np.abs(dE_dT))]

    # METHOD 2: Fit of peaks in magnetic susceptibility and heat capacity
    local_peaks_indices, _ = find_peaks(df['susceptibility'])
    local_peak_temperatures = df['temperature'].iloc[local_peaks_indices]
    local_peak_values = df['susceptibility'].iloc[local_peaks_indices]
    local_peaks_dict = {index: value for index, value in zip(local_peaks_indices, local_peak_values)}
    max_value = max(local_peaks_dict.values())
    max_index = [index for index, value in local_peaks_dict.items() if value == max_value][0]
    Tc_susceptibility = df['temperature'].iloc[max_index]

    local_peaks_indices, _ = find_peaks(df['heat'])
    local_peak_temperatures = df['temperature'].iloc[local_peaks_indices]
    local_peak_values = df['heat'].iloc[local_peaks_indices]
    local_peaks_dict = {index: value for index, value in zip(local_peaks_indices, local_peak_values)}
    max_value = max(local_peaks_dict.values())
    max_index = [index for index, value in local_peaks_dict.items() if value == max_value][0]
    Tc_heat = df['temperature'].iloc[max_index]

    Tc_estimates = np.array([Tc_magnetization, Tc_energy, Tc_susceptibility, Tc_heat])
    Tc_average = np.mean(Tc_estimates)
    Tc_std = np.std(Tc_estimates)
    return Tc_average, Tc_std