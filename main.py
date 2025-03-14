import streamlit as st
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils

# Sidebar for Inputs
st.sidebar.header("Input Parameters")

# Network Parameters
seed = st.sidebar.number_input("Random Seed", min_value=0, value=42)

N = st.sidebar.number_input("Number of Nodes, $N$", min_value=2, value=10)
radius = st.sidebar.slider("Radius for Random Geometric Graph", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

# Time Vector Parameters
max_time = st.sidebar.number_input("Maximum Time for Time Vector", min_value=1, value=6)
num_time_steps = st.sidebar.number_input("Number of Time Steps", min_value=10, value=500)

# Matrix A Parameters
# angle_min, angle_max = st.sidebar.slider("Angle Range for Eigenvalues of Adjacency Matrix", min_value=0, max_value=180, value=(0, 180))
# r_min, r_max = st.sidebar.slider("Radius Range for Eigenvalues of Adjacency Matrix", min_value=0.0, max_value=1.0, value=(0.0, 1.0))
max_influence = st.sidebar.slider("Maximum Influence Between Connected Nodes, $\\text{max}(\\mathbf{A})$", min_value=0.1, max_value=1.0, value=0.8, step=0.1)

# External Inputs Parameters
P = st.sidebar.number_input("Number of External Inputs (Features), $P$", min_value=1, value=2)
input_type = st.sidebar.selectbox("Type of $\\mathbf{x}_t$", ["sine", "shock"], index=0)
amplitude = st.sidebar.number_input("Amplitude of $\\mathbf{x}_t$", min_value=0.1, value=1.0)
frequency = st.sidebar.number_input("Frequency of $\\mathbf{x}_t$", min_value=0.1, value=2.0)

# Initial Conditions Parameters
abs_bound_IC = st.sidebar.number_input("Absolute Bound for Initial Conditions, $|\\mathbf{y}_0|$", value=1.0)

# Matrix B Parameters
abs_val_B = st.sidebar.number_input("Absolute Bound for Matrix, $|B|$", min_value=-1.0, value=0.5)

# Noise Parameters
mean_noise = st.sidebar.number_input("Mean for Noise, $\\mu_{\\epsilon_t}$", min_value=-10.0, value=0.0)
std_dev_noise = st.sidebar.number_input("Standard Deviation for Noise, $\\sigma_{\\epsilon_t}$", min_value=0.0, value=1.0)

# Submit Button
# submit = st.sidebar.button("Generate and Estimate")

# Streamlit App
st.title("Network Dynamics Simulation and Estimation")
st.write("""
This app simulates and analyzes the dynamics of a discrete-time linear network autoregressive model. The goal is to study how the states of nodes in a network evolve over time under the influence of internal interactions, external inputs, and random noise.

## Problem Description

We consider a system of $ N $ nodes whose states are governed by the following equation:

$$
\\mathbf{y}_t = \\mathbf{A} \\mathbf{y}_{t-1} + \\mathbf{B} \\mathbf{x}_t + \\mathbf{\\epsilon}_t,
$$

where:
- $ \\mathbf{y}_t $: State vector of the network at time $ t $ (shape $ N \\times 1 $).
- $ \\mathbf{A} $: Influence matrix encoding interactions between nodes (shape $ N \\times N $).
- $ \\mathbf{y}_{t-1} $: State vector at the previous time step.
- $ \\mathbf{B} $: External influence matrix encoding how external inputs affect nodes (shape $ N \\times P $).
- $ \\mathbf{x}_t $: External input vector at time $ t $ (shape $ P \\times 1 $).
- $ \\mathbf{\\epsilon}_t $: Noise vector at time $ t $ (shape $ N \\times 1 $).

## Key Objectives

1. **Simulate Network Dynamics**:
   - Generate synthetic data for $ \\mathbf{y}_t $ using predefined $ \\mathbf{A} $, $ \\mathbf{B} $, $ \\mathbf{x}_t $, and $ \\mathbf{\\epsilon}_t $.
   - Visualize the evolution of node states over time.

2. **Estimate Influence Matrix $ \\mathbf{A} $**:
   - Use least squares estimation to recover $ \\hat{\\mathbf{A}} $ from the simulated data, assuming $ \\mathbf{B} $, $ \\mathbf{x}_t $, and $\\mathbf{y}_t$ are known.
   - Compare $ \\hat{\\mathbf{A}} $ with the true $ \\mathbf{A} $ using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

3. **Analyze Results**:
   - Visualize the network structure.
   - Compare $ \\mathbf{A} $ and $ \\hat{\\mathbf{A}} $ side by side.
   - Plot the time series of node states, external inputs, and noise.

## Inputs and Outputs

### Inputs:
- Network parameters (number of nodes, radius, adjacency matrix).
- Time vector and dynamics parameters ($ \\mathbf{A} $, $ \\mathbf{B} $, $ \\mathbf{x}_t $, $ \\mathbf{\\epsilon}_t $).
- Initial conditions for $ \\mathbf{y}_0 $.

### Outputs:
- Simulated time series $ \\mathbf{y}_t $.
- Estimated influence matrix $ \\hat{\\mathbf{A}} $.
- Visualizations of the network, matrices, and time series.

## Stability Considerations

To ensure stable dynamics, we normalize $ \\mathbf{A} $ such that its spectral radius ($ \\rho(\\mathbf{A}) $) is less than 1. This prevents oscillatory or divergent behavior in $ \\mathbf{y}_t $. Additionally, we smooth external inputs and reduce noise magnitude to avoid high-frequency fluctuations.

Let’s proceed to configure the parameters and simulate the network dynamics!
""")

# if submit:
# Set Random Seed
utils.set_seed(seed)

# Generate Time Vector
time_vector = np.linspace(0, max_time, num_time_steps)

# Generate Network
G = nx.random_geometric_graph(N, radius)
adj_matrix = nx.adjacency_matrix(G).todense()

# Generate Matrices and Initial Conditions
y0 = utils.generate_initial_conditions(N, lower_bound=-abs_bound_IC, upper_bound=abs_bound_IC)
A, eigenvalues = utils.generate_matrix_A_from_adjacency(adj_matrix, max_influence=max_influence)
# eigenvalues_true = utils.generate_eigenvalues([angle_min, angle_max], [r_min, r_max], N)
# A, eigenvalues_final = utils.generate_matrix_A_from_eigenvalues(eigenvalues_true, adj_matrix)

B = utils.generate_synthetic_B(N, P, min_val=-abs_val_B, max_val=abs_val_B)

# Generate x_t and epsilon_t
x_t_specs = {'input_type': input_type, 'amplitude': amplitude, 'frequency': frequency}
x_t = utils.generate_x_t(P, time_vector, x_t_specs)
epsilon_t = utils.generate_noise(N, time_vector, mean=mean_noise, std_dev=std_dev_noise)

# Simulate Network Dynamics
y_series = utils.simulate_network_dynamics(A, B, x_t, epsilon_t, y0, time_vector)

# Estimate A_hat
A_hat = utils.estimate_A(y_series, B, x_t)
# Compare Matrices
mae, mse, rmse = utils.compare_matrices(A, A_hat)
st.write(f"Comparison Metrics: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}")

# Display Outputs
st.header("Network Graph")
st.plotly_chart(utils.plot_network_graph(G))

st.header("Eigenvalues of Matrix $\\mathbf{A}$")
st.plotly_chart(utils.plot_eigenvalues(eigenvalues))
st.header("Matrices Comparison")
st.write("""
The plot below shows side-by-side comparison of matrices $ \\mathbf{A} $ and $ \\hat{\\mathbf{A}} $.
""")
st.plotly_chart(utils.plot_matrix([A, A_hat], labels=["True A", "Estimated A"]))
# st.plotly_chart(utils.plot_matrix([A], labels=["True A"]))

st.header("Time Series, $\\mathbf{y}_t$")
st.plotly_chart(utils.plot_time_series(y_series, time_vector))

st.header("External Inputs, $\\mathbf{x}_t$")
st.plotly_chart(utils.plot_time_series(x_t, time_vector))