import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global seed value
_GLOBAL_SEED = None

def set_seed(seed_value):
    """
    Set the global random seed for reproducibility.
    
    Parameters:
        seed_value (int): The seed value to use for np.random.
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed_value
    np.random.seed(_GLOBAL_SEED)

def plot_network_graph(G):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_ID = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_ID.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title=dict(
              text='Node Connections',
              side='right'
            ),
            xanchor='left',
        ),
        line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'Node ID: {node}<br># of connections: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text


    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=600,
        height=600,
    )

    return fig

def plot_matrix(mats, labels=None):
    if labels is None: labels = [None] * len(mats)
    data = [go.Heatmap(
        z=mats[i], 
        colorscale='Viridis', 
        colorbar=dict(
            orientation='h',
            x= i/len(mats), 
            xanchor='left',    
            y=-0.2, 
            yanchor='top',
            len=0.3,
            thickness=10,
        ),
    ) for i in range(len(mats))]

    fig = make_subplots(rows=1, cols=len(mats), subplot_titles=labels)

    for i in range(len(mats)):
        fig.add_trace(data[i], row=1, col=i+1)
        fig.update_xaxes(scaleanchor=f"y{i+1}", scaleratio=1, row=1, col=i+1)
        fig.update_layout(showlegend=True)
    return fig

def generate_matrix_A_from_adjacency(AdjMat, max_influence=0.5):
    """
    Generate a synthetic NxN matrix A for a discrete-time linear network autoregressive model,
    constrained by the adjacency matrix G.
    
    Parameters:
        G (np.ndarray): NxN adjacency matrix of the network (binary or weighted).
        max_influence (float): Maximum absolute value of influence between connected nodes (default: 0.5).
        
    Returns:
        A (np.ndarray): NxN matrix representing the influence between nodes, constrained by G.
    """
    # Step 1: Validate the adjacency matrix
    N = AdjMat.shape[0]
    assert AdjMat.shape == (N, N), "Adjacency matrix must be square."
    
    # Step 2: Initialize matrix A with random values where G is non-zero
    A = np.zeros_like(AdjMat, dtype=float)  # Start with zeros
    non_zero_mask = AdjMat != 0            # Identify positions where G is non-zero
    A[non_zero_mask] = np.random.uniform(-max_influence, max_influence, size=np.sum(non_zero_mask))
    
    # Step 3: Ensure diagonal dominance for stability
    # Diagonal entries should dominate to prevent divergence
    for i in range(N):
        if AdjMat[i, i] == 0:  # If diagonal entry is zero in G, set it to a small positive value
            AdjMat[i, i] = 1   # Ensure diagonal dominance
        A[i, i] = np.random.uniform(max_influence/2, max_influence)  # Diagonal entries are positive and larger
    
    # Step 4: Normalize A to ensure stability (eigenvalues within unit circle)
    eigenvalues, _ = np.linalg.eig(A)
    spectral_radius = np.max(np.abs(eigenvalues))
    A = A / (spectral_radius)  # Scale down A to ensure stability
    
    return A, np.linalg.eigvals(A)

def generate_sine_wave_x_t(P, t, amplitude=1.0, frequency=1.0, phase=0.0):
    """
    Generate a synthetic sine wave input vector x_t.
    
    Parameters:
        P (int): Number of external inputs (features).
        t (int): Time step.
        amplitude (float): Amplitude of the sine wave (default: 1.0).
        frequency (float): Frequency of the sine wave (default: 1.0).
        phase (float): Phase shift of the sine wave (default: 0.0).
        
    Returns:
        x_t (np.ndarray): Synthetic sine wave input vector of shape (P, 1).
    """
    amplitude = np.random.uniform(low=0.5*amplitude, high=1.5*amplitude, size=(P, 1))
    frequency = np.random.uniform(low=0.5*frequency, high=1.5*frequency, size=(P, 1))
    x_t = amplitude * np.sin(t * frequency + phase) * np.ones((P, 1))
    return x_t

def generate_synthetic_B(N, P, min_val=-0.5, max_val=0.5):
    """
    Generate a synthetic influence matrix B.
    
    Parameters:
        N (int): Number of nodes in the network.
        P (int): Number of external inputs (features).
        min_val (float): Minimum value for entries in B (default: -0.5).
        max_val (float): Maximum value for entries in B (default: 0.5).
        
    Returns:
        B (np.ndarray): Synthetic influence matrix of shape (N, P).
    """
    B = np.random.uniform(min_val, max_val, size=(N, P))
    return B

def generate_noise(N, t, mean=0, std_dev=0.1):
    """
    Generate a synthetic noise matrix epsilon_t over a time vector.
    
    Parameters:
        N (int): Number of nodes in the network.
        t (np.ndarray): Time vector.
        mean (float): Mean of the noise distribution (default: 0).
        std_dev (float): Standard deviation of the noise distribution (default: 0.1).
        
    Returns:
        epsilon_t (np.ndarray): Synthetic noise matrix of shape (N, len(t)).
    """
    epsilon_t = np.random.normal(mean, std_dev, size=(N, len(t)))
    return epsilon_t

def generate_initial_conditions(N, lower_bound=-1.0, upper_bound=1.0):
    """
    Generate random initial conditions for a time series defined for each node in the network.
    
    Parameters:
        N (int): Number of nodes in the network.
        lower_bound (float): Lower bound for the initial conditions (default: -1.0).
        upper_bound (float): Upper bound for the initial conditions (default: 1.0).
        
    Returns:
        y_0 (np.ndarray): Initial state vector of shape (N, 1), with random values between lower_bound and upper_bound.
    """
    # Generate random values uniformly distributed between lower_bound and upper_bound
    y_0 = np.random.uniform(lower_bound, upper_bound, size=(N, 1))
    return y_0

def simulate_network_dynamics(A, B, x_t, epsilon_t, y_0, time_vector):
    """
    Simulate the dynamics of a discrete-time linear network autoregressive model.
    
    Parameters:
        A (np.ndarray): Influence matrix of shape (N, N).
        B (np.ndarray): External influence matrix of shape (N, P).
        x_t_func (function): Function to generate external inputs x_t at time t.
        epsilon_t_func (function): Function to generate noise epsilon_t at time t.
        y_0 (np.ndarray): Initial state vector of shape (N, 1).
        time_vector (np.ndarray): Time vector for simulation.
        
    Returns:
        y_series (list): List of state vectors [y_0, y_1, ..., y_T], each of shape (N, 1).
    """
    N = A.shape[0]  # Number of nodes
    T = len(time_vector)  # Number of time steps
    y_series = np.zeros(shape=(N, T+1))  # Initialize series of state vectors (N, T+1)
    y_series[:, 0] = y_0.flatten()  # Set initial state vector
    
    for t in range(1, len(time_vector) + 1):
        # Get the previous state
        y_prev = y_series[:, t-1]
        
        # Generate external input and noise at time t
        x = x_t[:,t-1]              # Shape (P, 1)
        epsilon = epsilon_t[:,t-1]  # Shape (N, 1)

        # Compute the current state using the formula
        y_t = np.dot(A, y_prev) + np.dot(B, x) + epsilon
        
        # Store the current state in the series
        y_series[:, t] = y_t.flatten()


    # y_series = np.hstack(y_series).reshape(N, len(time_vector) + 1)
    
    return y_series     # shape (T+1, N, 1)

def plot_time_series(y_series, time_vector):
    """
    Plot a list of time-series on top of each other.
    
    Parameters:
        y_series (list): List of state vectors [y_0, y_1, ..., y_T], each of shape (N, 1).
        time_vector (np.ndarray): Time vector for simulation.
    """
    fig = go.Figure()
    N = y_series.shape[0]
    for i in range(N):
        y_values = y_series[i, :]
        fig.add_trace(go.Scatter(x=time_vector, y=y_values, mode='lines', name=f'Node {i}'))
        if i>20: break
    
    fig.update_layout(
        title='Time Series of Network Nodes',
        xaxis_title='Time',
        yaxis_title='State Value',
        showlegend=True
    )
    
    return fig

def estimate_A(y_series, B, x_series):
    """
    Estimate the influence matrix A using least squares.
    
    Parameters:
        y_series (np.ndarray): State vectors over time, shape (N, T+1).
                               Each column corresponds to a state vector at a specific time step.
        B (np.ndarray): External influence matrix of shape (N, P).
        x_series (np.ndarray): External input vectors over time, shape (P, T).
                               Each column corresponds to an external input vector at a specific time step.
        
    Returns:
        A_hat (np.ndarray): Estimated influence matrix of shape (N, N).
    """
    # Extract dimensions
    # N, T_plus_1 = y_series.shape  # Number of nodes (N) and time steps (T+1)
    # T = T_plus_1 - 1              # Number of transitions
    # P = B.shape[1]                # Number of external inputs
    
    # Construct the design matrix X (previous states)
    X = y_series[:, :-1].T  # Shape (T, N), each row is y_{t-1}
    
    # Construct the target matrix Y (adjusted current states)
    Y = y_series[:, 1:].T - np.dot(B, x_series).T  # Shape (T, N), each row is y_t - B * x_t
    
    # Solve for A using least squares
    A_hat = np.linalg.lstsq(X, Y, rcond=None)[0].T  # Shape (N, N)
    return A_hat

def compare_matrices(A_true, A_hat):
    """
    Compare the true matrix A_true with the estimated matrix A_hat.
    
    Parameters:
        A_true (np.ndarray): True influence matrix of shape (N, N).
        A_hat (np.ndarray): Estimated influence matrix of shape (N, N).
        
    Returns:
        mae (float): Mean Absolute Error.
        mse (float): Mean Squared Error.
        rmse (float): Root Mean Squared Error.
    """
    diff = A_true - A_hat
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def generate_shock_x_t(P, t, amplitude=1.0, frequency=1.0):
    """
    Generate a synthetic shock input vector x_t.
    
    Parameters:
        P (int): Number of external inputs (features).
        t (int): Time step.
        amplitude (float): Amplitude of the shock (default: 1.0).
        frequency (float): Frequency of the shock (default: 2.0).
        
    Returns:
        x_t (np.ndarray): Synthetic shock input vector of shape (P, 1).
    """
    amplitude = np.random.uniform(low=0.5*amplitude, high=1.5*amplitude, size=(P, 1))
    frequency = np.random.uniform(low=0.5*frequency, high=1.5*frequency, size=(P, 1)).astype(int)
    x_t = np.zeros((P, len(t)))
    for i in range(P):
        shock_locs = np.random.choice(np.arange(len(t)), size=frequency[i], replace=False)
        x_t[i, shock_locs] = amplitude[i]
    return x_t

def generate_x_t(P,t, specs):
    if specs['input_type'] == 'sine':
        return generate_sine_wave_x_t(P, t, amplitude=specs['amplitude'], frequency=specs['frequency'])
    elif specs['input_type'] == 'shock':
        return generate_shock_x_t(P, t, amplitude=specs['amplitude'], frequency=specs['frequency'])
   
def plot_eigenvalues(eigenvalues_true, eigenvalues_final=None):
    """
    Plot the eigenvalues on the complex plane.
    
    Parameters:
        eigenvalues (np.ndarray): Array of eigenvalues.
    """
    fig = go.Figure()

    # Add eigenvalues as scatter points
    fig.add_trace(go.Scatter(
        x=np.real(eigenvalues_true),
        y=np.imag(eigenvalues_true),
        mode='markers',
        marker=dict(color='blue', size=10),
        name='Eigenvalues True'
    ))

    if eigenvalues_final is not None:
        fig.add_trace(go.Scatter(
            x=np.real(eigenvalues_final),
            y=np.imag(eigenvalues_final),
            mode='markers',
            marker=dict(color='red', size=10),
            name='Eigenvalues Final'
        ))

    # Add unit circle for reference
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=-1,
        y0=-1,
        x1=1,
        y1=1,
        line=dict(color="grey", dash="dash"),
        name='Unit Circle'
    )

    fig.update_layout(
        title='Eigenvalues on the Complex Plane',
        xaxis_title='Real Part',
        yaxis_title='Imaginary Part',
        showlegend=True,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=600,
        height=600,
    )

    return fig

def generate_eigenvalues(angle_range, radius_range, num_eigenvalues):
    """
    Generate a set of eigenvalues within specified angle and radius ranges.
    
    Parameters:
        angle_range (tuple): Range of angles in radians (min_angle, max_angle).
        radius_range (tuple): Range of radii (min_radius, max_radius).
        num_eigenvalues (int): Number of eigenvalues to generate.
        
    Returns:
        eigenvalues (np.ndarray): Array of complex eigenvalues within the specified ranges.
    """
    # Step 1: Validate inputs
    min_angle, max_angle = angle_range
    min_radius, max_radius = radius_range

    # Convert angles from degrees to radians
    min_angle = np.deg2rad(min_angle)
    max_angle = np.deg2rad(max_angle)
    
    assert min_angle >= 0 and max_angle <= 2 * np.pi, "Angles must be in the range [0, 2π]."
    assert min_radius >= 0 and max_radius <= 1, "Radii must be in the range [0, 1] for stability."
    assert min_angle < max_angle, "Angle range must satisfy min_angle < max_angle."
    assert min_radius < max_radius, "Radius range must satisfy min_radius < max_radius."
    assert num_eigenvalues > 0, "Number of eigenvalues must be positive."
    
    # Step 2: Generate random angles and radii
    half_num_eigenvalues = num_eigenvalues // 2
    angles = np.random.uniform(min_angle, max_angle, size=half_num_eigenvalues)
    radii = np.random.uniform(min_radius, max_radius, size=half_num_eigenvalues)
    
    # Step 3: Convert polar coordinates to complex eigenvalues
    eigenvalues = radii * (np.cos(angles) + 1j * np.sin(angles))
    
    # Step 4: Create conjugate pairs
    eigenvalues = np.concatenate([eigenvalues, np.conj(eigenvalues)])
    
    # Step 5: If num_eigenvalues is odd, add one real eigenvalue
    if num_eigenvalues % 2 != 0:
        real_eigenvalue = np.random.uniform(min_radius, max_radius)
        eigenvalues = np.append(eigenvalues, real_eigenvalue)
    
    # Step 6: Shuffle eigenvalues to avoid any specific order
    np.random.shuffle(eigenvalues)
    
    return eigenvalues

def generate_matrix_A_from_eigenvalues(eigenvalues, AdjMat=None):
    """
    Generate a stable NxN matrix A with given eigenvalues, optionally constrained by an adjacency matrix.
    
    Parameters:
        eigenvalues (list or np.ndarray): List of eigenvalues (must lie within the unit circle for stability).
        AdjMat (np.ndarray, optional): NxN adjacency matrix of the network (binary or weighted). 
                                       If provided, A will inherit the sparsity structure of AdjMat.
    
    Returns:
        A (np.ndarray): NxN matrix with the specified eigenvalues, optionally constrained by AdjMat.
    """
    # Step 1: Validate eigenvalues
    eigenvalues = np.array(eigenvalues)
    assert np.all(np.abs(eigenvalues) < 1), "All eigenvalues must lie within the unit circle for stability."
    N = len(eigenvalues)
    
    # Step 2: Construct a random orthogonal matrix Q (for spectral decomposition)
    # Using QR decomposition to generate an orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(N, N))
    
    # Step 3: Construct the diagonal matrix Lambda with the given eigenvalues
    Lambda = np.diag(eigenvalues)
    
    # Step 4: Reconstruct the matrix A using A = Q @ Lambda @ Q.T
    A = Q @ Lambda @ Q.T
    
    # Step 5: Enforce sparsity based on AdjMat (if provided)
    if AdjMat is not None:
        assert AdjMat.shape == (N, N), "Adjacency matrix must be square and match the size of eigenvalues."
        non_zero_mask = AdjMat != 0  # Identify positions where AdjMat is non-zero
        
        # Retain only the values in A where AdjMat is non-zero
        A = A * non_zero_mask
        
        # Ensure diagonal dominance for stability
        # for i in range(N):
        #     if AdjMat[i, i] == 0:  # If diagonal entry is zero in AdjMat, set it to 1
        #         AdjMat[i, i] = 1
        #     A[i, i] = np.random.uniform(0.5, 1.0)  # Diagonal entries are positive and larger
    
    eigenvalues_final = np.linalg.eigvals(A)
    return A, eigenvalues_final