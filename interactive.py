import numpy as np
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Slider, Div
from bokeh.layouts import column, row
from scipy.linalg import expm

# Define basis states for qubits |C> and |D>
C = np.array([1, 0])
D = np.array([0, 1])

C_op = np.eye(2)
D_op = np.array([[0, 1], [-1, 0]])
Q_op = np.array([[1j, 0], [0, -1j]])

# Generate a strategy operator


def strategy_vectorized(theta, phi):
    cos_theta_2 = np.cos(theta / 2)
    sin_theta_2 = np.sin(theta / 2)
    exp_i_phi = np.exp(1j * phi)
    exp_neg_i_phi = np.exp(-1j * phi)
    return np.stack([
        np.stack([cos_theta_2 * exp_i_phi, sin_theta_2], axis=-1),
        np.stack([-sin_theta_2, cos_theta_2 * exp_neg_i_phi], axis=-1)
    ], axis=-2)

# Entanglement operator J


def J(gamma):
    return expm(1j * gamma/2.0 * np.kron(D_op, D_op))

# Compute quantum payoff


def quantum_payoff_vectorized(states, payoff_vector):
    probabilities = np.abs(states) ** 2
    return np.dot(probabilities, payoff_vector)

# Compute payoff heatmap for Player 1


def get_average_payoff_fixed_vectorized(gamma, theta_grid, phi_grid, fixed_theta, fixed_phi):
    theta, phi = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    S_fixed = strategy_vectorized(fixed_theta, fixed_phi)
    S_variable = strategy_vectorized(theta, phi)
    strategies = np.einsum('...ij,kl->...ikjl', S_variable,
                           S_fixed).reshape(theta.shape + (4, 4))
    entanglement_op = J(gamma)
    initial_state = np.kron(C, C)
    final_states = np.einsum(
        'ij,...jk,kl,l->...i',
        entanglement_op.conj().T,
        strategies,
        entanglement_op,
        initial_state
    )
    payoff1 = np.array([3, 0, 5, 1])
    payoffs = quantum_payoff_vectorized(final_states, payoff1)
    return payoffs.T

# Compute payoff heatmap for Player 2


def get_average_payoff_fixed_vectorized_p2(gamma, theta_grid, phi_grid, fixed_theta, fixed_phi):
    theta, phi = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    S_fixed = strategy_vectorized(fixed_theta, fixed_phi)
    S_variable = strategy_vectorized(theta, phi)
    strategies = np.einsum('ij,...kl->...ikjl', S_fixed,
                           S_variable).reshape(theta.shape + (4, 4))
    entanglement_op = J(gamma)
    initial_state = np.kron(C, C)
    final_states = np.einsum(
        'ij,...jk,kl,l->...i',
        entanglement_op.conj().T,
        strategies,
        entanglement_op,
        initial_state
    )
    payoff2 = np.array([3, 5, 0, 1])
    payoffs = quantum_payoff_vectorized(final_states, payoff2)
    return payoffs.T


# Resolution and fixed values
theta_grid = np.linspace(0, np.pi, 50)
phi_grid = np.linspace(0, np.pi / 2, 50)
theta1_fixed = np.pi / 4
phi1_fixed = np.pi / 4
theta2_fixed = np.pi / 4
phi2_fixed = np.pi / 4
gamma_fixed = 1.0

# Initial data for Player 1
initial_data = get_average_payoff_fixed_vectorized(
    gamma_fixed, theta_grid, phi_grid, theta2_fixed, phi2_fixed)

# Initial data for Player 2
initial_data_p2 = get_average_payoff_fixed_vectorized_p2(
    gamma_fixed, theta_grid, phi_grid, theta1_fixed, phi1_fixed)

# Create Bokeh figure for Player 1
p = figure(title="Payoff Heatmap (Player 1)", x_axis_label="Theta1", y_axis_label="Phi1",
           x_range=(0, np.pi), y_range=(0, np.pi / 2), tools="pan,box_zoom,reset", width=600, height=600)
data_source = ColumnDataSource(data=dict(
    image=[initial_data], x=[0], y=[0], dw=[np.pi], dh=[np.pi / 2]))
p.image(image='image', x='x', y='y', dw='dw', dh='dh',
        source=data_source, palette="Viridis256")
move_source = ColumnDataSource(data=dict(x=[theta2_fixed], y=[phi2_fixed]))
p.circle(x='x', y='y', source=move_source, size=10,
         color='red', legend_label="Player 2's Move")

# Create Bokeh figure for Player 2
p2 = figure(title="Payoff Heatmap (Player 2)", x_axis_label="Theta2", y_axis_label="Phi2",
            x_range=(0, np.pi), y_range=(0, np.pi / 2), tools="pan,box_zoom,reset", width=600, height=600)
data_source_p2 = ColumnDataSource(data=dict(
    image=[initial_data_p2], x=[0], y=[0], dw=[np.pi], dh=[np.pi / 2]))
p2.image(image='image', x='x', y='y', dw='dw', dh='dh',
         source=data_source_p2, palette="Viridis256")
move_source_p1 = ColumnDataSource(data=dict(x=[theta1_fixed], y=[phi1_fixed]))
p2.circle(x='x', y='y', source=move_source_p1, size=10,
          color='blue', legend_label="Player 1's Move")

# Divs to display maximum values
max_info_p1 = Div(text="Player 1 Max: ", width=600, height=30)
max_info_p2 = Div(text="Player 2 Max: ", width=600, height=30)

# Create sliders for interactivity
theta1_slider = Slider(
    start=0, end=np.pi, value=theta1_fixed, step=np.pi/2000, title="Theta1")
phi1_slider = Slider(start=0, end=np.pi / 2,
                     value=phi1_fixed, step=np.pi/2000, title="Phi1")
theta2_slider = Slider(
    start=0, end=np.pi, value=theta2_fixed, step=np.pi/2000, title="Theta2")
phi2_slider = Slider(start=0, end=np.pi / 2,
                     value=phi2_fixed, step=np.pi/2000, title="Phi2")
gamma_slider = Slider(start=0, end=np.pi / 2,
                      value=gamma_fixed, step=np.pi / 2000, title="Gamma")

# Update function


def update_data(attr, old, new):
    theta1 = theta1_slider.value
    phi1 = phi1_slider.value
    theta2 = theta2_slider.value
    phi2 = phi2_slider.value
    gamma = gamma_slider.value

    updated_data = get_average_payoff_fixed_vectorized(
        gamma, theta_grid, phi_grid, theta2, phi2)
    data_source.data = dict(image=[updated_data], x=[0], y=[
        0], dw=[np.pi], dh=[np.pi / 2])

    max_idx = np.unravel_index(
        np.argmax(updated_data, axis=None), updated_data.shape)
    max_value = updated_data[max_idx]
    max_theta = theta_grid[max_idx[1]]
    max_phi = phi_grid[max_idx[0]]
    max_info_p1.text = f"Player 1 Max: Value = {
        max_value:.3f}, Theta = {max_theta:.3f}, Phi = {max_phi:.3f}"

    move_source.data = dict(x=[theta2], y=[phi2])

    updated_data_p2 = get_average_payoff_fixed_vectorized_p2(
        gamma, theta_grid, phi_grid, theta1, phi1)
    data_source_p2.data = dict(image=[updated_data_p2], x=[0], y=[
        0], dw=[np.pi], dh=[np.pi / 2])

    max_idx_p2 = np.unravel_index(
        np.argmax(updated_data_p2, axis=None), updated_data_p2.shape)
    max_value_p2 = updated_data_p2[max_idx_p2]
    max_theta_p2 = theta_grid[max_idx_p2[1]]
    max_phi_p2 = phi_grid[max_idx_p2[0]]
    max_info_p2.text = f"Player 2 Max: Value = {max_value_p2:.3f}, Theta = {
        max_theta_p2:.3f}, Phi = {max_phi_p2:.3f}"

    move_source_p1.data = dict(x=[theta1], y=[phi1])


theta1_slider.on_change('value', update_data)
phi1_slider.on_change('value', update_data)
theta2_slider.on_change('value', update_data)
phi2_slider.on_change('value', update_data)
gamma_slider.on_change('value', update_data)

layout = column(
    row(p, p2),
    row(max_info_p1, max_info_p2),
    row(column(theta1_slider, phi1_slider), column(theta2_slider, phi2_slider)),
    gamma_slider
)
curdoc().add_root(layout)
curdoc().title = "Interactive Payoff Visualization"
