import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.datasets import make_circles, make_blobs
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

# Setup Streamlit page layout
st.set_page_config(layout="wide", page_title="SVM Poly Kernel Visualization")
st.title("SVM with Degree 2 Homogeneous Polynomial Kernel")

# --- Control Bar ---
st.sidebar.header("1. Data Generation")
dataset_type = st.sidebar.selectbox(
    "Dataset Type",
    ("Circles", "XOR", "Gaussians", "Spirals"),
    help="Matches the constructions in image_9c381c.png"
)
n_samples = st.sidebar.slider("Number of samples", 50, 1000, 300, 50)
noise = st.sidebar.slider("Noise level", 0.0, 1.0, 0.1, 0.05)
generate_btn = st.sidebar.button("Generate New Sample")

st.sidebar.header("2. SVM Hyperparameters (Log Scaled)")

# Create log-spaced arrays from 10^-2 to 10^2
log_options = np.logspace(-2, 2, 101)

C = st.sidebar.select_slider(
    "C (Regularization)",
    options=log_options,
    value=1.0,
    format_func=lambda x: f"{x:.3f}"
)


# --- Data Generation Functions ---
def generate_spirals(n_samples, noise):
    n = n_samples // 2
    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi

    # Class 0
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    data_a += np.random.randn(n, 2) * noise * 3

    # Class 1
    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    data_b += np.random.randn(n, 2) * noise * 3

    X = np.vstack([data_a, data_b])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y


def generate_xor(n_samples, noise):
    X = np.random.randn(n_samples, 2) * 2
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    X += np.random.randn(n_samples, 2) * noise
    return X, y


def get_data(dataset_type, n_samples, noise):
    if dataset_type == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    elif dataset_type == "Gaussians":
        X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=noise * 3 + 0.5, random_state=None)
    elif dataset_type == "XOR":
        X, y = generate_xor(n_samples, noise)
    elif dataset_type == "Spirals":
        X, y = generate_spirals(n_samples, noise)
    return X, y


# --- Session State for Data ---
if "X" not in st.session_state or "y" not in st.session_state or generate_btn:
    X, y = get_data(dataset_type, n_samples, noise)
    st.session_state.X = X
    st.session_state.y = y

X, y = st.session_state.X, st.session_state.y

# --- Train SVM ---
# For a degree 2 homogeneous polynomial: K(x, z) = (<x, z>)^2
clf = SVC(kernel='poly', degree=2, gamma=1.0, coef0=0.0, C=C)
clf.fit(X, y)

# --- Plotting Preparation ---
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Colors
cmap_light = ListedColormap(['#FFD3A5', '#A5D6FF'])
cmap_bold = ListedColormap(['#FF8C00', '#0078D7'])

# --- Visualizations ---
# Change column ratios: 3D plot column is twice as wide as the others
col1, col2, col3 = st.columns([1, 1, 2])

# 1) Original Data
with col1:
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=20)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_title("1. Original Data")
    ax1.set_xticks(())
    ax1.set_yticks(())
    st.pyplot(fig1)

# 2) SVM Boundary & Support Vectors
with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Decision boundary and margins
    ax2.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], cmap=cmap_light, alpha=0.8)
    ax2.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')

    # Scatter data
    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=20)

    # Highlight support vectors
    ax2.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='red', label='Support Vectors')

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title("2. SVM Boundary")
    ax2.set_xticks(())
    ax2.set_yticks(())
    st.pyplot(fig2)

# 3) Projected Space Interactive 3D Plot (Plotly)
with col3:
    # Degree 2 polynomial mapping function: phi(x) = (x1^2, x2^2, sqrt(2)*x1*x2)
    def phi(data):
        return np.column_stack((
            data[:, 0] ** 2,
            data[:, 1] ** 2,
            np.sqrt(2) * data[:, 0] * data[:, 1]
        ))


    # Map the entire dataset to 3D space
    Phi_X = phi(X)

    # Get bounds of the mapped data to strictly control the axes
    z1_min, z1_max = Phi_X[:, 0].min(), Phi_X[:, 0].max()
    z2_min, z2_max = Phi_X[:, 1].min(), Phi_X[:, 1].max()
    z3_min, z3_max = Phi_X[:, 2].min(), Phi_X[:, 2].max()

    # Calculate appropriate margins (10% padding based on data spread)
    z1_margin = max((z1_max - z1_min) * 0.1, 0.2)
    z2_margin = max((z2_max - z2_min) * 0.1, 0.2)
    z3_margin = max((z3_max - z3_min) * 0.1, 0.2)

    # Assign colors based on labels
    colors = ['#0078D7' if label == 1 else '#FF8C00' for label in y]

    # 3D Scatter plot
    scatter = go.Scatter3d(
        x=Phi_X[:, 0],
        y=Phi_X[:, 1],
        z=Phi_X[:, 2],
        mode='markers',
        marker=dict(size=4, color=colors, line=dict(width=1, color='black')),
        name='Data'
    )

    plot_data = [scatter]

    # Hyperplane: w * phi(x) + b = 0  =>  w[0]*Z1 + w[1]*Z2 + w[2]*Z3 + b = 0
    phi_sv = phi(clf.support_vectors_)
    w = np.dot(clf.dual_coef_, phi_sv).flatten()
    b = clf.intercept_[0]

    # Dynamically solve the plane equation for the dimension with the largest coefficient
    max_idx = np.argmax(np.abs(w))

    Z1_range = np.linspace(z1_min - z1_margin, z1_max + z1_margin, 30)
    Z2_range = np.linspace(z2_min - z2_margin, z2_max + z2_margin, 30)
    Z3_range = np.linspace(z3_min - z3_margin, z3_max + z3_margin, 30)

    if max_idx == 2:  # Solve for Z3 (z-axis)
        Z1_mesh, Z2_mesh = np.meshgrid(Z1_range, Z2_range)
        Z3_mesh = -(w[0] * Z1_mesh + w[1] * Z2_mesh + b) / w[2]
        Z3_mesh[(Z3_mesh < z3_min - z3_margin) | (Z3_mesh > z3_max + z3_margin)] = np.nan
        x_mesh, y_mesh, z_mesh = Z1_mesh, Z2_mesh, Z3_mesh

    elif max_idx == 1:  # Solve for Z2 (y-axis)
        Z1_mesh, Z3_mesh = np.meshgrid(Z1_range, Z3_range)
        Z2_mesh = -(w[0] * Z1_mesh + w[2] * Z3_mesh + b) / w[1]
        Z2_mesh[(Z2_mesh < z2_min - z2_margin) | (Z2_mesh > z2_max + z2_margin)] = np.nan
        x_mesh, y_mesh, z_mesh = Z1_mesh, Z2_mesh, Z3_mesh

    else:  # Solve for Z1 (x-axis)
        Z2_mesh, Z3_mesh = np.meshgrid(Z2_range, Z3_range)
        Z1_mesh = -(w[1] * Z2_mesh + w[2] * Z3_mesh + b) / w[0]
        Z1_mesh[(Z1_mesh < z1_min - z1_margin) | (Z1_mesh > z1_max + z1_margin)] = np.nan
        x_mesh, y_mesh, z_mesh = Z1_mesh, Z2_mesh, Z3_mesh

    surface = go.Surface(
        x=x_mesh, y=y_mesh, z=z_mesh,
        colorscale='Greys', opacity=0.5, showscale=False, name='Hyperplane'
    )
    plot_data.append(surface)

    layout = go.Layout(
        title=dict(text="3. Projected Feature Space", x=0.5, font=dict(size=14)),
        height=600,  # Make the plot taller to match the new width
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title="x₁²",
            yaxis_title="x₂²",
            zaxis_title="√2 x₁ x₂",
            xaxis=dict(range=[z1_min - z1_margin, z1_max + z1_margin]),
            yaxis=dict(range=[z2_min - z2_margin, z2_max + z2_margin]),
            zaxis=dict(range=[z3_min - z3_margin, z3_max + z3_margin])
        ),
        showlegend=False
    )

    fig3 = go.Figure(data=plot_data, layout=layout)
    st.plotly_chart(fig3, use_container_width=True)

# --- Metrics Display ---
st.markdown("---")
n_support = len(clf.support_vectors_)
pct_support = (n_support / len(X)) * 100
train_error = (1.0 - clf.score(X, y)) * 100

mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("Support Vectors", f"{n_support}")
mcol2.metric("SV Percentage", f"{pct_support:.1f}%")
mcol3.metric("Training Error", f"{train_error:.1f}%")

# --- Hyperplane Parameters ---
st.markdown("### Hyperplane Parameters in Projected Space")
st.latex(f"w = [{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}]^T, \\quad b = {b:.4f}")
st.latex(f"{w[0]:.4f}x_1^2 {w[1]:+.4f}x_2^2 {w[2]:+.4f}\\sqrt{{2}}x_1x_2 {b:+.4f} = 0")
