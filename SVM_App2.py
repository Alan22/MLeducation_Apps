import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_blobs
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

# Setup Streamlit page layout
st.set_page_config(layout="wide", page_title="SVM RBF Kernel Visualization")
st.title("SVM with RBF Kernel")

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

gamma = st.sidebar.select_slider(
    "Gamma (Kernel Coefficient)",
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
clf = SVC(kernel='rbf', C=C, gamma=gamma)
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
col1, col2, col3 = st.columns(3)

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

# 3) RBF Kernel Contours
with col3:
    fig3, ax3 = plt.subplots(figsize=(5, 5))

    # Calculate RBF kernel shape centered at origin (0,0)
    Z_rbf = np.exp(-gamma * (xx ** 2 + yy ** 2))

    # Fixed contour levels independent of gamma
    levels = [0.05, 0.5, 0.95]

    contour = ax3.contour(xx, yy, Z_rbf, levels=levels, colors=['#000080', '#0000FF', '#ADD8E6'], linewidths=2)
    ax3.clabel(contour, inline=True, fontsize=10)

    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_title("3. RBF Kernel Contours")
    ax3.set_xticks(())
    ax3.set_yticks(())

    # Add origin marker
    ax3.scatter([0], [0], c='red', marker='x', s=100, label="Origin")
    st.pyplot(fig3)

# --- Metrics Display ---
st.markdown("---")
n_support = len(clf.support_vectors_)
pct_support = (n_support / len(X)) * 100
train_error = (1.0 - clf.score(X, y)) * 100

mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("Support Vectors", f"{n_support}")
mcol2.metric("SV Percentage", f"{pct_support:.1f}%")
mcol3.metric("Training Error", f"{train_error:.1f}%")
