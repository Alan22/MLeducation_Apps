import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kernel Ridge Regression Demo", layout="wide")

st.title("Kernel Ridge Regression Visualization")
st.markdown("This app visualizes the components of Kernel Ridge Regression, inspired by **image_6cd75a.png**.")

# Sidebar Controls
st.sidebar.header("KRR Parameters")

# Lambda (regularization) parameter
# Using log scale for lambda is standard practice
log_lambda = st.sidebar.slider(
    "Regularization ($\lambda$): log10 scale",
    min_value=-5.0, max_value=2.0, value=-2.0, step=0.1
)
lambda_reg = 10 ** log_lambda
st.sidebar.write(f"**Current $\lambda$:** `{lambda_reg:.5f}`")

# Bandwidth (sigma) parameter for the RBF kernel
bandwidth = st.sidebar.slider(
    "Kernel Bandwidth ($\sigma$)",
    min_value=0.1, max_value=3.0, value=0.5, step=0.1
)

# 1. Generate synthetic data resembling the relationship in image_6cd75a.png
np.random.seed(42)
n_samples = 50
# X values between -3 and 3
X = np.sort(np.random.uniform(-3, 3, n_samples))
# Creating a target function that dips around -1.5 and peaks around +1.5
y_true = np.sin(1.5 * X) - 0.5 * np.cos(2.5 * X)
y = y_true + np.random.normal(0, 0.25, n_samples)

X_col = X.reshape(-1, 1)

# 2. Define the RBF Kernel function
def rbf_kernel(Xa, Xb, sigma):
    # Vectorized calculation of squared Euclidean distance
    dist_sq = (Xa**2) + (Xb.T**2) - 2 * np.dot(Xa, Xb.T)
    return np.exp(-dist_sq / (2 * sigma**2))

# 3. Calculate KRR coefficients manually
# K is the Kernel matrix (n_samples x n_samples)
K = rbf_kernel(X_col, X_col, bandwidth)

# Solve for alpha: alpha = (K + lambda * I)^-1 * y
alpha = np.linalg.inv(K + lambda_reg * np.eye(n_samples)).dot(y)

# 4. Generate grid for plotting continuous curves
X_grid = np.linspace(-3.2, 3.2, 300).reshape(-1, 1)
K_grid = rbf_kernel(X_grid, X_col, bandwidth) # Shape: (300, 50)
y_pred = K_grid.dot(alpha)

# 5. Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Plot settings
for ax in axes:
    ax.set_xlim(-3, 3)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.scatter(X, y, color='blue', zorder=5, label='Data points')

# --- Left Plot: Unweighted Kernels ---
axes[0].set_ylim(-2, 2)
axes[0].set_title("Unweighted Kernels at each data point")
# Plot each unweighted kernel (K(x, x_i))
for i in range(n_samples):
    axes[0].plot(X_grid, K_grid[:, i], linewidth=1, alpha=0.8)

# --- Middle Plot: Weighted Kernels ---
# Dynamically scale Y-axis for middle plot based on max weight to prevent clipping
max_weight = np.max(np.abs(alpha))
y_lim_mid = max(3, max_weight * 1.2)
axes[1].set_ylim(-y_lim_mid, y_lim_mid)
axes[1].set_title(r"Weighted Kernels ($\alpha_i K(x, x_i)$)")
# Plot each weighted kernel (alpha_i * K(x, x_i))
for i in range(n_samples):
    axes[1].plot(X_grid, alpha[i] * K_grid[:, i], linewidth=1, alpha=0.8)

# --- Right Plot: Final Prediction ---
axes[2].set_ylim(-2, 2)
axes[2].set_title("Final KRR Prediction (Sum of Weighted Kernels)")
# Plot the final prediction sum
axes[2].plot(X_grid, y_pred, color=[0.8, 0.3, 0.1], linestyle='dotted', linewidth=3, zorder=4, label='Prediction')
axes[2].legend(loc='lower right')

plt.tight_layout()

# Render plot in Streamlit
st.pyplot(fig)
