import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.svm import SVC

st.set_page_config(page_title="OSM Hyperplane Demo", layout="centered")

st.title("OSM / Linear SVM Hyperplane")
st.markdown(
    "Visualizing the Optimal Separating Margin for **Multivariate Normal** data, "
    "showing density contours, margins, and distance lines (red) for points inside the margin."
)

# Sidebar Controls
st.sidebar.header("Controls")

# Logarithmic slider for C parameter
log_C = st.sidebar.slider(
    "Regularization Parameter: log10(C)",
    min_value=-3.0, max_value=3.0, value=0.0, step=0.1
)
C_param = 10 ** log_C
st.sidebar.write(f"**Current C:** `{C_param:.3f}`")

# Variance slider for the Gaussian distributions
variance = st.sidebar.slider(
    "Distribution Variance ($\sigma^2$)",
    min_value=0.1, max_value=5.0, value=1.5, step=0.1
)

generate_btn = st.sidebar.button("Generate New Sample")

# Fixed mean centers for the two classes
mean_neg = np.array([-2.0, -2.0])
mean_pos = np.array([2.0, 2.0])

# Session state handling to preserve data when tweaking sliders
if ('X' not in st.session_state or
        'variance' not in st.session_state or
        st.session_state['variance'] != variance or
        generate_btn):
    cov = variance * np.eye(2)
    X_neg = np.random.multivariate_normal(mean_neg, cov, 10)
    X_pos = np.random.multivariate_normal(mean_pos, cov, 10)

    st.session_state['X'] = np.vstack([X_neg, X_pos])
    st.session_state['y'] = np.hstack([-np.ones(10), np.ones(10)])
    st.session_state['variance'] = variance

X = st.session_state['X']
y = st.session_state['y']

# Determine symmetric axis boundaries centered at 0 [-max_abs, +max_abs]
padding = 1.5
all_data = np.concatenate([X[:, 0], X[:, 1]])
max_abs_val = np.max(np.abs(all_data)) + padding

min_val = -max_abs_val
max_val = max_abs_val

# Generate grid for density contours and SVM decision boundary
grid_res = 200
x_grid = np.linspace(min_val, max_val, grid_res)
y_grid = np.linspace(min_val, max_val, grid_res)
XX, YY = np.meshgrid(x_grid, y_grid)
grid_coords = np.vstack([XX.ravel(), YY.ravel()]).T

# Evaluate Multivariate Normal PDFs for both classes
cov = variance * np.eye(2)
rv_neg = multivariate_normal(mean_neg, cov)
rv_pos = multivariate_normal(mean_pos, cov)

Z_neg = rv_neg.pdf(grid_coords).reshape(XX.shape)
Z_pos = rv_pos.pdf(grid_coords).reshape(XX.shape)

# Fit Linear SVM / Optimal Separating Margin
clf = SVC(kernel='linear', C=C_param)
clf.fit(X, y)

# Evaluate decision function over grid
Z_svm = clf.decision_function(grid_coords).reshape(XX.shape)

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))

# Reference zero-lines to highlight symmetry around 0
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

# 1. Density Contours for underlying distributions
ax.contour(XX, YY, Z_neg, levels=5, colors='blue', alpha=0.3, linestyles='dotted')
ax.contour(XX, YY, Z_pos, levels=5, colors='green', alpha=0.3, linestyles='dotted')

# 2. Scatter plot of data points
ax.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', edgecolor='k', s=70, label='Class -1', zorder=4)
ax.scatter(X[y == 1, 0], X[y == 1, 1], color='green', edgecolor='k', s=70, label='Class +1', zorder=4)

# 3. Decision Boundary (0) and Margins (-1, 1)
ax.contour(
    XX, YY, Z_svm,
    colors=['blue', 'black', 'green'],
    levels=[-1, 0, 1],
    alpha=0.8,
    linestyles=['--', '-', '--'],
    linewidths=[1.5, 2, 1.5],
    zorder=3
)

# 4. Orthogonal projections for points inside the margin (red lines)
w = clf.coef_[0]
decision_values = clf.decision_function(X)
margin_violations = y * decision_values < 1

for i in range(len(X)):
    if margin_violations[i]:
        xi = X[i]
        yi = y[i]
        # Calculate projection onto correct margin w^T * x + b = y_i
        t = (yi - decision_values[i]) / np.dot(w, w)
        pi = xi + t * w
        ax.plot([xi[0], pi[0]], [xi[1], pi[1]], color='red', linestyle='-', linewidth=2, alpha=0.8, zorder=5)

# Set symmetric axes centered strictly around 0
ax.set_xlim([min_val, max_val])
ax.set_ylim([min_val, max_val])
ax.set_aspect('equal', adjustable='box')

ax.set_xlabel('$X_1$')
ax.set_ylabel('$X_2$')
ax.set_title(f"OSM Hyperplane (C = {C_param:.3f}, Variance = {variance:.1f})")
ax.legend(loc='upper left')
ax.grid(True, linestyle=':', alpha=0.4)

st.pyplot(fig)
