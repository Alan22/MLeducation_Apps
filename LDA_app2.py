import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# --- App Configuration ---
st.set_page_config(page_title="LDA with Pooled Covariance", layout="wide")
st.title("LDA: Pooled Covariance & Empirical Error")

st.markdown("""
This app extends LDA by allowing each class to have its own true covariance matrix. Because LDA assumes equal covariances, 
it calculates the decision boundary using the **pooled covariance matrix**. We then calculate the resulting LDA error vs. the theoretical minimum Bayes error.
""")

# --- Sidebar Controls ---
st.sidebar.header("Class 0 Parameters")
mu0_x = st.sidebar.number_input("Class 0 Mean X", value=0.0)
mu0_y = st.sidebar.number_input("Class 0 Mean Y", value=0.0)
var0_x = st.sidebar.slider("Class 0 Variance X", 0.1, 5.0, 1.0, 0.1)
var0_y = st.sidebar.slider("Class 0 Variance Y", 0.1, 5.0, 1.0, 0.1)
corr0 = st.sidebar.slider("Class 0 Correlation", -0.95, 0.95, 0.0, 0.05)

st.sidebar.header("Class 1 Parameters")
mu1_x = st.sidebar.number_input("Class 1 Mean X", value=1.0)
mu1_y = st.sidebar.number_input("Class 1 Mean Y", value=1.0)
var1_x = st.sidebar.slider("Class 1 Variance X", 0.1, 5.0, 1.0, 0.1)
var1_y = st.sidebar.slider("Class 1 Variance Y", 0.1, 5.0, 1.0, 0.1)
corr1 = st.sidebar.slider("Class 1 Correlation", -0.95, 0.95, 0.0, 0.05)

# --- Computations ---
mu0 = np.array([mu0_x, mu0_y])
mu1 = np.array([mu1_x, mu1_y])

# True Covariances
cov0_xy = corr0 * np.sqrt(var0_x * var0_y)
Sigma0 = np.array([[var0_x, cov0_xy], [cov0_xy, var0_y]])

cov1_xy = corr1 * np.sqrt(var1_x * var1_y)
Sigma1 = np.array([[var1_x, cov1_xy], [cov1_xy, var1_y]])

# 1. Pooled Covariance & Inverse
# Assuming equal priors (P(C0) = 0.5, P(C1) = 0.5)
Sigma_pooled = 0.5 * Sigma0 + 0.5 * Sigma1
Sigma_inv = np.linalg.inv(Sigma_pooled)

# 2. Normal Vector (w)
mu_diff = mu1 - mu0
w = Sigma_inv.dot(mu_diff)

# 3. Midpoint and Bias (b)
midpoint = 0.5 * (mu0 + mu1)
b = -np.dot(w, midpoint)

# --- Empirical Error Integration ---
# Expand the grid limits dynamically based on standard deviations to catch the tails
max_std = np.sqrt(max(var0_x, var0_y, var1_x, var1_y))
pad = 4 * max_std

x_min, x_max = min(mu0[0], mu1[0]) - pad, max(mu0[0], mu1[0]) + pad
y_min, y_max = min(mu0[1], mu1[1]) - pad, max(mu0[1], mu1[1]) + pad

grid_size = 400  # High resolution for integration
X, Y = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
pos = np.dstack((X, Y))

dx = (x_max - x_min) / grid_size
dy = (y_max - y_min) / grid_size
area = dx * dy

rv0 = multivariate_normal(mu0, Sigma0)
rv1 = multivariate_normal(mu1, Sigma1)

# P(x|C) * P(C)
pdf0 = rv0.pdf(pos) * 0.5
pdf1 = rv1.pdf(pos) * 0.5

# Bayes Error: Integral of min(P(x, C0), P(x, C1))
bayes_error = np.sum(np.minimum(pdf0, pdf1)) * area

# LDA Error: Integral of False Positives and False Negatives
# LDA predicts Class 1 when w^T x + b > 0
lda_pred_1 = (w[0] * X + w[1] * Y + b) > 0
lda_pred_0 = ~lda_pred_1

# Error = (Class 1 actuals predicted as 0) + (Class 0 actuals predicted as 1)
lda_error = (np.sum(pdf1[lda_pred_0]) + np.sum(pdf0[lda_pred_1])) * area

# --- UI Layout ---
# Display metrics at the top
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Bayes Error (Theoretical Min)", f"{bayes_error:.4%}")
col_m2.metric("LDA Error (Actual)", f"{lda_error:.4%}")
error_diff = lda_error - bayes_error
col_m3.metric("Error Penalty (LDA vs Bayes)", f"+{error_diff:.4%}",
              help="Difference between LDA error and optimal Bayes error. Drops to ~0 when covariances match.")

col1, col2 = st.columns([3, 3])

with col1:
    fig, ax = plt.subplots(figsize=(8, 8))

    # Use a smaller meshgrid specifically for plotting contours so it renders faster
    X_plot, Y_plot = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    pos_plot = np.dstack((X_plot, Y_plot))

    ax.contour(X_plot, Y_plot, rv0.pdf(pos_plot), levels=5, colors='blue', alpha=0.5)
    ax.contour(X_plot, Y_plot, rv1.pdf(pos_plot), levels=5, colors='red', alpha=0.5)

    ax.plot(mu0[0], mu0[1], 'bo', markersize=8, label='Class 0 Mean')
    ax.plot(mu1[0], mu1[1], 'ro', markersize=8, label='Class 1 Mean')

    # Plot Decision Boundary: w0*x + w1*y + b = 0 => y = (-b - w0*x) / w1
    if w[1] != 0:
        x_vals = np.array([x_min, x_max])
        y_vals = (-b - w[0] * x_vals) / w[1]
        ax.plot(x_vals, y_vals, 'k--', linewidth=2, label='LDA Boundary')
    else:
        ax.axvline(x=-b / w[0], color='k', linestyle='--', linewidth=2, label='LDA Boundary')

    # Plot Normal Vector
    scale = 1.0 / np.linalg.norm(w)
    ax.quiver(midpoint[0], midpoint[1], w[0] * scale, w[1] * scale, angles='xy', scale_units='xy', scale=1,
              color='green', width=0.01, label='Normal Vector (w)')
    ax.plot(midpoint[0], midpoint[1], 'ko', markersize=5)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    ax.set_title("Distributions and LDA Boundary")

    st.pyplot(fig)

with col2:
    st.subheader("The Mathematics")

    st.markdown("**1. Calculate Pooled Covariance:**")
    st.markdown(
        "Because LDA requires a single shared covariance matrix, we average the two true covariances (assuming equal priors).")
    st.latex(r"\Sigma_{pooled} = 0.5\Sigma_0 + 0.5\Sigma_1")

    st.markdown("**2. Calculate the Normal Vector ($w$):**")
    st.markdown(
        r"To find the orientation of the boundary, we multiply the inverse pooled covariance by the difference in means $(\mu_1 - \mu_0)$.")
    st.latex(
        r"\Sigma_{pooled}^{-1} = \begin{bmatrix} " + f"{Sigma_inv[0, 0]:.2f} & {Sigma_inv[0, 1]:.2f} \\\\ {Sigma_inv[1, 0]:.2f} & {Sigma_inv[1, 1]:.2f}" + r" \end{bmatrix}")
    st.latex(r"(\mu_1 - \mu_0) = \begin{bmatrix} " + f"{mu_diff[0]:.2f} \\\\ {mu_diff[1]:.2f}" + r" \end{bmatrix}")

    st.markdown("Expanding the matrix multiplication:")
    st.latex(r"""
    w = \begin{bmatrix} S_{11} & S_{12} \\ S_{21} & S_{22} \end{bmatrix} 
    \begin{bmatrix} d_x \\ d_y \end{bmatrix} 
    = \begin{bmatrix} (S_{11} \cdot d_x) + (S_{12} \cdot d_y) \\ (S_{21} \cdot d_x) + (S_{22} \cdot d_y) \end{bmatrix}
    """)
    st.latex(
        r"w = \begin{bmatrix} " + f"({Sigma_inv[0, 0]:.2f} \\times {mu_diff[0]:.2f}) + ({Sigma_inv[0, 1]:.2f} \\times {mu_diff[1]:.2f}) \\\\ " +
        f"({Sigma_inv[1, 0]:.2f} \\times {mu_diff[0]:.2f}) + ({Sigma_inv[1, 1]:.2f} \\times {mu_diff[1]:.2f})" + r" \end{bmatrix}" +
        r" = \begin{bmatrix} " + f"{w[0]:.2f} \\\\ {w[1]:.2f}" + r" \end{bmatrix}")

    st.markdown("**3. Construct the Linear Equation ($w^Tx + b = 0$):**")
    st.markdown(
        "The boundary passes through the midpoint $x_0$ of the two distributions. By substituting $x_0$ into the standard linear equation $w^T x + b = 0$, we can solve for the bias term $b$.")
    st.latex(
        r"x_0 = \frac{\mu_0 + \mu_1}{2} = \begin{bmatrix} " + f"{midpoint[0]:.2f} \\\\ {midpoint[1]:.2f}" + r" \end{bmatrix}")

    st.markdown("Set the equation equal to zero at the midpoint:")
    st.latex(r"w^T x_0 + b = 0 \implies b = -w^T x_0")
    st.latex(
        r"b = -(" + f"{w[0]:.2f} \\times {midpoint[0]:.2f} + {w[1]:.2f} \\times {midpoint[1]:.2f}" + f") = {b:.2f}")

    st.markdown("Which gives us the final decision boundary equation:")
    st.latex(f"({w[0]:.2f})x + ({w[1]:.2f})y + ({b:.2f}) = 0")