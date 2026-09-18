import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import trapezoid

st.set_page_config(layout="wide", page_title="Empirical Risk Minimization")

st.title("Empirical Risk Minimization (ERM) Basics")
st.markdown("Explore how different loss functions evaluate a decision boundary $\\theta$ for a 1D binary classification problem. We assume equal class priors.")

# --- SIDEBAR: DATA GENERATION CONTROLS ---
st.sidebar.header("1. Data Generation")
mu_neg = st.sidebar.slider("Mean Class -1 ($\mu_{-1}$)", -5.0, 5.0, -1.0, 0.1)
mu_pos = st.sidebar.slider("Mean Class +1 ($\mu_{+1}$)", -5.0, 5.0, 1.0, 0.1)
n = st.sidebar.number_input("Samples per class ($n$)", min_value=1, max_value=100, value=10)

# Initialize data in session state so it doesn't regenerate on every slider tweak
if 'X_neg' not in st.session_state or 'X_pos' not in st.session_state:
    st.session_state.X_neg = np.random.normal(mu_neg, 1.0, n)
    st.session_state.X_pos = np.random.normal(mu_pos, 1.0, n)

if st.sidebar.button("Generate New Random Sample"):
    st.session_state.X_neg = np.random.normal(mu_neg, 1.0, n)
    st.session_state.X_pos = np.random.normal(mu_pos, 1.0, n)

# --- SIDEBAR: LOSS CONTROLS ---
st.sidebar.header("2. Loss Controls")
loss_type = st.sidebar.selectbox("Loss Function", ["0-1", "Hinge", "Logistic"])
theta = st.sidebar.slider("Decision Boundary ($\\theta$)", -5.0, 5.0, 0.0, 0.1)

# Retrieve data
X_neg = st.session_state.X_neg
X_pos = st.session_state.X_pos

# --- HELPER: LOSS FUNCTION CALCULATOR ---
def calculate_loss(y, x, boundary, loss_name):
    margin = y * (x - boundary)
    if loss_name == "0-1":
        return np.where(margin <= 0, 1.0, 0.0)
    elif loss_name == "Hinge":
        return np.maximum(0.0, 1.0 - margin)
    elif loss_name == "Logistic":
        return np.log2(1.0 + np.exp(-margin))

# --- PLOT 1: DISTRIBUTIONS & DATA ---
st.subheader("Distributions & Sample Data")
fig1, ax1 = plt.subplots(figsize=(10, 3))

x_grid = np.linspace(-8, 8, 1000)
pdf_neg = norm.pdf(x_grid, loc=mu_neg, scale=1.0)
pdf_pos = norm.pdf(x_grid, loc=mu_pos, scale=1.0)

ax1.plot(x_grid, pdf_neg, color='red', label='Class -1 PDF')
ax1.plot(x_grid, pdf_pos, color='blue', label='Class +1 PDF')

ax1.scatter(X_neg, np.zeros_like(X_neg) - 0.02, color='red', marker='o', alpha=0.7, label='Class -1 Data')
ax1.scatter(X_pos, np.zeros_like(X_pos) - 0.02, color='blue', marker='x', alpha=0.7, label='Class +1 Data')

ax1.axvline(theta, color='green', linestyle='--', label=f'Current Boundary ($\\theta={theta:.2f}$)')

ax1.set_xlim([-8, 8])
ax1.set_ylim([-0.05, max(max(pdf_neg), max(pdf_pos)) + 0.1])
ax1.set_xlabel("x")
ax1.set_ylabel("Density")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# --- OPTIMIZATION TO FIND MINIMA ---
search_grid = np.linspace(-8, 8, 1600)

# 1. Empirical Minimum Search
loss_neg_emp = calculate_loss(-1, X_neg[:, None], search_grid[None, :], loss_type)
loss_pos_emp = calculate_loss(1, X_pos[:, None], search_grid[None, :], loss_type)
emp_risk_grid = (np.sum(loss_neg_emp, axis=0) + np.sum(loss_pos_emp, axis=0)) / (2 * n)
best_emp_idx = np.argmin(emp_risk_grid)
theta_emp_opt = search_grid[best_emp_idx]

# 2. Theoretical Minimum Search
x_int_grid = np.linspace(-20, 20, 2000)[:, None]
p_x_neg = norm.pdf(x_int_grid, loc=mu_neg, scale=1.0)
p_x_pos = norm.pdf(x_int_grid, loc=mu_pos, scale=1.0)

loss_neg_theo = calculate_loss(-1, x_int_grid, search_grid[None, :], loss_type)
loss_pos_theo = calculate_loss(1, x_int_grid, search_grid[None, :], loss_type)

exp_loss_neg = trapezoid(loss_neg_theo * p_x_neg, x_int_grid[:, 0], axis=0)
exp_loss_pos = trapezoid(loss_pos_theo * p_x_pos, x_int_grid[:, 0], axis=0)
theo_risk_grid = 0.5 * exp_loss_neg + 0.5 * exp_loss_pos
best_theo_idx = np.argmin(theo_risk_grid)
theta_theo_opt = search_grid[best_theo_idx]

# --- PLOT 2: LOSS FUNCTIONS & STEM PLOT ---
st.subheader(f"{loss_type} Loss Evaluation")
fig2, ax2 = plt.subplots(figsize=(10, 4))

loss_grid_neg = calculate_loss(-1, x_grid, theta, loss_type)
loss_grid_pos = calculate_loss(1, x_grid, theta, loss_type)

ax2.plot(x_grid, loss_grid_neg, color='red', alpha=0.3, linewidth=3, label='Loss surface for Class -1')
ax2.plot(x_grid, loss_grid_pos, color='blue', alpha=0.3, linewidth=3, label='Loss surface for Class +1')

loss_data_neg = calculate_loss(-1, X_neg, theta, loss_type)
loss_data_pos = calculate_loss(1, X_pos, theta, loss_type)

ax2.stem(X_neg, loss_data_neg, linefmt='r-', markerfmt='ro', basefmt=' ')
ax2.stem(X_pos, loss_data_pos, linefmt='b-', markerfmt='bx', basefmt=' ')

# Vertical lines for decision boundaries
ax2.axvline(theta, color='green', linestyle='--', linewidth=2, label=f'Current ($\\theta={theta:.2f}$)')
ax2.axvline(theta_emp_opt, color='darkorange', linestyle=':', linewidth=2.5, label=f'Empirical Min ($\\hat{{\\theta}}^*={theta_emp_opt:.2f}$)')
ax2.axvline(theta_theo_opt, color='purple', linestyle='-.', linewidth=2.5, label=f'Theoretical Min ($\\theta^*={theta_theo_opt:.2f}$)')

ax2.set_xlim([-8, 8])
ax2.set_ylim([-0.1, max(np.max(loss_grid_neg), np.max(loss_grid_pos)) + 0.5])
ax2.set_xlabel("x")
ax2.set_ylabel("Loss / Penalty")
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# --- CALCULATE CURRENT RISKS ---
empirical_risk = (np.sum(loss_data_neg) + np.sum(loss_data_pos)) / (2 * n)

loss_int_neg_curr = calculate_loss(-1, x_int_grid[:, 0], theta, loss_type)
loss_int_pos_curr = calculate_loss(1, x_int_grid[:, 0], theta, loss_type)
expected_loss_neg_curr = trapezoid(loss_int_neg_curr * p_x_neg[:, 0], x_int_grid[:, 0])
expected_loss_pos_curr = trapezoid(loss_int_pos_curr * p_x_pos[:, 0], x_int_grid[:, 0])
theoretical_risk = 0.5 * expected_loss_neg_curr + 0.5 * expected_loss_pos_curr

# --- DISPLAY METRICS ---
col1, col2 = st.columns(2)
col1.metric(
    label="**Empirical Risk ($\\hat{R}$)**",
    value=f"{empirical_risk:.4f}",
    delta=f"Min: {emp_risk_grid[best_emp_idx]:.4f} at $\\hat{{\\theta}}^*={theta_emp_opt:.2f}$",
    delta_color="off",
    help="The average loss calculated directly from the generated data points."
)
col2.metric(
    label="**Theoretical Risk ($R$)**",
    value=f"{theoretical_risk:.4f}",
    delta=f"Min: {theo_risk_grid[best_theo_idx]:.4f} at $\\theta^*={theta_theo_opt:.2f}$",
    delta_color="off",
    help="The expected loss calculated by integrating over the true underlying distributions."
)