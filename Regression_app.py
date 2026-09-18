import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

st.set_page_config(page_title="Regression Loss Functions", layout="wide")
st.title("Linear Regression: The Impact of an Outlier")

st.markdown("""
The first 9 points are locked perfectly on the true line. Use the sliders next to the plot to move the **10th point** (the orange dot) anywhere on the grid. See how different loss functions react to the outlier!
""")

# --- 1. Initialize State ---
if 'active_loss' not in st.session_state:
    st.session_state.active_loss = "MSE"

# The 9 fixed points
x_fixed = np.arange(1, 10, dtype=float)
y_fixed = 0.5 * x_fixed + 2.0


# --- 2. Define Loss Functions ---
def calc_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def calc_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def calc_huber(y_true, y_pred, delta=1.5):
    err = np.abs(y_true - y_pred)
    return np.mean(np.where(err <= delta, 0.5 * err ** 2, delta * (err - 0.5 * delta)))


# --- 3. Layout: Plot on the Left, Sliders on the Right ---
col_plot, col_sliders = st.columns([4, 1])

with col_sliders:
    st.markdown("### Move the Outlier")
    st.markdown("Adjust the 10th point:")

    # Sliders to move the 10th point
    x10 = st.slider("X Coordinate", min_value=-5.0, max_value=20.0, value=10.0, step=0.5)
    y10 = st.slider("Y Coordinate", min_value=-5.0, max_value=15.0, value=7.0, step=0.5)

# Combine fixed points with the movable 10th point
x_data = np.append(x_fixed, x10)
y_data = np.append(y_fixed, y10)


# --- 4. Optimization ---
def objective(params):
    m, c = params
    y_pred = m * x_data + c
    if st.session_state.active_loss == "MSE":
        return calc_mse(y_data, y_pred)
    elif st.session_state.active_loss == "MAE":
        return calc_mae(y_data, y_pred)
    elif st.session_state.active_loss == "Huber":
        return calc_huber(y_data, y_pred)


# Optimize based on the currently selected criteria
result = opt.minimize(objective, [0.5, 2.0], method='Nelder-Mead')
m_fit, c_fit = result.x
y_pred_optimal = m_fit * x_data + c_fit

# Calculate all 3 losses for the current line
mse_val = calc_mse(y_data, y_pred_optimal)
mae_val = calc_mae(y_data, y_pred_optimal)
huber_val = calc_huber(y_data, y_pred_optimal)

# --- 5. Clickable Loss Function Boxes ---
st.markdown("### Click a box to minimize that loss function:")
col1, col2, col3 = st.columns(3)

if col1.button(f"Mean Squared Error (MSE)\n\nCurrent: {mse_val:.3f}",
               type="primary" if st.session_state.active_loss == "MSE" else "secondary",
               use_container_width=True):
    st.session_state.active_loss = "MSE"
    st.rerun()

if col2.button(f"Mean Absolute Error (MAE)\n\nCurrent: {mae_val:.3f}",
               type="primary" if st.session_state.active_loss == "MAE" else "secondary",
               use_container_width=True):
    st.session_state.active_loss = "MAE"
    st.rerun()

if col3.button(f"Huber Loss\n\nCurrent: {huber_val:.3f}",
               type="primary" if st.session_state.active_loss == "Huber" else "secondary",
               use_container_width=True):
    st.session_state.active_loss = "Huber"
    st.rerun()

# --- 6. Plotting ---
with col_plot:
    fig, ax = plt.subplots(figsize=(10, 5))

    # True Line
    x_plot = np.linspace(-5, 20, 100)
    y_true_line = 0.5 * x_plot + 2.0
    ax.plot(x_plot, y_true_line, 'k-', linewidth=2, label='True Line ($y = 0.5x + 2$)')

    # Regression Line
    y_fit_line = m_fit * x_plot + c_fit
    ax.plot(x_plot, y_fit_line, 'r--', linewidth=2, label=f'Best Fit ({st.session_state.active_loss})')

    # Data Points (Fixed)
    ax.plot(x_fixed, y_fixed, 'bo', markersize=8, label='Fixed Points')

    # Data Point (Movable)
    ax.plot(x10, y10, 'o', color='darkorange', markersize=12, label='Movable Outlier')

    # Residuals
    for i in range(len(x_data)):
        color = 'darkorange' if i == 9 else 'gray'
        alpha = 0.8 if i == 9 else 0.5
        ax.plot([x_data[i], x_data[i]], [y_data[i], m_fit * x_data[i] + c_fit],
                color=color, linestyle=':', alpha=alpha)

    ax.set_xlim([-5, 20])
    ax.set_ylim([-5, 15])
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')

    st.pyplot(fig)