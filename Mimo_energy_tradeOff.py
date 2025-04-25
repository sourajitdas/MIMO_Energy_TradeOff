# ========================
# 📚 1. Imports & Setup
# ========================
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Helper: Water-filling power allocation
def water_filling(lam, total_power):
    n = len(lam)
    sorted_lam = np.sort(lam)[::-1]
    for i in range(n):
        mu = (total_power + sum(1.0 / sorted_lam[:i+1])) / (i + 1)
        if all(mu - 1.0 / sorted_lam[j] >= 0 for j in range(i + 1)):
            return mu
    return None

# ==============================
# 🌐 2. Channel Generation
# ==============================
def generate_iid_rayleigh(m, n):
    return np.random.randn(m, n) + 1j * np.random.randn(m, n)

def generate_correlated_rayleigh(m, n, corr_coef=0.7):
    Rt = np.eye(n) * (1 - corr_coef) + corr_coef * np.ones((n, n))
    Rr = np.eye(m) * (1 - corr_coef) + corr_coef * np.ones((m, m))
    H_iid = generate_iid_rayleigh(m, n)
    return Rr @ H_iid @ Rt

def generate_rician(m, n, K=5):
    H_los = np.ones((m, n))
    H_nlos = generate_iid_rayleigh(m, n)
    return np.sqrt(K/(K+1)) * H_los + np.sqrt(1/(K+1)) * H_nlos

# ==========================
# 🧮 3. Rate Computation
# ==========================
def compute_rate(H, power_levels, sigma2):
    U, S, Vh = np.linalg.svd(H)
    lam = (S**2) / sigma2
    rates = []

    for P in power_levels:
        mu = water_filling(lam, P)
        if mu is None:
            continue
        p_alloc = np.maximum(mu - 1.0 / lam, 0)
        rate = np.sum(np.log2(1 + lam * p_alloc))
        rates.append((P, rate))
    return np.array(rates)

# =================================
# 🎛️ 4. Interactive Simulation
# =================================
def run_simulation(mimo_size, snr_inv, channel_type, num_channels=1):
    m, n = mimo_size
    sigma2 = snr_inv
    power_levels = np.linspace(0.1, 10, 100)

    all_rates = []
    for _ in range(num_channels):
        if channel_type == 'i.i.d Rayleigh':
            H = generate_iid_rayleigh(m, n)
        elif channel_type == 'Correlated Rayleigh':
            H = generate_correlated_rayleigh(m, n)
        elif channel_type == 'Rician':
            H = generate_rician(m, n)
        else:
            raise ValueError("Invalid channel type")

        rate_power = compute_rate(H, power_levels, sigma2)
        all_rates.append(rate_power)

    return all_rates

def plot_results(results, mimo_size, snr_inv, channel_type):
    plt.figure(figsize=(10, 7))
    for i, curve in enumerate(results):
        plt.plot(curve[:, 0], curve[:, 1], label=f"Channel {i+1}", alpha=0.7)
        if i == 0:  # Highlight specific points
            idxs = [10, 30, 50, 70, 90]
            for idx in idxs:
                if idx < len(curve):
                    p, r = curve[idx]
                    plt.scatter(p, r, color='black')
                    plt.text(p+0.2, r, f"{r:.2f}", fontsize=8)

    plt.xlabel("Transmit Power (Watts)")
    plt.ylabel("Achievable Rate (bits/s/Hz)")
    plt.title(f"{channel_type} | {mimo_size[0]}x{mimo_size[1]} MIMO | SNR=1/{snr_inv}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =================================
# 🖱️ 5. GUI Widgets
# =================================
mimo_dropdown = widgets.Dropdown(
    options=[(f"{m}x{n}", (m, n)) for m, n in [(2, 2), (4, 4), (8, 8)]],
    description="MIMO Size:",
)

channel_dropdown = widgets.Dropdown(
    options=['i.i.d Rayleigh', 'Correlated Rayleigh', 'Rician'],
    description="Channel Type:",
)

snr_slider = widgets.FloatSlider(
    value=1.0,
    min=0.1,
    max=5.0,
    step=0.1,
    description="SNR (1/σ²):",
)

num_channels_slider = widgets.IntSlider(
    value=1,
    min=1,
    max=5,
    step=1,
    description="Num Channels:",
)

run_button = widgets.Button(description="Run Simulation")

output_area = widgets.Output()

def on_run_button_clicked(b):
    with output_area:
        clear_output(wait=True)
        mimo_size = mimo_dropdown.value
        channel_type = channel_dropdown.value
        snr_inv = snr_slider.value
        num_channels = num_channels_slider.value

        results = run_simulation(mimo_size, snr_inv, channel_type, num_channels)
        plot_results(results, mimo_size, snr_inv, channel_type)

run_button.on_click(on_run_button_clicked)

# Display Widgets
display(mimo_dropdown, channel_dropdown, snr_slider, num_channels_slider, run_button, output_area)
