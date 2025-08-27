#Dynamics of Conflict-Intimacy Alternations in Marriage: A Lévy-Driven and State-Dependent Jump SDE Framework
#By Lipu Zhang, zhanglipu@cuz.edu.cn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm

# ========================
# Font configuration (ensure proper rendering)
# ========================
preferred_fonts = ["Segoe UI", "Arial", "Helvetica", "Times New Roman"]
font_paths = fm.findSystemFonts()
font_names = []
for path in font_paths:
    try:
        font_prop = fm.FontProperties(fname=path)
        font_names.append(font_prop.get_name())
    except:
        continue
available_fonts = [f for f in preferred_fonts if f in font_names]
if not available_fonts:
    available_fonts = ["sans-serif"]

# ========================
# Journal-level plotting configuration
# ========================
rcParams.update({
    "font.family": available_fonts,
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "lines.linewidth": 1.8,
    "figure.figsize": (8, 5),
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.format": "png",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.transparent": False,
    "axes.edgecolor": "#333333"
})

# ========================
# Model parameters (randomized impact amplitudes)
# ========================
T = 365  # Total simulation days
dt = 0.1  # Time step (days)
steps = int(T / dt)  # Total steps
t = np.linspace(0, T, steps)  # Time array

# Intimacy level parameters
I_baseline = 0.80  # Intimacy baseline
r_I = 0.10  # Rate of intimacy recovery to baseline
# Impact amplitude ranges (core randomization: amplitudes vary within ranges)
Delta_I_small_range = [0.05, 0.15]  # Negative impact range of small conflicts on intimacy
Delta_I_large_range = [0.15, 0.25]  # Negative impact range of large conflicts on intimacy
Delta_I_positive_range = [0.1, 0.2]  # Positive impact range of positive events on intimacy

# Conflict intensity parameters
r_C = 0.15  # Natural decay rate of conflict
# Event occurrence intensities (Poisson process parameters controlling randomness)
lambda_small = 0.05  # Probability of small conflict per day
lambda_large = 1 / 100  # Large conflict occurs on average once every 100 days
lambda_positive = 0.06  # Probability of positive event per day
# Conflict intensity impact ranges (randomized)
Delta_C_small_range = [0.3, 0.5]  # Positive impact range of small conflicts on conflict intensity
Delta_C_large_range = [0.6, 0.8]  # Positive impact range of large conflicts on conflict intensity
Delta_C_p_range = [0.4, 0.6]  # Negative impact range (mitigation) of positive events on conflict intensity (Γₚ)

# ========================
# Generate events: temporal randomness + amplitude randomness
# ========================
np.random.seed(42)  # Fix random seed for reproducibility

# 1. Time masks for event occurrences (Poisson process: controlling "when events happen")
small_conflict_mask = np.random.rand(steps) < lambda_small * dt  # Time points for small conflicts (True/False)
large_conflict_mask = np.random.rand(steps) < lambda_large * dt  # Time points for large conflicts
positive_event_mask = np.random.rand(steps) < lambda_positive * dt  # Time points for positive events


# 2. Generate random amplitudes for occurred events (controlling "how much impact")
def random_amplitude(mask, amplitude_range):
    """Generate random amplitudes for occurred events (where mask is True), 0 for non-occurrences"""
    amplitudes = np.zeros_like(mask, dtype=float)  # Initialize amplitude array (all zeros)
    event_indices = np.where(mask)[0]  # Find indices where events occur
    # Generate random amplitudes within specified range for event positions
    amplitudes[event_indices] = np.random.uniform(
        low=amplitude_range[0],
        high=amplitude_range[1],
        size=len(event_indices)
    )
    return amplitudes


# Generate random impact amplitudes for all event types (raw amplitudes, unadjusted for state dependence)
Delta_I_small_raw = random_amplitude(small_conflict_mask, Delta_I_small_range)
Delta_I_large_raw = random_amplitude(large_conflict_mask, Delta_I_large_range)
Delta_I_positive_raw = random_amplitude(positive_event_mask, Delta_I_positive_range)

Delta_C_small_raw = random_amplitude(small_conflict_mask, Delta_C_small_range)
Delta_C_large_raw = random_amplitude(large_conflict_mask, Delta_C_large_range)
Delta_C_p_raw = random_amplitude(positive_event_mask, Delta_C_p_range)  # Raw impact amplitudes for Γₚ

# ========================
# Dynamical simulation (with state-dependent jump amplitudes)
# ========================
I = np.ones(steps) * I_baseline  # Initial intimacy level = baseline
C = np.zeros(steps)  # Initial conflict intensity = 0

for i in range(1, steps):
    # Current state (before event occurrence)
    current_I = I[i - 1]
    current_C = C[i - 1]

    # ------------------------
    # 1. Intimacy level update (state-dependent jumps)
    # ------------------------
    # State-dependent jump amplitude adjustment (dynamically limit impact based on current I)
    delta_I_small = min(Delta_I_small_raw[i], current_I)  # Negative impact not exceeding current intimacy
    delta_I_large = min(Delta_I_large_raw[i], current_I)  # Negative impact not exceeding current intimacy
    delta_I_positive = min(Delta_I_positive_raw[i], 1 - current_I)  # Positive impact not exceeding remaining capacity

    # Continuous term + state-dependent jump terms
    dI_dt = r_I * (I_baseline - current_I)
    dI = dI_dt * dt - delta_I_small - delta_I_large + delta_I_positive
    I[i] = current_I + dI  # No need for clipping as jump amplitudes are state-constrained

    # ------------------------
    # 2. Conflict intensity update (state-dependent jumps)
    # ------------------------
    # State-dependent jump amplitude adjustment (dynamically limit impact based on current C)
    delta_C_small = min(Delta_C_small_raw[i], 1 - current_C)  # Positive impact not exceeding remaining capacity
    delta_C_large = min(Delta_C_large_raw[i], 1 - current_C)  # Positive impact not exceeding remaining capacity
    delta_C_p = min(Delta_C_p_raw[i],
                    current_C)  # Negative impact of positive events on conflict (Γₚ), not exceeding current intensity

    # Continuous term + state-dependent jump terms
    dC_dt = -r_C * current_C
    dC = dC_dt * dt + delta_C_small + delta_C_large - delta_C_p  # Using Γₚ corresponding impact amplitude
    C[i] = current_C + dC  # No need for clipping as jump amplitudes are state-constrained


# ========================
# Trend line smoothing (reduce noise, highlight trends)
# ========================
def smooth(x, window=100):
    """Moving average to smooth the curve"""
    return np.convolve(x, np.ones(window) / window, mode='same')


I_smooth = smooth(I)  # Intimacy level trend line
C_smooth = smooth(C)  # Conflict intensity trend line

# ========================
# Extract key event time points (for annotation)
# ========================
large_times = t[large_conflict_mask]  # Times of large conflict occurrences
positive_times = t[positive_event_mask]  # Times of positive event occurrences

# ========================
# Plotting: Intimacy level + Conflict intensity dynamics
# ========================
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.1})

# Upper panel: Intimacy level
ax1.plot(t, I, color="#1f77b4", alpha=0.15, label="Raw intimacy level")
ax1.plot(t, I_smooth, color="#1f77b4", linestyle="-", label="Smoothed trend")
ax1.axhline(y=I_baseline, color="#7f7f7f", linestyle="--", linewidth=0.8, label="Baseline level")

# Annotate major conflicts (red arrows)
for idx, time in enumerate(large_times[:5]):  # Annotate first 5 to avoid overcrowding
    i = np.argmin(np.abs(t - time))  # Find corresponding index for time
    ax1.annotate(
        f"Major conflict {idx + 1}",
        xy=(time, I[i]),
        xytext=(time, I[i] - 0.15),
        arrowprops=dict(facecolor="#d62728", shrink=0.05, width=0.5, headwidth=4),
        ha='center',
        va='top',
        fontsize=9,
        color="#d62728"
    )

# Annotate positive events (green arrows)
for idx, time in enumerate(positive_times[::8]):  # Annotate every 8th to avoid overcrowding
    i = np.argmin(np.abs(t - time))
    ax1.annotate(
        f"Positive event {idx + 1}",
        xy=(time, I[i]),
        xytext=(time, I[i] + 0.12),
        arrowprops=dict(facecolor="#2ca02c", shrink=0.05, width=0.5, headwidth=4),
        ha='center',
        va='bottom',
        fontsize=9,
        color="#2ca02c"
    )

# Lower panel: Conflict intensity
ax2.plot(t, C, color="#d62728", alpha=0.15, label="Raw conflict intensity")
ax2.plot(t, C_smooth, color="#d62728", linestyle="-", label="Smoothed trend")

# Highlight event impact periods with shaded regions
for time in large_times:
    ax2.axvspan(time, time + 3, color="#ffcccc", alpha=0.5)  # Major conflict shading
for time in positive_times:
    ax2.axvspan(time, time + 2, color="#ccffcc", alpha=0.5)  # Positive event shading

# Axis and legend settings
ax1.set_ylabel("Intimacy Level")
ax1.set_ylim(0.6, 1.0)
ax1.set_yticks(np.arange(0.6, 1.01, 0.2))
ax1.legend(loc="lower right", frameon=False, ncol=3, columnspacing=1)

ax2.set_ylabel("Conflict Intensity")
ax2.set_xlabel("Time (days)")
ax2.set_xlim(0, T)
ax2.set_ylim(0, 0.8)
ax2.set_yticks(np.arange(0, 0.81, 0.4))
ax2.legend(loc="upper right", frameon=False, ncol=2, columnspacing=1)

fig.suptitle("Dynamic Changes in Intimacy Level and Conflict Intensity (State-Dependent Jump Amplitudes)", y=0.98)

# Save figure
plt.savefig("intimacy_conflict_state_dependent_english.png",
            dpi=600,
            bbox_inches="tight",
            pil_kwargs={"compress_level": 9})

plt.show()

