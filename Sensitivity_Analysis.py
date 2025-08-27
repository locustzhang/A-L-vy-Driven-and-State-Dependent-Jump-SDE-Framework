#%%灵敏性分析Dynamics of Conflict-Intimacy Alternations in Marriage: A Lévy-Driven and State-Dependent Jump SDE Framework
#By Lipu Zhang, zhanglipu@cuz.edu.cn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import warnings
from scipy import stats as sp_stats
from scipy.stats import chi2_contingency

# ---------- 1. Ignore Irrelevant Warnings ---------- #
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='matplotlib')

# ---------- 2. Font Configuration (English Only) ---------- #
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'savefig.dpi': 600,
    'axes.unicode_minus': False,
    'mathtext.fontset': 'cm'
})

# ---------- 3. Color Scheme (Paper Style) ---------- #
COLORS = {
    'intimacy': '#1f77b4',
    'conflict': '#ff7f0e',
    'baseline': '#7f7f7f',
    'highlight': '#17becf'
}


# ---------- 4. Parameter Validation (Paper Table 4 Ranges) ---------- #
def validate_model_input(params: dict) -> dict:
    """Validate parameters against Paper Table 4 (only physical bounds)"""
    violations = []
    checks = [
        ('lambda_s', lambda x: 0.05 <= x <= 0.15, "λ_s ∈ [0.05, 0.15] (Paper Tab.4)"),
        ('lambda_l', lambda x: 0.008 <= x <= 0.012, "λ_l ∈ [0.008, 0.012] (Paper Tab.4)"),
        ('lambda_p', lambda x: 0.04 <= x <= 0.08, "λ_p ∈ [0.04, 0.08] (Paper Tab.4)"),
        ('r_I', lambda x: 0.08 <= x <= 0.12, "r_I ∈ [0.08, 0.12] (Paper Tab.4)"),
        ('r_C', lambda x: 0.12 <= x <= 0.18, "r_C ∈ [0.12, 0.18] (Paper Tab.4)"),
        ('Delta_I_s', lambda x: 0.08 <= x <= 0.12, "ΔI_s ∈ [0.08, 0.12] (Paper Tab.4)"),
        ('Delta_I_l', lambda x: 0.18 <= x <= 0.22, "ΔI_l ∈ [0.18, 0.22] (Paper Tab.4)"),
        ('Delta_I_p', lambda x: 0.12 <= x <= 0.18, "ΔI_p ∈ [0.12, 0.18] (Paper Logic)"),
        ('Delta_C_s', lambda x: 0.10 <= x <= 0.14, "ΔC_s ∈ [0.10, 0.14] (Paper Logic)"),
        ('Delta_C_l', lambda x: 0.32 <= x <= 0.38, "ΔC_l ∈ [0.32, 0.38] (Paper Logic)"),
        ('Delta_C_p', lambda x: 0.22 <= x <= 0.28, "ΔC_p ∈ [0.22, 0.28] (Paper Logic)"),
        ('I_baseline', lambda x: 0.75 <= x <= 0.85, "I_baseline ∈ [0.75, 0.85] (Paper Tab.4)")
    ]
    for key, cond, msg in checks:
        if key in params and not cond(params[key]):
            violations.append(msg)
    if violations:
        raise ValueError(f"Parameter validation failed:\n" + "\n".join(violations))
    return params


# ---------- 5. Core Model (Paper Assumptions + State-Dependent Impacts) ---------- #
class MaritalModel:
    def __init__(self, params: dict = None):
        """Default parameters from Paper Table 4"""
        self.default_params = {
            'I_baseline': 0.80, 'r_I': 0.10, 'r_C': 0.15,
            'lambda_s': 0.09, 'lambda_l': 1 / 100, 'lambda_p': 0.06,
            'Delta_I_s': 0.10, 'Delta_I_l': 0.20, 'Delta_I_p': 0.15,
            'Delta_C_s': 0.12, 'Delta_C_l': 0.35, 'Delta_C_p': 0.25
        }
        input_params = params or {}
        all_params = {**self.default_params, **input_params}
        self.params = validate_model_input(all_params)
        # Cache key parameters
        self.I_base = self.params['I_baseline']
        self.r_I, self.r_C = self.params['r_I'], self.params['r_C']
        self.lambda_s, self.lambda_l, self.lambda_p = \
            self.params['lambda_s'], self.params['lambda_l'], self.params['lambda_p']
        self.Delta = {
            'I_s': self.params['Delta_I_s'], 'I_l': self.params['Delta_I_l'], 'I_p': self.params['Delta_I_p'],
            'C_s': self.params['Delta_C_s'], 'C_l': self.params['Delta_C_l'], 'C_p': self.params['Delta_C_p']
        }

    def simulate(self, T: int = 365, num_simulations: int = 150, random_seed: int = 42) -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Simulation with Paper Assumptions (Poisson events + Uniform impacts)"""
        if random_seed is not None:
            np.random.seed(random_seed)
        t = np.arange(T)
        I = np.full((num_simulations, T), self.I_base)
        C = np.zeros((num_simulations, T))

        for sim in range(num_simulations):
            for day in range(1, T):
                # Deterministic drift
                I_prev, C_prev = I[sim, day - 1], C[sim, day - 1]
                I_drift = I_prev + self.r_I * (self.I_base - I_prev)
                C_drift = C_prev - self.r_C * C_prev

                # Small conflicts (Poisson + Uniform)
                num_small = np.random.poisson(self.lambda_s)
                for _ in range(num_small):
                    max_impact_I = min(self.Delta['I_s'], I_drift)
                    impact_I = np.random.uniform(0, max_impact_I)
                    max_impact_C = min(self.Delta['C_s'], 1 - C_drift)
                    impact_C = np.random.uniform(0, max_impact_C)
                    I_drift -= impact_I
                    C_drift += impact_C

                # Large conflicts (Poisson + Uniform)
                num_large = np.random.poisson(self.lambda_l)
                for _ in range(num_large):
                    max_impact_I = min(self.Delta['I_l'], I_drift)
                    impact_I = np.random.uniform(0, max_impact_I)
                    max_impact_C = min(self.Delta['C_l'], 1 - C_drift)
                    impact_C = np.random.uniform(0, max_impact_C)
                    I_drift -= impact_I
                    C_drift += impact_C

                # Positive events (Poisson + Uniform)
                num_pos = np.random.poisson(self.lambda_p)
                for _ in range(num_pos):
                    max_impact_I = min(self.Delta['I_p'], 1 - I_drift)
                    impact_I = np.random.uniform(0, max_impact_I)
                    max_impact_C = min(self.Delta['C_p'], C_drift)
                    impact_C = np.random.uniform(0, max_impact_C)
                    I_drift += impact_I
                    C_drift -= impact_C

                # Bounds
                I[sim, day] = np.clip(I_drift, 0.001, 0.999)
                C[sim, day] = np.clip(C_drift, 0.001, 0.999)

        return t, I, C

    def get_statistics(self, I: np.ndarray, C: np.ndarray) -> dict:
        """Paper-focused metrics"""
        num_sim, T = I.shape
        stats = {}
        stats['I_mean'] = np.mean(I, axis=1)
        stats['I_base_dev'] = np.mean(np.abs(I - self.I_base), axis=1)
        stats['C_mean'] = np.mean(C, axis=1)
        stats['C_max'] = np.max(C, axis=1)
        return stats

    def analyze_distributions(self, I: np.ndarray, C: np.ndarray) -> list:
        """Distribution analysis for transparency (no validation)"""
        T = I.shape[1]
        num_sim = I.shape[0]
        messages = []
        jump_thresh = 0.005

        # 1. Poisson Process Reference
        for event_type, lambda_val, desc in [
            ('Small Conflict', self.lambda_s, 'Negative I jump'),
            ('Positive Event', self.lambda_p, 'Positive I jump'),
            ('Large Conflict', self.lambda_l, 'Positive C jump')
        ]:
            event_counts = []
            for sim in range(num_sim):
                if event_type in ['Small Conflict', 'Positive Event']:
                    diffs = np.diff(I[sim, :])
                    jumps = np.sum(diffs < -jump_thresh) if event_type == 'Small Conflict' else np.sum(
                        diffs > jump_thresh)
                else:
                    diffs = np.diff(C[sim, :])
                    jumps = np.sum(diffs > jump_thresh * 2)
                event_counts.append(jumps)

            event_counts = np.array(event_counts)
            total_events = event_counts.sum()
            if total_events < 15:
                messages.append(f"{event_type}: Total events={total_events}<15 (Poisson ref skipped)")
                continue

            poisson_mean = np.mean(event_counts)
            poisson_var = np.var(event_counts)
            mean_var_ratio = poisson_var / poisson_mean if poisson_mean > 0 else 0
            messages.append(
                f"{event_type}: Poisson ref (mean={poisson_mean:.2f}, var={poisson_var:.2f}, ratio={mean_var_ratio:.2f})")

        # 2. Uniform Distribution Reference
        for event_type, delta_key, is_intimacy in [
            ('Small Conflict', 'I_s', True), ('Large Conflict', 'I_l', True), ('Positive Event', 'I_p', True),
            ('Small Conflict', 'C_s', False), ('Large Conflict', 'C_l', False), ('Positive Event', 'C_p', False)
        ]:
            impact_amps = []
            for sim in range(num_sim):
                diffs = np.diff(I[sim, :]) if is_intimacy else np.diff(C[sim, :])
                if event_type == 'Small Conflict':
                    if is_intimacy:
                        impacts = -diffs[(diffs < -jump_thresh) & (diffs > -self.Delta[delta_key])]
                    else:
                        impacts = diffs[(diffs > jump_thresh) & (diffs < self.Delta[delta_key])]
                elif event_type == 'Large Conflict':
                    if is_intimacy:
                        impacts = -diffs[diffs <= -self.Delta[delta_key]]
                    else:
                        impacts = diffs[diffs >= self.Delta[delta_key]]
                else:
                    if is_intimacy:
                        impacts = diffs[diffs > jump_thresh]
                    else:
                        impacts = -diffs[diffs < -jump_thresh]
                impact_amps.extend(impacts)

            impact_amps = np.array(impact_amps)
            if len(impact_amps) < 10:
                messages.append(
                    f"{event_type} ({delta_key}): Impact samples={len(impact_amps)}<10 (Uniform ref skipped)")
                continue

            uniform_theo_mean = self.Delta[delta_key] / 2
            impact_mean = np.mean(impact_amps)
            mean_ratio = impact_mean / uniform_theo_mean if uniform_theo_mean > 0 else 0
            messages.append(
                f"{event_type} ({delta_key}): Uniform ref (theo mean={uniform_theo_mean:.3f}, actual mean={impact_mean:.3f}, ratio={mean_ratio:.2f})")

        return messages


# ---------- 6. Sensitivity Analysis (No Distribution Filtering) ---------- #
def sensitivity_analysis(param_name: str, param_values: np.ndarray, T: int = 365, num_sim: int = 150) -> tuple[
    np.ndarray, list[dict]]:
    """Sensitivity analysis: retain all valid-parameter sets"""
    model = MaritalModel()
    valid_params = ['r_I', 'lambda_l', 'lambda_p', 'I_baseline', 'lambda_s']
    if param_name not in valid_params:
        raise ValueError(f"Supported params: {valid_params}, input: {param_name}")

    valid_vals = []
    results = []
    print(
        f"=== Analyzing Paper Parameter {param_name} (Range: {param_values.min():.4f} ~ {param_values.max():.4f}) ===")

    for val in tqdm(param_values):
        try:
            config = {param_name: val}
            sim_model = MaritalModel(config)
            t, I, C = sim_model.simulate(T=T, num_simulations=num_sim)
            stats = sim_model.get_statistics(I, C)
            dist_info = sim_model.analyze_distributions(I, C)

            valid_vals.append(val)
            results.append({
                'param_value': val,
                'stats': stats,
                'I': I,
                'C': C,
                'distribution_info': dist_info
            })

        except ValueError as e:
            warnings.warn(f"Skipping invalid {param_name}={val:.4f}: {str(e)}")
            continue

    return np.array(valid_vals), results


# ---------- 7. Plotting Functions (English Only) ---------- #
def plot_basic_simulation(model: MaritalModel, t: np.ndarray, I: np.ndarray, C: np.ndarray,
                          fig_label: str = "Fig.1") -> plt.Figure:
    """Basic simulation plot (Paper Fig.1 style)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)

    # Intimacy
    num_plot = min(10, I.shape[0])
    for i in range(num_plot):
        ax1.plot(t, I[i, :], color=COLORS['intimacy'], alpha=0.6, linewidth=1.2,
                 label=f'Simulation {i + 1}' if i == 0 else "")
    I_mean = np.mean(I, axis=0)
    ax1.plot(t, I_mean, color=COLORS['highlight'], linewidth=2, label='Mean Intimacy (150 sims)')
    ax1.axhline(y=model.I_base, color=COLORS['baseline'], linestyle='--', linewidth=1.5,
                label=f'Intimacy Baseline $I_{{baseline}}$={model.I_base:.2f}')
    ax1.text(0.02, 0.98, 'Assumptions: Events=Poisson, Impacts=Uniform',
             transform=ax1.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_ylabel('Intimacy $I(t)$ (0=No Bond, 1=Perfect)')
    ax1.set_ylim(0.65, 0.95)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax1.grid(True)

    # Conflict
    for i in range(num_plot):
        ax2.plot(t, C[i, :], color=COLORS['conflict'], alpha=0.6, linewidth=1.2)
    C_mean = np.mean(C, axis=0)
    ax2.plot(t, C_mean, color=COLORS['conflict'], linewidth=2, label='Mean Conflict (150 sims)')

    ax2.set_xlabel('Time (Days)')
    ax2.set_ylabel('Conflict $C(t)$ (0=None, 1=Extreme)')
    ax2.set_ylim(0, 0.4)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax2.grid(True)

    return fig


def plot_sensitivity(param_vals: np.ndarray, results: list[dict], param_name: str,
                     fig_label: str = "Fig.2") -> plt.Figure:
    """Sensitivity plot: improved color differentiation for trajectories"""
    param_labels = {
        'r_I': '$r_I$ (Intimacy Recovery Rate)',
        'lambda_l': '$\lambda_l$ (Large Conflict Rate)',
        'lambda_p': '$\lambda_p$ (Positive Event Rate)',
        'I_baseline': '$I_{baseline}$ (Intimacy Baseline)',
        'lambda_s': '$\lambda_s$ (Small Conflict Rate)'
    }
    param_label = param_labels.get(param_name, param_name)
    default_val = MaritalModel().default_params[param_name]

    if not results:
        raise ValueError("No valid results to plot")

    # Metrics calculation
    I_mean = np.array([np.mean(res['stats']['I_mean']) for res in results])
    I_mean_se = np.array([np.std(res['stats']['I_mean']) / np.sqrt(len(res['stats']['I_mean'])) for res in results])
    I_dev = np.array([np.mean(res['stats']['I_base_dev']) for res in results])
    C_mean = np.array([np.mean(res['stats']['C_mean']) for res in results])

    # Create figure with GridSpec
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # (a) Mean Intimacy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(param_vals, I_mean, yerr=I_mean_se, fmt='o-', color=COLORS['intimacy'],
                 capsize=3, markersize=5, linewidth=1.5)
    ax1.axvline(x=default_val, color=COLORS['baseline'], linestyle=':', linewidth=1.5,
                label=f'Paper Default: {default_val:.4f}')
    ax1.set_xlabel(param_label)
    ax1.set_ylabel('Mean Intimacy $\\mathbb{E}[I(t)]$')
    ax1.set_title('(a) Mean Intimacy', fontweight='bold')
    ax1.legend()
    ax1.grid(True)

    # (b) Baseline Deviation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(param_vals, I_dev, 's-', color=COLORS['highlight'], markersize=5, linewidth=1.5)
    ax2.axvline(x=default_val, color=COLORS['baseline'], linestyle=':')
    ax2.set_xlabel(param_label)
    ax2.set_ylabel('Intimacy Baseline Deviation')
    ax2.set_title('(b) Intimacy Stability', fontweight='bold')
    ax2.grid(True)

    # (c) Mean Conflict
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(param_vals, C_mean, '^-', color=COLORS['conflict'], markersize=5, linewidth=1.5)
    ax3.axvline(x=default_val, color=COLORS['baseline'], linestyle=':')
    ax3.set_xlabel(param_label)
    ax3.set_ylabel('Mean Conflict $\\mathbb{E}[C(t)]$')
    ax3.set_title('(c) Mean Conflict', fontweight='bold')
    ax3.grid(True)

    # (d) Representative Trajectories with improved color differentiation
    ax4 = fig.add_subplot(gs[1, 1])
    key_indices = [0, len(results) // 2, -1] if len(results) >= 3 else range(len(results))
    line_styles = ['-', '--', '-.'][:len(key_indices)]

    # Enhanced color palette with high contrast colors (journal-friendly)
    TRAJECTORY_COLORS = [
        '#1f77b4',  # Blue (original intimacy color)
        '#ff7f0e',  # Orange (original conflict color)
        '#17becf',  # Cyan (original highlight color)
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd'  # Purple
    ]

    for i, (idx, ls) in enumerate(zip(key_indices, line_styles)):
        res = results[idx]
        val = res['param_value']
        I_mean_traj = np.mean(res['I'], axis=0)
        # Cycle through colors using modulo to handle any number of trajectories
        color = TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)]
        ax4.plot(np.arange(len(I_mean_traj)), I_mean_traj,
                 color=color, linestyle=ls, linewidth=1.2,
                 label=f'{param_label} = {val:.4f}')

    ax4.axhline(y=MaritalModel().default_params['I_baseline'], color=COLORS['baseline'], linestyle=':',
                label=f'$I_{{baseline}}$={MaritalModel().default_params["I_baseline"]:.2f}')
    ax4.set_xlabel('Time (Days)')
    ax4.set_ylabel('Intimacy $I(t)$')
    ax4.set_title('(d) Representative Trajectories', fontweight='bold')
    ax4.legend(fontsize=7)
    ax4.grid(True)

    return fig


# ---------- 8. Parameter Correlation (Compact & Elegant Visualization) ---------- #
def parameter_correlation(param_ranges: dict, num_sets: int = 50, T: int = 365) -> tuple[plt.Figure, pd.DataFrame]:
    """Optimized: Heatmap + Key Scatters instead of pairplot for brevity and aesthetics"""
    np.random.seed(42)
    param_sets = []
    # 1. Generate parameter combinations
    for _ in range(num_sets):
        params = {}
        for param, (min_val, max_val) in param_ranges.items():
            params[param] = np.random.uniform(min_val, max_val)
        param_sets.append(params)

    # 2. Simulate and filter valid data
    data = []
    print(f"=== Multi-Parameter Correlation (=50 Sets) ===")
    for params in tqdm(param_sets):
        try:
            model = MaritalModel(params)
            t, I, C = model.simulate(T=T, num_simulations=50)
            stats = model.get_statistics(I, C)
            # Organize data row (parameters + output metrics)
            row = {**params}
            row['I_mean'] = np.mean(stats['I_mean'])    # Mean intimacy
            row['I_dev'] = np.mean(stats['I_base_dev']) # Intimacy baseline deviation
            row['C_mean'] = np.mean(stats['C_mean'])    # Mean conflict
            data.append(row)
        except ValueError as e:
            warnings.warn(f"Skipping invalid set: {str(e)}")
            continue

    if len(data) == 0:
        raise ValueError("No valid parameter sets. Check paper constraints.")
    df = pd.DataFrame(data)

    # 3. Define academic labels for variables
    var_labels = {
        'r_I': '$r_I$',
        'lambda_l': '$\lambda_l$',
        'lambda_p': '$\lambda_p$',
        'I_mean': '$\mathbb{E}[I(t)]$',
        'I_dev': '$\mathbb{E}[|I-I_{baseline}|]$',
        'C_mean': '$\mathbb{E}[C(t)]$'
    }
    # Filter variables for plotting
    plot_vars = list(param_ranges.keys()) + ['I_mean', 'I_dev', 'C_mean']
    df_plot = df[plot_vars].copy()
    # Replace column names with academic labels
    df_plot.columns = [var_labels[col] for col in df_plot.columns]

    # 4. Calculate correlation matrix with significance markers
    corr_matrix = df_plot.corr()
    # Calculate significance (p-values)
    def calculate_correlation_pvalues(df):
        pvals = np.zeros((df.shape[1], df.shape[1]))
        for i in range(df.shape[1]):
            for j in range(df.shape[1]):
                if i != j:
                    _, p = sp_stats.pearsonr(df.iloc[:, i], df.iloc[:, j])
                    pvals[i, j] = p
        return pvals
    p_matrix = calculate_correlation_pvalues(df_plot)

    # 5. Plot combined heatmap + key scatter plots
    # 关键调整1：减小垂直方向间距（hspace从0.3改为0.15），缩小上下图形间隔
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)  # 微调图高，避免拥挤
    gs = GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.3)  # hspace控制上下间隔

    # Subplot 1: Correlation heatmap
    ax_heatmap = fig.add_subplot(gs[0, :])
    im = ax_heatmap.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8, pad=0.02)
    cbar.set_label('Pearson Correlation Coefficient', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Annotate heatmap with correlation values and significance stars
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            p_val = p_matrix[i, j]
            # Significance stars: ***p<0.001, **p<0.01, *p<0.05
            if p_val < 0.001:
                star = '***'
            elif p_val < 0.01:
                star = '**'
            elif p_val < 0.05:
                star = '*'
            else:
                star = ''
            # Text color based on background contrast
            text_color = 'white' if abs(corr_val) > 0.5 else 'black'
            ax_heatmap.text(j, i, f'{corr_val:.2f}{star}',
                           ha='center', va='center', fontsize=8, color=text_color)

    # Heatmap labels
    ax_heatmap.set_xticks(range(len(corr_matrix.columns)))
    ax_heatmap.set_yticks(range(len(corr_matrix.columns)))
    # 关键调整2：横轴标注rotation=0（取消倾斜），ha='center'（水平居中）
    ax_heatmap.set_xticklabels(corr_matrix.columns, rotation=0, ha='center', fontsize=8)
    ax_heatmap.set_yticklabels(corr_matrix.columns, fontsize=8)
    ax_heatmap.set_title('(a) Parameter-Metric Correlation Matrix', fontweight='bold', fontsize=10)
    # Remove unnecessary spines
    ax_heatmap.spines['top'].set_visible(False)
    ax_heatmap.spines['right'].set_visible(False)
    ax_heatmap.spines['bottom'].set_visible(False)
    ax_heatmap.spines['left'].set_visible(False)

    # Subplots 2-3: Key scatter plots for strong correlations
    # Identify strong correlations (|r| > 0.5)
    strong_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                strong_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
    # Fallback pairs if insufficient strong correlations
    if len(strong_corr_pairs) < 2:
        strong_corr_pairs = [
            ('$\lambda_l$', '$\mathbb{E}[C(t)]$'),
            ('$\lambda_p$', '$\mathbb{E}[I(t)]$')
        ]

    # Subplot 2: First strong correlation
    ax_scatter1 = fig.add_subplot(gs[1, 0])
    x1, y1 = strong_corr_pairs[0][0], strong_corr_pairs[0][1]
    ax_scatter1.scatter(df_plot[x1], df_plot[y1],
                       color=COLORS['conflict'], alpha=0.7, s=40, edgecolor='white', linewidth=0.5)
    # Add trend line
    z1 = np.polyfit(df_plot[x1], df_plot[y1], 1)
    p1 = np.poly1d(z1)
    ax_scatter1.plot(df_plot[x1], p1(df_plot[x1]),
                    color=COLORS['baseline'], linestyle='--', linewidth=1.2, alpha=0.8)
    # Annotate correlation coefficient
    corr1 = corr_matrix.loc[x1, y1]
    ax_scatter1.text(0.05, 0.95, f'Corr = {corr1:.2f}', transform=ax_scatter1.transAxes,
                    fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_scatter1.set_xlabel(x1, fontsize=8)
    ax_scatter1.set_ylabel(y1, fontsize=8)
    # 散点图横轴标注同步取消倾斜
    ax_scatter1.tick_params(axis='x', rotation=0, labelsize=7)
    ax_scatter1.set_title(f'(b) {x1} vs {y1}', fontweight='bold', fontsize=9)
    ax_scatter1.grid(True, linestyle='--', alpha=0.5)
    ax_scatter1.tick_params(labelsize=7)

    # Subplot 3: Second strong correlation
    ax_scatter2 = fig.add_subplot(gs[1, 1])
    x2, y2 = strong_corr_pairs[1][0], strong_corr_pairs[1][1]
    ax_scatter2.scatter(df_plot[x2], df_plot[y2],
                       color=COLORS['intimacy'], alpha=0.7, s=40, edgecolor='white', linewidth=0.5)
    # Add trend line
    z2 = np.polyfit(df_plot[x2], df_plot[y2], 1)
    p2 = np.poly1d(z2)
    ax_scatter2.plot(df_plot[x2], p2(df_plot[x2]),
                    color=COLORS['baseline'], linestyle='--', linewidth=1.2, alpha=0.8)
    # Annotate correlation coefficient
    corr2 = corr_matrix.loc[x2, y2]
    ax_scatter2.text(0.05, 0.95, f'Corr = {corr2:.2f}', transform=ax_scatter2.transAxes,
                    fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_scatter2.set_xlabel(x2, fontsize=8)
    ax_scatter2.set_ylabel(y2, fontsize=8)
    # 散点图横轴标注同步取消倾斜
    ax_scatter2.tick_params(axis='x', rotation=0, labelsize=7)
    ax_scatter2.set_title(f'(c) {x2} vs {y2}', fontweight='bold', fontsize=9)
    ax_scatter2.grid(True, linestyle='--', alpha=0.5)
    ax_scatter2.tick_params(labelsize=7)

    # Figure title
    fig.suptitle(f'Fig.4: Parameter-Metric Correlation ({len(df)}/{num_sets} Valid Sets)',
                 y=1.02, fontsize=11, fontweight='bold')
    return fig, df

# ---------- 9. Main Workflow (No Distribution Filtering) ---------- #
def main():
    """Main workflow: retain all valid-parameter sets"""
    # 1. Basic Simulation
    print("=" * 50)
    print("1. Running Basic Simulation (365 Days, 150 Reps)")
    base_model = MaritalModel()
    t, I, C = base_model.simulate(T=365, num_simulations=150)

    # Distribution analysis (reference only)
    dist_info = base_model.analyze_distributions(I, C)
    print("Distribution Reference Info (for transparency):")
    for msg in dist_info:
        print(f"  - {msg}")

    # Save plot
    base_fig = plot_basic_simulation(base_model, t, I, C, fig_label="Fig.1")
    base_fig.savefig("basic_simulation.png")
    print("Basic simulation plot saved: basic_simulation.png")

    # 2. Sensitivity Analysis
    print("\n" + "=" * 50)
    print("2. Running Parameter Sensitivity Analysis")

    # 2.1 r_I (Intimacy Recovery Rate)
    rI_values = np.linspace(0.08, 0.12, 10)
    rI_vals, rI_results = sensitivity_analysis('r_I', rI_values)
    if rI_results:
        rI_fig = plot_sensitivity(rI_vals, rI_results, 'r_I', fig_label="Fig.2")
        rI_fig.savefig("sensitivity_rI.png")
        print(f"r_I sensitivity plot saved (Sets: {len(rI_results)}/10)")
    else:
        print("No valid r_I results")

    # 2.2 lambda_l (Large Conflict Rate)
    lambda_l_values = np.linspace(0.008, 0.012, 10)
    lambda_l_vals, lambda_l_results = sensitivity_analysis('lambda_l', lambda_l_values)
    if lambda_l_results:
        lambda_l_fig = plot_sensitivity(lambda_l_vals, lambda_l_results, 'lambda_l', fig_label="Fig.3")
        lambda_l_fig.savefig("sensitivity_lambda_l.png")
        print(f"lambda_l sensitivity plot saved (Sets: {len(lambda_l_results)}/10)")
    else:
        print("No valid lambda_l results")

    # 3. Correlation Analysis
    print("\n" + "=" * 50)
    print("3. Running Multi-Parameter Correlation")
    param_ranges = {
        'r_I': (0.08, 0.12),
        'lambda_l': (0.008, 0.012),
        'lambda_p': (0.04, 0.08)
    }

    try:
        corr_fig, corr_df = parameter_correlation(param_ranges, num_sets=50)
        corr_fig.savefig("parameter_correlation.png")
        corr_df.to_csv("correlation_data.csv", index=False)
        print(f"Correlation plot/data saved (Valid sets: {len(corr_df)}/50)")
    except ValueError as e:
        print(f"Correlation failed: {str(e)}")

    # Summary
    print("\n" + "=" * 50)
    print("Analysis Completed! Generated Files:")
    print("1. basic_simulation.png - Basic Dynamics")
    print("2. sensitivity_rI.png    - r_I Sensitivity")
    print("3. sensitivity_lambda_l.png - lambda_l Sensitivity")
    print("4. parameter_correlation.png - Correlation Matrix")
    print("5. correlation_data.csv  - Correlation Raw Data")


if __name__ == "__main__":
    main()
