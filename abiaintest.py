# -*- coding: utf-8 -*-
"""
终极一次性完美版 —— 消融实验（匹配参考图形布局）
2行3列布局+I/C双轨迹展示，可直接用于顶刊投稿
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
######这个代码是根据审稿人1的意见来进行150次蒙特卡洛消融实验
# 彻底解决数学符号字体警告 + 顶刊格式配置
plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'DejaVu Sans',   # 支持所有数学符号
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.linewidth': 1.2,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': 'black',
    'lines.linewidth': 1.8,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ==================== 参数 ====================
I_baseline = 0.80
r_I, r_C = 0.10, 0.15
lambda_s, lambda_l, lambda_p = 0.09, 0.01, 0.06

Xi_bounds    = [(-0.15, -0.05), (-0.25, -0.15), (0.10, 0.20)]
Gamma_bounds = [(0.08,  0.18),  (0.20,  0.30),  (-0.22, -0.12)]

# ==================== 采样器（Beta(3,5)版）===================
def make_sampler(kind):
    if kind == "Uniform":
        return lambda low, high, rng: rng.uniform(low, high)
    elif kind == "Beta(2,2)":
        return lambda low, high, rng: (high-low) * rng.beta(2, 2) + low
    elif kind == "Beta(3,5)":
        return lambda low, high, rng: (high-low) * rng.beta(3, 5) + low

configs = {
    "Uniform":   {"sampler": make_sampler("Uniform"),   "color": "#0072B2", "label": "Uniform"},
    "Beta(2,2)": {"sampler": make_sampler("Beta(2,2)"), "color": "#D55E00", "label": "Beta(2,2)"},
    "Beta(3,5)": {"sampler": make_sampler("Beta(3,5)"), "color": "#009E73", "label": "Beta(3,5)"},
}

# ==================== 模拟函数（返回I/C双轨迹）===================
def simulate(sampler_func, T=3650, seed=0):
    rng = np.random.default_rng(seed)
    sample = lambda low, high: sampler_func(low, high, rng)

    t = 0.0
    I = I_baseline
    C = 0.0
    I_list = [I]
    C_list = [C]

    # 预生成事件时间
    N_s = rng.poisson(lambda_s * T * 1.6)
    N_l = rng.poisson(lambda_l * T * 1.6)
    N_p = rng.poisson(lambda_p * T * 1.6)
    times_s = np.cumsum(rng.exponential(1/lambda_s, N_s))
    times_l = np.cumsum(rng.exponential(1/lambda_l, N_l))
    times_p = np.cumsum(rng.exponential(1/lambda_p, N_p))

    i_s = i_l = i_p = 0

    while t < T:
        candidates = []
        if i_s < N_s: candidates.append(times_s[i_s])
        if i_l < N_l: candidates.append(times_l[i_l])
        if i_p < N_p: candidates.append(times_p[i_p])
        if not candidates: break
        next_t = min(candidates)
        if next_t > T: break

        dt = next_t - t
        I += r_I * (I_baseline - I) * dt
        C -= r_C * C * dt
        C = max(C, 0.0)

        # 跳跃 + 正确截断
        if i_s < N_s and next_t == times_s[i_s]:
            Xi = -min(-sample(*Xi_bounds[0]), I)
            Gamma = min(sample(*Gamma_bounds[0]), 1 - C)
            i_s += 1
        elif i_l < N_l and next_t == times_l[i_l]:
            Xi = -min(-sample(*Xi_bounds[1]), I)
            Gamma = min(sample(*Gamma_bounds[1]), 1 - C)
            i_l += 1
        else:
            raw_Xi = sample(*Xi_bounds[2])
            Xi = min(raw_Xi, 1.0 - I)
            raw_Gamma = sample(*Gamma_bounds[2])
            Gamma = -min(-raw_Gamma, C)
            i_p += 1

        I = np.clip(I + Xi, 0, 1)
        C = np.clip(C + Gamma, 0, 1)
        t = next_t
        I_list.append(I)
        C_list.append(C)

    return np.array(I_list), np.array(C_list)

# ==================== 150×3 蒙特卡洛 ====================
n_runs = 150
rep_idx = 66  # 选第66次作为代表性轨迹
results = {}

for name, cfg in configs.items():
    print(f"Running {name} ...")
    results[name] = []
    for run in range(n_runs):
        seed = run + n_runs * ["Uniform", "Beta(2,2)", "Beta(3,5)"].index(name)
        I_traj, C_traj = simulate(cfg["sampler"], seed=seed)
        results[name].append({
            "mean_I": I_traj.mean(),
            "mean_C": C_traj.mean(),
            "dev_I":  np.mean(np.abs(I_traj - I_baseline)),
            "I_traj": I_traj,
            "C_traj": C_traj
        })

# ==================== 统计输出 ====================
print("\n=== 消融实验结果 ===")
stats = []
for name in configs:
    mean_Is = [r["mean_I"] for r in results[name]]
    mean_Cs = [r["mean_C"] for r in results[name]]
    devs    = [r["dev_I"]  for r in results[name]]
    print(f"{name:9}: ⟨I⟩ = {np.mean(mean_Is):.4f} ± {np.std(mean_Is):.4f}   "
          f"⟨C⟩ = {np.mean(mean_Cs):.4f}   dev = {np.mean(devs):.4f}")
    stats.append([name,
                  f"{np.mean(mean_Is):.4f}±{np.std(mean_Is):.4f}",
                  f"{np.mean(mean_Cs):.4f}±{np.std(mean_Cs):.4f}",
                  f"{np.mean(devs):.4f}±{np.std(devs):.4f}"])

# ==================== 画图（匹配参考布局：2行3列）===================
fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.35)

# 子图1：柱状图（mean_I, mean_C, dev_I）
ax1 = fig.add_subplot(gs[0, :2])
x = np.arange(3)
width = 0.25
for i, (name, cfg) in enumerate(configs.items()):
    means = [np.mean([r["mean_I"] for r in results[name]]),
             np.mean([r["mean_C"] for r in results[name]]),
             np.mean([r["dev_I"]  for r in results[name]])]
    stds  = [np.std([r["mean_I"] for r in results[name]]),
             np.std([r["mean_C"] for r in results[name]]),
             np.std([r["dev_I"]  for r in results[name]])]
    ax1.bar(x + i*width, means, width, yerr=stds, label=cfg["label"],
            color=cfg["color"], edgecolor='black', capsize=4)

ax1.set_xticks(x + width)
ax1.set_xticklabels(["Mean Intimacy ⟨I⟩", "Mean Conflict ⟨C⟩", "Deviation |I−I₀|"])
ax1.set_ylabel("Value")
ax1.set_title("Long-term Statistics")
ax1.legend()
ax1.grid(axis='y', alpha=0.4)

# 子图2：代表性轨迹（插值平滑）
ax2 = fig.add_subplot(gs[0, 2])
t_year = np.linspace(0, 10, 400)
for name, cfg in configs.items():
    I_full = results[name][rep_idx]["I_traj"]
    t_full = np.linspace(0, 10, len(I_full))
    I_interp = np.interp(t_year, t_full, I_full)
    ax2.plot(t_year, I_interp, color=cfg["color"], label=cfg["label"])
ax2.axhline(I_baseline, color='gray', ls='--', lw=1.5)
ax2.set_xlabel("Time (years)")
ax2.set_ylabel("Intimacy I(t)")
ax2.set_title("Representative Trajectories")
ax2.legend()
ax2.grid(alpha=0.4)

# 子图3-5：各分布的I/C双轨迹
for i, (name, cfg) in enumerate(configs.items()):
    ax = fig.add_subplot(gs[1, i])
    I_full = results[name][rep_idx]["I_traj"]
    C_full = results[name][rep_idx]["C_traj"]
    t_full = np.linspace(0, 10, len(I_full))
    ax.plot(t_full, I_full, color=cfg["color"], lw=1.8, label="I(t)")
    ax.plot(t_full, C_full, color=cfg["color"], alpha=0.7, lw=2.5, label="C(t)")
    ax.axhline(I_baseline, color='gray', ls='--')
    ax.set_title(cfg["label"])
    ax.set_xlabel("Time (years)")
    if i == 0: ax.set_ylabel("State value")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.4)

# 全局标题
fig.suptitle("Ablation Study: Jump Magnitude Distribution", fontsize=17, y=0.97)

# 保存结果
plt.savefig("ablation_final_layout.png", dpi=300, facecolor='white')
plt.savefig("ablation_final_layout.pdf", facecolor='white')
plt.close()

# 保存统计表格
pd.DataFrame(stats, columns=["Distribution", "Mean Intimacy ⟨I⟩", "Mean Conflict ⟨C⟩", "Deviation |I−I₀|"]).to_csv("ablation_final_layout.csv", index=False)

print("\n全部完成！已生成：")
print("   ablation_final_layout.png  (300 dpi)")
print("   ablation_final_layout.pdf  (矢量图)")
print("   ablation_final_layout.csv  (统计表格)")