import numpy as np
import matplotlib.pyplot as plt


# ====================================================
# 1. 读取函数
# ====================================================

def read_satellite_file(filename):
    """读取卫星坐标文件 (.sat)"""
    record_len = 8  # [PRN, time, X, Y, Z, Xdot, Ydot, Zdot]
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float64)
    n_records = data.size // record_len
    sat = data.reshape((n_records, record_len))
    return sat[sat[:, 0] != 0]  # 删除空通道


def read_obs_file(filename):
    """读取流动站观测文件 (.obs)"""
    record_len = 6  # [PRN, time, pseudorange, L1phase, Doppler, L2phase]
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float64)
    n_records = data.size // record_len
    obs = data.reshape((n_records, record_len))
    return obs[obs[:, 0] != 0]


# ====================================================
# 2. 构建时间 × PRN 的数据结构
# ====================================================

def build_epoch_satellite_dict(sat_data, obs_data, max_epochs=300):
    """把时间作为一维，PRN作为另一维，存储每个卫星对应的参数"""
    epochs = np.unique(sat_data[:, 1])
    epochs = epochs[:max_epochs]  # 限制前300个epoch

    sat_dict = {}
    for t in epochs:
        sat_dict[t] = {}
        sats_this_epoch = sat_data[sat_data[:, 1] == t]
        obs_this_epoch = obs_data[obs_data[:, 1] == t]

        for prn in np.unique(sats_this_epoch[:, 0]):
            sat_entry = sats_this_epoch[sats_this_epoch[:, 0] == prn][0]
            obs_entry = obs_this_epoch[obs_this_epoch[:, 0] == prn]
            # 可能有的PRN没有观测
            if obs_entry.size > 0:
                obs_entry = obs_entry[0]
                sat_dict[t][int(prn)] = {
                    "pos": sat_entry[2:5],
                    "vel": sat_entry[5:8],
                    "pseudorange": obs_entry[2],
                    "L1_phase": obs_entry[3],
                    "Doppler": obs_entry[4],
                    "L2_phase": obs_entry[5],
                }
    return sat_dict, epochs


# ====================================================
# 3. 可视化与分析任务
# ====================================================

def plot_satellite_3D(sat_dict, epochs):
    """Task 1a - 3D 卫星轨迹"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 汇总各PRN轨迹
    prns = sorted({prn for t in epochs for prn in sat_dict[t].keys()})
    for prn in prns:
        coords = np.array([sat_dict[t][prn]["pos"] for t in epochs if prn in sat_dict[t]])
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], label=f"PRN {prn}")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Satellite positions (first 300 epochs)")
    ax.legend(fontsize=8)
    plt.show()


def plot_observations(sat_dict, epochs):
    """Task 1c - 各PRN的伪距、相位、Doppler随时间变化"""
    prns = sorted({prn for t in epochs for prn in sat_dict[t].keys()})
    t0 = epochs[0]
    time_rel = epochs - t0  # 相对时间，方便绘图

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].set_title("Pseudorange vs Time")
    axes[1].set_title("L1 Phase vs Time")
    axes[2].set_title("Doppler vs Time")

    for prn in prns:
        prn_data = [sat_dict[t][prn] for t in epochs if prn in sat_dict[t]]
        times = [t - t0 for t in epochs if prn in sat_dict[t]]
        axes[0].plot(times, [d["pseudorange"] for d in prn_data], label=f"PRN {prn}")
        axes[1].plot(times, [d["L1_phase"] for d in prn_data], label=f"PRN {prn}")
        axes[2].plot(times, [d["Doppler"] for d in prn_data], label=f"PRN {prn}")

    for ax in axes:
        ax.set_ylabel("Value")
        ax.legend(fontsize=6)
    axes[2].set_xlabel("Time from start (s)")
    plt.tight_layout()
    plt.show()


def analyze_variations(sat_dict, epochs):
    """Task 1d - 简单分析伪距与相位变化"""
    prns = sorted({prn for t in epochs for prn in sat_dict[t].keys()})
    print("----- Discussion Summary -----")
    for prn in prns:
        values = np.array([[t, sat_dict[t][prn]["pseudorange"], sat_dict[t][prn]["L1_phase"]]
                           for t in epochs if prn in sat_dict[t]])
        if len(values) > 2:
            delta_pr = np.diff(values[:, 1])
            delta_ph = np.diff(values[:, 2])
            print(f"PRN {prn}: pseudorange Δmean={np.mean(np.abs(delta_pr)):.3f} m, "
                  f"L1 phase Δmean={np.mean(np.abs(delta_ph)):.3f} cycles")
    print("Check whether the pseudorange and phase change rates match expectations based on Doppler.")


# ====================================================
# 4. 主函数
# ====================================================

if __name__ == "__main__":
    sat_file = "Satellites.sat"
    obs_file = "RemoteL1L2.obs"

    sat_data = read_satellite_file(sat_file)
    obs_data = read_obs_file(obs_file)

    sat_dict, epochs = build_epoch_satellite_dict(sat_data, obs_data, max_epochs=600)

    # 调用 Task 1 所有部分
    plot_satellite_3D(sat_dict, epochs)
    plot_observations(sat_dict, epochs)
    analyze_variations(sat_dict, epochs)


