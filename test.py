from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ReadData import read_obs_file, read_satellite_file, build_epoch_satellite_dict
from ComputeDop import geodetic_to_ecef, ecef_to_enu_rotation, compute_dops, plot_dops, plot_satellite_count

def plot_satellite_3D(sat_dict, epochs):
    """satellite 3D track"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    prns = sorted({prn for t in epochs for prn in sat_dict[t].keys()})
    for prn in prns:
        prn_epochs = [t for t in epochs if prn in sat_dict[t]]
        if not prn_epochs:
            continue
        coords = np.array([sat_dict[t][prn]["pos"] for t in prn_epochs])
        if coords.size == 0:
            continue
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], label=f"PRN {prn}")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Satellite tracks")
    ax.legend(fontsize=8)
    plt.show()


def plot_pseudorange(sat_dict, epochs):
    """Pseudorange with time"""
    prns = sorted({prn for t in epochs for prn in sat_dict[t].keys()})
    t0 = epochs[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Pseudorange over Time")
    for prn in prns:
        prn_epochs = [t for t in epochs if prn in sat_dict[t]]
        if not prn_epochs:
            continue
        prn_data = [sat_dict[t][prn] for t in prn_epochs]
        times = [t - t0 for t in prn_epochs]
        ax.plot(times, [d["pseudorange"] for d in prn_data], label=f"PRN {prn}")
    ax.set_xlabel("Time from beginning on test (s)")
    ax.set_ylabel("Pseudorange (m)")
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.show()


def plot_L1phase(sat_dict, epochs):
    """L1 carrier phase with time"""
    prns = sorted({prn for t in epochs for prn in sat_dict[t].keys()})
    t0 = epochs[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("L1 Carrier Phase over Time")
    for prn in prns:
        prn_epochs = [t for t in epochs if prn in sat_dict[t]]
        if not prn_epochs:
            continue
        prn_data = [sat_dict[t][prn] for t in prn_epochs]
        times = [t - t0 for t in prn_epochs]
        ax.plot(times, [d["L1_phase"] for d in prn_data], label=f"PRN {prn}")
    ax.set_xlabel("Time from beginning on test (s)")
    ax.set_ylabel("L1 Carrier Phase (cycles)")
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.show()


def plot_doppler(sat_dict, epochs):
    """L1 doppler with time"""
    prns = sorted({prn for t in epochs for prn in sat_dict[t].keys()})
    t0 = epochs[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Doppler over Time")
    for prn in prns:
        prn_epochs = [t for t in epochs if prn in sat_dict[t]]
        if not prn_epochs:
            continue
        prn_data = [sat_dict[t][prn] for t in prn_epochs]
        times = [t - t0 for t in prn_epochs]
        ax.plot(times, [d["Doppler"] for d in prn_data], label=f"PRN {prn}")
    ax.set_xlabel("Time from beginning on test (s)")
    ax.set_ylabel("Doppler (Hz)")
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.show()


def analyze_variations(sat_dict, epochs):
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
    data_dir = Path(__file__).resolve().parent
    sat_file = data_dir / "Satellites1.sat"
    obs_file = data_dir / "RemoteL1L2.obs"

    if not sat_file.exists():
        raise FileNotFoundError(f"Satellite file not found: {sat_file}")
    if not obs_file.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_file}")

    sat_data = read_satellite_file(sat_file)
    obs_data = read_obs_file(obs_file)
    sat_dict, epochs = build_epoch_satellite_dict(sat_data, obs_data, max_epochs=3600)

    # Task 1
    plot_satellite_3D(sat_dict, epochs)
    plot_pseudorange(sat_dict, epochs)
    plot_L1phase(sat_dict, epochs)
    plot_doppler(sat_dict, epochs)
    analyze_variations(sat_dict, epochs)

    # Task 2
    # Remote receiver position (WGS-84)
    lat = 51 + 15/60 + 31.11582/3600       # 51.258643°
    lon = -(114 + 6/60 + 1.76988/3600)     # −114.100492°
    h = 1127.345
    approx_rcv_xyz = geodetic_to_ecef(lat, lon, h)
    print("Receiver ECEF position (m):", approx_rcv_xyz)

    dop_results = compute_dops(sat_dict, epochs, approx_rcv_xyz, lat, lon)
    plot_dops(dop_results, epochs)
    plot_satellite_count(dop_results, epochs)
