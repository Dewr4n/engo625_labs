# ====================================================
# All the plotting functions used in the project so far
# ====================================================

import numpy as np
import matplotlib.pyplot as plt

def plot_dops(dop_results, epochs):
    t0 = epochs[0]
    times = dop_results[:, 0] - t0
    HDOP = dop_results[:, 1]
    VDOP = dop_results[:, 2]
    plt.figure(figsize=(8, 5))
    plt.plot(times, HDOP, label="HDOP")
    plt.plot(times, VDOP, label="VDOP")
    plt.xlabel("Time from beginning on test (s)")
    plt.ylabel("DOP Value")
    plt.title("HDOP and VDOP over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_satellite_count(dop_results, epochs):
    t0 = epochs[0]
    times = dop_results[:, 0] - t0
    num_sat = dop_results[:, 3]
    plt.figure(figsize=(8, 4))
    plt.plot(times, num_sat, color="purple")
    plt.xlabel("Time from beginning on test (s)")
    plt.ylabel("Number of Satellites")
    plt.title("Number of Satellites over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
    plt.grid(True)
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
    plt.grid(True)
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
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_enu_errors(epochs, enu_errors, enu_stds):
    """Plot ENU errors and ±std envelopes."""
    labels = ["East error", "North error", "Up error"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(epochs, enu_errors[:, i], label=f"{labels[i]} (m)")
        ax.plot(epochs, enu_stds[:, i], "r--", label="+1σ (estimated)")
        ax.plot(epochs, -enu_stds[:, i], "r--", label="−1σ (estimated)")
        ax.set_ylabel(f"{labels[i]} (m)")
        ax.grid(True)
        ax.legend()
    axes[-1].set_xlabel("Time from beginning on test (s)")
    plt.tight_layout()
    plt.show()


def plot_residuals_by_sat(residuals_dict):
    """plot residuals for each satellite vs time."""
    plt.figure(figsize=(8, 5))
    for prn, vals in residuals_dict.items():
        times = np.array([v[0] for v in vals])
        res = np.array([v[1] for v in vals])
        plt.scatter(times - times[0], res, s=10, alpha=0.6, label=f"PRN {prn}")
    plt.xlabel("Time from beginning on test (s)")
    plt.ylabel("Pseudorange residual (m)")
    plt.title("Pseudorange Residuals over Time for Each Satellite")
    plt.legend(fontsize=6)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_residuals_vs_elev(residuals_elev):
    """plot residuals vs elevation."""
    plt.figure(figsize=(7, 5))

    # residuals_elev: list of (elev, resid, prn)
    prn_groups = {}
    for elev, resid, prn in residuals_elev:
        prn_groups.setdefault(prn, []).append((elev, resid))

    for prn, vals in prn_groups.items():
        vals = np.array(vals)
        plt.scatter(vals[:, 0], vals[:, 1], s=12, alpha=0.7, label=f"PRN {prn}")

    plt.xlabel("Elevation angle (deg)")
    plt.ylabel("Pseudorange residual (m)")
    plt.title("Pseudorange Residuals vs Elevation Angle for Each Satellite")
    plt.legend(fontsize=6)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cycle_slip_metrics(slips, threshold):
    """Each PRN one color"""
    plt.figure(figsize=(9, 5))
    for prn, vals in slips.items():
        if not vals:
            continue
        t = [x[0] for x in vals]
        m = [x[1] for x in vals]
        plt.scatter(
            t, m,
            s=14,
            alpha=0.8,
            label=f"PRN {prn}"
        )

    # Threshold lines
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1)
    plt.axhline(-threshold, color="red", linestyle="--", linewidth=1)

    plt.xlabel("Time from beginning on test (s)")
    plt.ylabel("Phase-rate residual (cycles)")
    plt.title("Cycle Slip Detection (Double-Difference Phase-Rate)")
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_float_ambiguities(times, ambiguities, prns):
    """Plot float double-difference ambiguities versus time.

    Parameters
    ----------
    times : array-like
        Array of epoch times (seconds from start of test).
    ambiguities : np.ndarray
        2D array with shape (n_epochs, n_ambiguities) containing float ambiguity
        estimates (cycles) for each epoch.
    prns : list[int]
        List of PRNs corresponding to the ambiguity columns.
    """

    plt.figure(figsize=(8, 5))
    for idx, prn in enumerate(prns):
        plt.plot(times, ambiguities[:, idx], label=f"PRN {prn} (float)")

    plt.xlabel("Time from beginning on test (s)")
    plt.ylabel("Float ambiguity (cycles)")
    plt.title("Double-Difference Float Ambiguities over Time")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()