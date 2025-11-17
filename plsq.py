# ====================================================
# PLSQ operation
# ====================================================

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from ComputeDop import geodetic_to_ecef, ecef_to_enu_rotation
from ReadData import build_epoch_satellite_dict, read_obs_file, read_satellite_file
from plottings import plot_residuals_vs_elev, plot_residuals_by_sat, plot_enu_errors

def _design_matrix_and_misclosure(
    epoch_data: Dict[int, Dict[str, np.ndarray]],
    state: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the linearised least-squares for one epoch."""
    rows = []
    misclosures = []
    pos = state[:3]
    clk = state[3]

    for data in epoch_data.values():
        sat_pos = data["pos"]
        rho_obs = data["pseudorange"]
        vec = sat_pos - pos
        geom_range = np.linalg.norm(vec)
        if geom_range == 0:
            continue

        # Line-of-sight direction from receiver to satellite
        los = -vec / geom_range
        computed = geom_range + clk
        rows.append([los[0], los[1], los[2], 1.0])
        misclosures.append(rho_obs - computed)

    if not rows:
        return np.empty((0, 4)), np.empty((0,))

    return np.array(rows), np.array(misclosures)


def satellite_elevation(rcv_xyz, sat_xyz, lat, lon):
    """Compute elevation angle (deg) of satellite relative to receiver."""
    vec = sat_xyz - rcv_xyz
    rho = np.linalg.norm(vec)
    if rho == 0:
        return 0.0
    los = vec / rho
    R = ecef_to_enu_rotation(lat, lon)
    enu = R @ los
    east, north, up = enu
    el = np.degrees(np.arctan2(up, np.sqrt(east**2 + north**2)))
    return el


def solve_epoch(
    epoch_data: Dict[int, Dict[str, np.ndarray]],
    initial_state: np.ndarray,
    max_iterations: int = 8,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Iteratively solve."""
    state = initial_state.astype(float).copy()

    for _ in range(max_iterations):
        A, w = _design_matrix_and_misclosure(epoch_data, state)
        if A.shape[0] < 4:
            raise ValueError("Not enough satellite observations for a solution")

        delta, *_ = np.linalg.lstsq(A, w, rcond=None)
        state += delta

        if np.linalg.norm(delta[:3]) < tol and abs(delta[3]) < tol:
            break

    # Final residual and covariance
    A, w = _design_matrix_and_misclosure(epoch_data, state)
    residuals = w
    n, u = A.shape[0], A.shape[1]
    sigma0 = np.sqrt((residuals @ residuals) / max(n - u, 1))
    Q = np.linalg.inv(A.T @ A)
    Cov = sigma0**2 * Q
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return state, rms, Cov

def run_positioning(max_epochs: int = 3600) -> None:
    """Solve for receiver state at each epoch and analyze ENU errors."""
    data_dir = Path(__file__).resolve().parent
    sat_file = data_dir / "Satellites1.sat"
    obs_file = data_dir / "RemoteL1L2.obs"

    sat_data = read_satellite_file(sat_file)
    obs_data = read_obs_file(obs_file)
    sat_dict, epochs = build_epoch_satellite_dict(sat_data, obs_data, max_epochs=max_epochs)

    # Ground truth of remote station
    true_lat = 51 + 15 / 60 + 31.11582 / 3600
    true_lon = -(114 + 6 / 60 + 1.76988 / 3600)
    true_h = 1127.345
    true_xyz = geodetic_to_ecef(true_lat, true_lon, true_h)
    R = ecef_to_enu_rotation(true_lat, true_lon)

    state = np.hstack([true_xyz, 0.0])

    print("Epoch (s)    X (m)          Y (m)          Z (m)          Clock bias (m)    RMS (m)")
    print("----------  ------------  ------------  ------------  ----------------  --------")

    epochs_list, enu_err_list, enu_std_list = [], [], []
    residuals_by_sat = {}
    residuals_elev = []

    for epoch in epochs:
        epoch_data = sat_dict.get(epoch, {})
        if len(epoch_data) < 4:
            continue

        try:
            state, rms, Cov = solve_epoch(epoch_data, state)
        except ValueError:
            continue

        # ENU error
        diff_xyz = state[:3] - true_xyz
        enu_err = R @ diff_xyz
        Cov_xyz = Cov[:3, :3]
        Cov_enu = R @ Cov_xyz @ R.T
        sigma_enu = np.sqrt(np.diag(Cov_enu))

        epochs_list.append(epoch)
        enu_err_list.append(enu_err)
        enu_std_list.append(sigma_enu)

        # Residual check
        A, w = _design_matrix_and_misclosure(epoch_data, state)
        for prn, data in epoch_data.items():
            sat_pos = data["pos"]
            rho_obs = data["pseudorange"]
            geom_range = np.linalg.norm(sat_pos - state[:3])
            comp = geom_range + state[3]
            resid = rho_obs - comp
            elev = satellite_elevation(state[:3], sat_pos, true_lat, true_lon)
            residuals_by_sat.setdefault(prn, []).append((epoch, resid))
            residuals_elev.append((elev, resid, prn))

        print(
            f"{epoch:10.1f}  {state[0]:12.3f}  {state[1]:12.3f}  {state[2]:12.3f}  "
            f"{state[3]:16.3f}  {rms:8.3f}"
        )

        if epoch - epochs[0] > 3600:
            break

    # ENU error plotting
    epochs_arr = np.array(epochs_list)
    enu_errors = np.vstack(enu_err_list)
    enu_stds = np.vstack(enu_std_list)
    plot_enu_errors(epochs_arr, enu_errors, enu_stds)

    mean_true_err = np.sqrt(np.mean(enu_errors**2, axis=0))
    mean_est_std = np.mean(enu_stds, axis=0)
    print("\n=== Accuracy Summary ===")
    print(f"True RMS errors (E, N, U): {mean_true_err[0]:.3f}, {mean_true_err[1]:.3f}, {mean_true_err[2]:.3f} m")
    print(f"Estimated 1σ stds  (E, N, U): {mean_est_std[0]:.3f}, {mean_est_std[1]:.3f}, {mean_est_std[2]:.3f} m")
    print("=========================")

    # Residual plotting
    plot_residuals_by_sat(residuals_by_sat)
    plot_residuals_vs_elev(residuals_elev)

