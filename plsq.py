# ====================================================
# PLSQ operation
# ====================================================

from typing import Dict, Tuple
import numpy as np
from ComputeDop import geodetic_to_ecef, ecef_to_enu_rotation
from ReadData import build_epoch_satellite_dict
from plottings import plot_residuals_vs_elev, plot_residuals_by_sat, plot_enu_errors

def _design_matrix_and_misclosure(
    epoch_data: Dict[int, Dict[str, np.ndarray]],
    state: np.ndarray,
    base_epoch_data: Dict[int, Dict[str, np.ndarray]] | None = None,
    base_pos: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the linearised least-squares for one epoch."""
    rows = []
    misclosures = []
    pos = state[:3]
    clk = state[3]

    for prn, data in epoch_data.items():
        sat_pos = data["pos"]
        vec = sat_pos - pos
        geom_range = np.linalg.norm(vec)
        if geom_range == 0:
            continue

        los = -vec / geom_range

        if base_epoch_data is None or base_pos is None:
            rho_obs = data["pseudorange"]
            computed = geom_range + clk
            misclosure = rho_obs - computed
        else:
            base_data = base_epoch_data.get(prn)
            if base_data is None:
                continue
            base_vec = base_data["pos"] - base_pos
            base_geom_range = np.linalg.norm(base_vec)
            if base_geom_range == 0:
                continue
            diff_obs = data["pseudorange"] - base_data["pseudorange"]
            diff_geom = geom_range - base_geom_range
            misclosure = diff_obs - (diff_geom + clk)

        rows.append([los[0], los[1], los[2], 1.0])
        misclosures.append(misclosure)

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
    base_epoch_data: Dict[int, Dict[str, np.ndarray]] | None = None,
    base_pos: np.ndarray | None = None,
    max_iterations: int = 8,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Iteratively solve."""
    state = initial_state.astype(float).copy()

    for _ in range(max_iterations):
        A, w = _design_matrix_and_misclosure(epoch_data, state, base_epoch_data, base_pos)
        if A.shape[0] < 4:
            raise ValueError("Not enough satellite observations for a solution")

        delta, *_ = np.linalg.lstsq(A, w, rcond=None)
        state += delta

        if np.linalg.norm(delta[:3]) < tol and abs(delta[3]) < tol:
            break

    # Final residual and covariance␊
    A, w = _design_matrix_and_misclosure(epoch_data, state, base_epoch_data, base_pos)
    residuals = w
    n, u = A.shape[0], A.shape[1]
    sigma0 = np.sqrt((residuals @ residuals) / max(n - u, 1))
    Q = np.linalg.inv(A.T @ A)
    Cov = sigma0**2 * Q
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return state, rms, Cov

def run_positioning(
    sat_data,
    rover_obs,
    base_obs,
    max_epochs: int = 3600,
) -> None:
    """Solve for receiver state at each epoch and analyze ENU errors."""
    rover_dict, rover_epochs = build_epoch_satellite_dict(sat_data, rover_obs, max_epochs=max_epochs)
    base_dict, base_epochs = build_epoch_satellite_dict(sat_data, base_obs, max_epochs=max_epochs)

    # Ground truth of remote station
    true_lat = 51 + 15 / 60 + 31.11582 / 3600
    true_lon = -(114 + 6 / 60 + 1.76988 / 3600)
    true_h = 1127.345
    true_xyz = geodetic_to_ecef(true_lat, true_lon, true_h)
    R = ecef_to_enu_rotation(true_lat, true_lon)

    # Known reference station coordinates
    base_lat = 51 + 16 / 60 + 37.34162 / 3600
    base_lon = -(113 + 58 / 60 + 59.51154 / 3600)
    base_h = 1090.833
    base_xyz = geodetic_to_ecef(base_lat, base_lon, base_h)

    state = np.array([0.0, 0.0, 0.0, 0.0])

    print("Epoch (s)    X (m)          Y (m)          Z (m)          Clock bias (m)    RMS (m)")
    print("----------  ------------  ------------  ------------  ----------------  --------")

    epochs_list, enu_err_list, enu_std_list = [], [], []
    residuals_by_sat = {}
    residuals_elev = []

    common_epochs = [t for t in rover_epochs if t in set(base_epochs)]

    for epoch in common_epochs:
        epoch_data = rover_dict.get(epoch, {})
        base_epoch_data = base_dict.get(epoch, {})
        common_prns = set(epoch_data) & set(base_epoch_data)
        if len(common_prns) < 4:
            continue

        filtered_epoch_data = {prn: epoch_data[prn] for prn in common_prns}
        filtered_base_data = {prn: base_epoch_data[prn] for prn in common_prns}

        try:
            state, rms, Cov = solve_epoch(
                filtered_epoch_data, state, base_epoch_data=filtered_base_data, base_pos=base_xyz
            )
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
        A, w = _design_matrix_and_misclosure(
            filtered_epoch_data, state, base_epoch_data=filtered_base_data, base_pos=base_xyz
        )
        for prn in filtered_epoch_data:
            sat_pos = filtered_epoch_data[prn]["pos"]
            base_pos_sat = filtered_base_data[prn]["pos"]
            diff_obs = filtered_epoch_data[prn]["pseudorange"] - filtered_base_data[prn]["pseudorange"]
            geom_range = np.linalg.norm(sat_pos - state[:3])
            base_geom_range = np.linalg.norm(base_pos_sat - base_xyz)
            diff_geom = geom_range - base_geom_range
            resid = diff_obs - (diff_geom + state[3])
            elev = satellite_elevation(state[:3], sat_pos, true_lat, true_lon)
            residuals_by_sat.setdefault(prn, []).append((epoch, resid))
            residuals_elev.append((elev, resid, prn))

        print(
            f"{epoch:10.1f}  {state[0]:12.3f}  {state[1]:12.3f}  {state[2]:12.3f}  "
            f"{state[3]:16.3f}  {rms:8.3f}"
        )

        if common_epochs and epoch - common_epochs[0] > 3600:
            break

    # ENU error plotting
    if epochs_list:
        epochs_arr = np.array(epochs_list)
        enu_errors = np.vstack(enu_err_list)
        enu_stds = np.vstack(enu_std_list)
        plot_enu_errors(epochs_arr, enu_errors, enu_stds)

        mean_true_err = np.sqrt(np.mean(enu_errors ** 2, axis=0))
        mean_est_std = np.mean(enu_stds, axis=0)
        print("\n=== Accuracy Summary ===")
        print(
            f"True RMS errors (E, N, U): {mean_true_err[0]:.3f}, {mean_true_err[1]:.3f}, {mean_true_err[2]:.3f} m"
        )
        print(
            f"Estimated 1σ stds  (E, N, U): {mean_est_std[0]:.3f}, {mean_est_std[1]:.3f}, {mean_est_std[2]:.3f} m"
        )
        print("=========================")

    # Residual plotting
    plot_residuals_by_sat(residuals_by_sat)
    plot_residuals_vs_elev(residuals_elev)

