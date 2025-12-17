from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from ComputeDop import ecef_to_enu_rotation, geodetic_to_ecef
from ReadData import build_epoch_satellite_dict
from plottings import (
    plot_L1phase,
    plot_cycle_slip_metrics,
    plot_doppler,
    plot_enu_errors,
    plot_pseudorange,
    plot_satellite_3D,
)
from plsq import satellite_elevation

L1_WAVELENGTH = 0.190293672798365
def _detect_cycle_slips_phase_rate_dd(
    rover_dict: Dict[float, Dict[int, Dict[str, float]]],
    base_dict: Dict[float, Dict[int, Dict[str, float]]],
    epochs: List[float],
    prns: List[int],
    ref_prn: int,
    threshold_cycles: float = 0.5,
) -> Dict[int, List[Tuple[float, float, bool]]]:

    slips: Dict[int, List[Tuple[float, float, bool]]] = {}
    if not epochs:
        return slips

    t0 = epochs[0]
    for prn in prns:
        if prn == ref_prn:
            continue

        prn_epochs = [
            t
            for t in epochs
            if prn in rover_dict[t]
            and prn in base_dict[t]
            and ref_prn in rover_dict[t]
            and ref_prn in base_dict[t]
        ]
        if len(prn_epochs) < 2:
            continue

        for i in range(1, len(prn_epochs)):
            t_prev, t_cur = prn_epochs[i - 1], prn_epochs[i]
            dt = t_cur - t_prev
            if dt <= 0:
                continue

            rover_prev = rover_dict[t_prev]
            base_prev = base_dict[t_prev]
            rover_cur = rover_dict[t_cur]
            base_cur = base_dict[t_cur]

            dd_phase_prev = (rover_prev[prn]["L1_phase"] - base_prev[prn]["L1_phase"]) - (
                rover_prev[ref_prn]["L1_phase"] - base_prev[ref_prn]["L1_phase"]
            )
            dd_phase_cur = (rover_cur[prn]["L1_phase"] - base_cur[prn]["L1_phase"]) - (
                rover_cur[ref_prn]["L1_phase"] - base_cur[ref_prn]["L1_phase"]
            )

            dd_dopp_prev = (rover_prev[prn]["Doppler"] - base_prev[prn]["Doppler"]) - (
                rover_prev[ref_prn]["Doppler"] - base_prev[ref_prn]["Doppler"]
            )

            predicted_delta = -dd_dopp_prev * dt
            actual_delta = dd_phase_cur - dd_phase_prev
            metric = actual_delta - predicted_delta
            flagged = abs(metric) > threshold_cycles
            slips.setdefault(prn, []).append((t_cur - t0, metric, flagged))

    plot_cycle_slip_metrics(slips, threshold_cycles)
    return slips


def _double_difference_design(
    rover_epoch: Dict[int, Dict[str, float]],
    base_epoch: Dict[int, Dict[str, float]],
    prns: List[int],
    ref_prn: int,
    rover_pos: np.ndarray,
    base_pos: np.ndarray,
    amb_state: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    misclosures = []
    ref_data_rover = rover_epoch[ref_prn]
    ref_data_base = base_epoch[ref_prn]

    vec_ref = ref_data_rover["pos"] - rover_pos
    rho_ref_r = np.linalg.norm(vec_ref)
    los_ref = -vec_ref / rho_ref_r

    rho_ref_b = np.linalg.norm(ref_data_base["pos"] - base_pos)
    geom_ref = rho_ref_r - rho_ref_b

    for prn in prns:
        if prn == ref_prn:
            continue

        sat_r = rover_epoch[prn]["pos"]
        sat_b = base_epoch[prn]["pos"]

        vec_r = sat_r - rover_pos
        rho_r = np.linalg.norm(vec_r)
        los_r = -vec_r / rho_r

        rho_b = np.linalg.norm(sat_b - base_pos)

        geom = rho_r - rho_b
        geom_dd = geom - geom_ref

        code_dd = (rover_epoch[prn]["pseudorange"] - base_epoch[prn]["pseudorange"]) - (
            ref_data_rover["pseudorange"] - ref_data_base["pseudorange"]
        )
        phase_dd = (rover_epoch[prn]["L1_phase"] - base_epoch[prn]["L1_phase"]) - (
            ref_data_rover["L1_phase"] - ref_data_base["L1_phase"]
        )

        grad = los_r - los_ref

        # Code observation (meters)
        rows.append(np.concatenate([grad, np.zeros(len(prns) - 1)]))
        misclosures.append(code_dd - geom_dd)

        # Phase observation (meters), ambiguity per satellite in meters
        amb_row = np.concatenate([grad, np.zeros(len(prns) - 1)])
        amb_idx = [p for p in prns if p != ref_prn].index(prn)
        amb_row[3 + amb_idx] = 1.0
        current_amb = 0.0 if amb_state is None else amb_state[amb_idx]
        rows.append(amb_row)
        misclosures.append(phase_dd * L1_WAVELENGTH - geom_dd - current_amb)

    return np.vstack(rows), np.array(misclosures)


def run_DD_positioning(
    sat_data,
    rover_obs,
    base_obs,
    duration_s: int = 1800,
) -> None:
    rover_dict, rover_epochs = build_epoch_satellite_dict(sat_data, rover_obs, max_epochs=3600)
    base_dict, base_epochs = build_epoch_satellite_dict(sat_data, base_obs, max_epochs=3600)

    if len(rover_epochs) == 0:
        return

    all_epochs = [t for t in rover_epochs if t in set(base_epochs)]
    if not all_epochs:
        return

    # Receiver coordinates
    rover_lat = 51 + 15 / 60 + 31.11582 / 3600
    rover_lon = -(114 + 6 / 60 + 1.76988 / 3600)
    rover_h = 1127.345
    base_lat = 51 + 16 / 60 + 37.34162 / 3600
    base_lon = -(113 + 58 / 60 + 59.51154 / 3600)
    base_h = 1090.833

    true_rover_xyz = geodetic_to_ecef(rover_lat, rover_lon, rover_h)
    base_xyz = geodetic_to_ecef(base_lat, base_lon, base_h)
    R = ecef_to_enu_rotation(rover_lat, rover_lon)

    # Candidate satellites with measurements on both receivers at any time
    available_prns = sorted({prn for t in all_epochs for prn in rover_dict[t] if prn in base_dict[t]})
    if not available_prns:
        return

    # Reference satellite: highest elevation at first epoch with shared data
    first_epoch = all_epochs[0]
    first_prns = [prn for prn in available_prns if prn in rover_dict[first_epoch] and prn in base_dict[first_epoch]]
    elevs = {
        prn: satellite_elevation(true_rover_xyz, rover_dict[first_epoch][prn]["pos"], rover_lat, rover_lon)
        for prn in first_prns
    }
    ref_prn = max(elevs, key=elevs.get)

    # Cycle-slip detection over full dataset
    _detect_cycle_slips_phase_rate_dd(rover_dict, base_dict, all_epochs, available_prns, ref_prn)

    # Double-difference window from first epoch for duration_s seconds
    t_start = all_epochs[0]
    t_end = t_start + duration_s
    dd_epochs = [t for t in all_epochs if t <= t_end]
    if not dd_epochs:
        return

    # Satellites common to all DD epochs
    common_prns = set(rover_dict[dd_epochs[0]].keys()) & set(base_dict[dd_epochs[0]].keys())
    for t in dd_epochs[1:]:
        common_prns &= set(rover_dict[t].keys()) & set(base_dict[t].keys())
    prns = sorted(common_prns)
    if len(prns) < 4:
        raise ValueError("Not enough common satellites for double-difference solution")

    # Ensure the reference is in the common set; if not, pick the highest within it
    if ref_prn not in prns:
        elevs_dd = {
            prn: satellite_elevation(true_rover_xyz, rover_dict[dd_epochs[0]][prn]["pos"], rover_lat, rover_lon)
            for prn in prns
        }
        ref_prn = max(elevs_dd, key=elevs_dd.get)

    print("Common satellites used for DD (PRNs):", prns)
    print("Reference satellite (highest elevation):", ref_prn)

    # Treat the position state as a correction to an a priori baseline to avoid
    # large fixed biases in both the linearization point and error reporting.
    baseline_apriori = true_rover_xyz - base_xyz
    state = np.zeros(3 + (len(prns) - 1))
    N = np.zeros((len(state), len(state)))
    b = np.zeros(len(state))

    epochs_out: List[float] = []
    enu_errors: List[np.ndarray] = []
    enu_sigmas: List[np.ndarray] = []

    for t in dd_epochs:
        rover_epoch = {p: rover_dict[t][p] for p in prns}
        base_epoch = {p: base_dict[t][p] for p in prns}

        rover_xyz = base_xyz + baseline_apriori + state[:3]
        A, w = _double_difference_design(
            rover_epoch, base_epoch, prns, ref_prn, rover_xyz, base_xyz, amb_state=state[3:]
        )
        if A.size == 0:
            continue

        N += A.T @ A
        b += A.T @ w

        if np.linalg.matrix_rank(N) < len(state):
            continue

        state = np.linalg.solve(N, b)
        Cov = np.linalg.inv(N)
        residuals = A @ state - w

        residual_rms = float(np.sqrt(np.mean(residuals**2)))

        Cov_xyz = Cov[:3, :3]
        Cov_enu = R @ Cov_xyz @ R.T
        sigma_enu = np.sqrt(np.diag(Cov_enu))

        rover_xyz = base_xyz + baseline_apriori + state[:3]
        enu_err = R @ state[:3]
        print(
            f"t={t - t_start:7.1f}s | dX={state[0]:.3f} dY={state[1]:.3f} dZ={state[2]:.3f} "
            f"| ENU=({enu_err[0]:.3f}, {enu_err[1]:.3f}, {enu_err[2]:.3f}) m | RMS={residual_rms:.3f}"
        )
        epochs_out.append(t - t_start)
        enu_errors.append(enu_err)
        enu_sigmas.append(sigma_enu)

    if epochs_out:
        epochs_arr = np.array(epochs_out)
        enu_arr = np.vstack(enu_errors)
        enu_std = np.vstack(enu_sigmas)
        plot_enu_errors(epochs_arr, enu_arr, enu_std)

        enu_rmse = np.sqrt(np.mean(enu_arr**2, axis=0))
        print(
            f"ENU RMSE (m): East={enu_rmse[0]:.3f}, North={enu_rmse[1]:.3f}, Up={enu_rmse[2]:.3f}"
        )

        dd_sat_dict = {t: rover_dict[t] for t in dd_epochs}
        plot_satellite_3D(dd_sat_dict, dd_epochs)
        plot_pseudorange(dd_sat_dict, dd_epochs)
        plot_L1phase(dd_sat_dict, dd_epochs)
        plot_doppler(dd_sat_dict, dd_epochs)