from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ComputeDop import ecef_to_enu_rotation, geodetic_to_ecef
from ReadData import build_epoch_satellite_dict
from plottings import (
    plot_L1phase,
    plot_doppler,
    plot_enu_errors,
    plot_float_ambiguities,
    plot_pseudorange,
    plot_satellite_3D,
)
from plsq import satellite_elevation

L1_WAVELENGTH = 0.190293672798365

# Measurement noise (standard deviation, meters) for double-differenced data
CODE_NOISE_DD = 0.30
PHASE_NOISE_DD = 0.01


def _reference_satellite(
    rover_epoch: Dict[int, Dict[str, float]],
    lat: float,
    lon: float,
    approx_xyz: np.ndarray,
) -> int:
    """Select the highest-elevation satellite in the given epoch."""

    elevs = {
        prn: satellite_elevation(approx_xyz, rover_epoch[prn]["pos"], lat, lon)
        for prn in rover_epoch
    }
    return max(elevs, key=elevs.get)


def _common_satellites(
    rover_dict: Dict[float, Dict[int, Dict[str, float]]],
    base_dict: Dict[float, Dict[int, Dict[str, float]]],
    epochs: List[float],
) -> List[int]:
    """Return satellites observed at both receivers for all provided epochs."""

    common: set[int] = set(rover_dict[epochs[0]].keys()) & set(
        base_dict[epochs[0]].keys()
    )
    for t in epochs[1:]:
        common &= set(rover_dict[t].keys()) & set(base_dict[t].keys())
    return sorted(common)


def _initial_state(
    prns: List[int],
    ref_prn: int,
    rover_epoch: Dict[int, Dict[str, float]],
    base_epoch: Dict[int, Dict[str, float]],
) -> np.ndarray:
    """Initialize baseline to zeros and float ambiguities from rounded DD phase."""

    state = np.zeros(3 + (len(prns) - 1))
    ref_rover = rover_epoch[ref_prn]
    ref_base = base_epoch[ref_prn]

    prn_to_amb = {prn: idx for idx, prn in enumerate([p for p in prns if p != ref_prn])}

    for prn in prns:
        if prn == ref_prn:
            continue
        dd_phase_cycles = (rover_epoch[prn]["L1_phase"] - base_epoch[prn]["L1_phase"]) - (
            ref_rover["L1_phase"] - ref_base["L1_phase"]
        )
        state[3 + prn_to_amb[prn]] = round(dd_phase_cycles)

    return state


def _build_dd_matrices(
    rover_epoch: Dict[int, Dict[str, float]],
    base_epoch: Dict[int, Dict[str, float]],
    prns: List[int],
    ref_prn: int,
    rover_pos: np.ndarray,
    base_pos: np.ndarray,
    state: np.ndarray,
    prn_to_amb: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct design matrix A and residual vector w for one linearization step."""

    rows: List[np.ndarray] = []
    residuals: List[float] = []

    ref_data_rover = rover_epoch[ref_prn]
    ref_data_base = base_epoch[ref_prn]

    vec_ref = ref_data_rover["pos"] - rover_pos
    rho_ref_r = np.linalg.norm(vec_ref)
    los_ref = -vec_ref / rho_ref_r

    rho_ref_b = np.linalg.norm(ref_data_base["pos"] - base_pos)
    geom_ref = rho_ref_r - rho_ref_b

    ambiguities = state[3:]

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
        phase_dd_cycles = (rover_epoch[prn]["L1_phase"] - base_epoch[prn]["L1_phase"]) - (
            ref_data_rover["L1_phase"] - ref_data_base["L1_phase"]
        )

        grad = los_r - los_ref

        # Code double difference: z = rho_dd, residual w = z - Hx
        code_row = np.concatenate([grad, np.zeros(len(ambiguities))])
        code_res = code_dd - geom_dd
        rows.append(code_row)
        residuals.append(code_res)

        # Phase double difference: z = rho_dd + lambda*N
        amb_idx = prn_to_amb[prn]
        phase_row = np.concatenate([grad, np.zeros(len(ambiguities))])
        phase_row[3 + amb_idx] = L1_WAVELENGTH
        phase_pred = geom_dd + L1_WAVELENGTH * ambiguities[amb_idx]
        phase_res = phase_dd_cycles * L1_WAVELENGTH - phase_pred
        rows.append(phase_row)
        residuals.append(phase_res)

    return np.vstack(rows), np.array(residuals)


def _normal_equation_solution(
    A: np.ndarray, w: np.ndarray, R: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the batch weighted least-squares normal equations."""

    R_inv = np.linalg.inv(R)
    N = A.T @ R_inv @ A
    n = A.T @ R_inv @ w
    delta = np.linalg.solve(N, n)
    return delta, np.linalg.inv(N)


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
    R_enu = ecef_to_enu_rotation(rover_lat, rover_lon)

    # Candidate satellites with measurements on both receivers at any time
    available_prns = sorted(
        {prn for t in all_epochs for prn in rover_dict[t] if prn in base_dict[t]}
    )
    if not available_prns:
        return

    # Reference satellite: highest elevation at first epoch with shared data
    first_epoch = all_epochs[0]
    first_common = {
        prn for prn in available_prns if prn in rover_dict[first_epoch] and prn in base_dict[first_epoch]
    }
    ref_prn = _reference_satellite({p: rover_dict[first_epoch][p] for p in first_common}, rover_lat, rover_lon, true_rover_xyz)

    # Double-difference window from first epoch for duration_s seconds
    t_start = all_epochs[0]
    t_end = t_start + duration_s
    dd_epochs = [t for t in all_epochs if t <= t_end]
    if not dd_epochs:
        return

    # Satellites common to all DD epochs
    prns = _common_satellites(rover_dict, base_dict, dd_epochs)
    if len(prns) < 4:
        raise ValueError("Not enough common satellites for double-difference solution")

    # Ensure reference belongs to the common set; otherwise pick the best inside
    if ref_prn not in prns:
        ref_prn = _reference_satellite(
            {p: rover_dict[dd_epochs[0]][p] for p in prns}, rover_lat, rover_lon, true_rover_xyz
        )

    print("Common satellites used for DD (PRNs):", prns)
    print("Reference satellite (highest elevation):", ref_prn)

    prn_to_amb = {prn: idx for idx, prn in enumerate([p for p in prns if p != ref_prn])}

    # Initial float state (baseline meters + ambiguities in cycles)
    init_epoch = dd_epochs[0]
    state = _initial_state(prns, ref_prn, rover_dict[init_epoch], base_dict[init_epoch])

    epochs_out: List[float] = []
    enu_errors: List[np.ndarray] = []
    enu_sigmas: List[np.ndarray] = []
    ambiguities_over_time: List[np.ndarray] = []

    sigma_vector = np.concatenate(
        [np.full(len(prns) - 1, CODE_NOISE_DD), np.full(len(prns) - 1, PHASE_NOISE_DD)]
    )
    R_meas = np.diag(sigma_vector**2)

    for t in dd_epochs:
        rover_epoch = {p: rover_dict[t][p] for p in prns}
        base_epoch = {p: base_dict[t][p] for p in prns}

        rover_xyz = base_xyz + state[:3]
        cov_epoch = np.zeros((len(state), len(state)))

        for _ in range(5):
            A, w = _build_dd_matrices(
                rover_epoch, base_epoch, prns, ref_prn, rover_xyz, base_xyz, state, prn_to_amb
            )
            if A.size == 0:
                break

            delta, cov_epoch = _normal_equation_solution(A, w, R_meas)
            state += delta
            rover_xyz = base_xyz + state[:3]

            if np.linalg.norm(delta[:3]) < 1e-4 and np.linalg.norm(delta[3:]) < 1e-4:
                break

        Cov_xyz = cov_epoch[:3, :3]
        Cov_enu = R_enu @ Cov_xyz @ R_enu.T
        sigma_enu = np.sqrt(np.diag(Cov_enu))

        enu_err = R_enu @ (rover_xyz - true_rover_xyz)
        print(
            f"t={t - t_start:7.1f}s | dX={state[0]:.3f} dY={state[1]:.3f} dZ={state[2]:.3f} "
            f"| ENU=({enu_err[0]:.3f}, {enu_err[1]:.3f}, {enu_err[2]:.3f}) m"
        )
        epochs_out.append(t - t_start)
        enu_errors.append(enu_err)
        enu_sigmas.append(sigma_enu)
        ambiguities_over_time.append(state[3:].copy())

    if epochs_out:
        epochs_arr = np.array(epochs_out)
        enu_arr = np.vstack(enu_errors)
        enu_std = np.vstack(enu_sigmas)
        plot_enu_errors(epochs_arr, enu_arr, enu_std)

        enu_rmse = np.sqrt(np.mean(enu_arr**2, axis=0))
        print(
            f"ENU RMSE (m): East={enu_rmse[0]:.3f}, North={enu_rmse[1]:.3f}, Up={enu_rmse[2]:.3f}"
        )

        ambiguities_arr = np.vstack(ambiguities_over_time)
        float_prns = [p for p in prns if p != ref_prn]
        plot_float_ambiguities(epochs_arr, ambiguities_arr, float_prns)

        dd_sat_dict = {t: rover_dict[t] for t in dd_epochs}
        plot_satellite_3D(dd_sat_dict, dd_epochs)
        plot_pseudorange(dd_sat_dict, dd_epochs)
        plot_L1phase(dd_sat_dict, dd_epochs)
        plot_doppler(dd_sat_dict, dd_epochs)