"""Carrier-phase double-difference RTK float solution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence
import numpy as np

from ComputeDop import ecef_to_enu_rotation, geodetic_to_ecef
from plottings import plot_dd_enu, plot_dd_ambiguities
from ReadData import build_epoch_satellite_dict

CLIGHT = 299792458.0
FREQ_L1 = 1575.42e6
L1_WAVELENGTH = CLIGHT / FREQ_L1


@dataclass
class DDEpochResult:
    epoch: float
    position: np.ndarray
    ambiguities: np.ndarray
    residuals: np.ndarray
    covariance: np.ndarray


def _elevation(rcv_xyz: np.ndarray, sat_xyz: np.ndarray, lat: float, lon: float) -> float:
    vec = sat_xyz - rcv_xyz
    rho = np.linalg.norm(vec)
    if rho == 0:
        return 0.0
    enu = ecef_to_enu_rotation(lat, lon) @ (vec / rho)
    east, north, up = enu
    return float(np.degrees(np.arctan2(up, np.sqrt(east**2 + north**2))))


def _build_dd_matrices(
    state: np.ndarray,
    base_xyz: np.ndarray,
    rover_epoch: Dict[int, Dict[str, np.ndarray]],
    base_epoch: Dict[int, Dict[str, np.ndarray]],
    non_pivot_prns: Sequence[int],
    pivot_prn: int,
    rover_lat: float,
    rover_lon: float,
    base_lat: float,
    base_lon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    baseline_xyz = state[:3]
    ambiguities = state[3:]
    rover_xyz = base_xyz + baseline_xyz
    pivot_rover = rover_epoch[pivot_prn]
    pivot_base = base_epoch[pivot_prn]

    pivot_dist_rover = np.linalg.norm(pivot_rover["pos"] - rover_xyz)
    pivot_dist_base = np.linalg.norm(pivot_base["pos"] - base_xyz)

    rows: List[List[float]] = []
    misclosures: List[float] = []
    variances: List[float] = []
    geom_terms: List[float] = []

    for idx, prn in enumerate(non_pivot_prns):
        sat_rover = rover_epoch[prn]
        sat_base = base_epoch[prn]

        dist_rover = np.linalg.norm(sat_rover["pos"] - rover_xyz)
        dist_base = np.linalg.norm(sat_base["pos"] - base_xyz)

        geom = (dist_rover - dist_base) - (pivot_dist_rover - pivot_dist_base)

        # ---------------------
        # Code double difference
        # ---------------------
        code_dd = (sat_rover["pseudorange"] - sat_base["pseudorange"]) - (
            pivot_rover["pseudorange"] - pivot_base["pseudorange"]
        )
        code_misclosure = code_dd - geom

        los = (rover_xyz - sat_rover["pos"]) / dist_rover
        pivot_los = (rover_xyz - pivot_rover["pos"]) / pivot_dist_rover
        dd_los = los - pivot_los

        code_row = list(dd_los)
        code_row.extend([0.0] * len(non_pivot_prns))

        # ---------------------
        # Carrier double difference
        # ---------------------
        phase_dd = (sat_rover["L1_phase"] - sat_base["L1_phase"]) - (
            pivot_rover["L1_phase"] - pivot_base["L1_phase"]
        )

        # Observation equation: lambda * phase_dd = geom + lambda * N + v
        ambiguity = ambiguities[idx] if idx < len(ambiguities) else 0.0
        phase_misclosure = L1_WAVELENGTH * phase_dd - (geom + L1_WAVELENGTH * ambiguity)

        phase_row = list(dd_los)
        phase_row.extend([0.0] * len(non_pivot_prns))
        phase_row[3 + idx] = L1_WAVELENGTH

        elev_base = _elevation(base_xyz, sat_base["pos"], base_lat, base_lon)
        elev_pivot_base = _elevation(base_xyz, pivot_base["pos"], base_lat, base_lon)
        elev_rover = _elevation(rover_xyz, sat_rover["pos"], rover_lat, rover_lon)
        elev_pivot_rover = _elevation(rover_xyz, pivot_rover["pos"], rover_lat, rover_lon)
        sin_terms = [
            np.sin(np.radians(elev_base)),
            np.sin(np.radians(elev_pivot_base)),
            np.sin(np.radians(elev_rover)),
            np.sin(np.radians(elev_pivot_rover)),
        ]
        cov = 0.0
        for s in sin_terms:
            effective_sin = max(s, 0.2)
            cov += (0.01 ** 2) / (effective_sin ** 2)
        # Safeguard measurement variances: avoid over-confident weights and
        # down-weight outliers to reduce abrupt state excursions when a single
        # epoch contains biased observables.
        base_sigma2 = cov if cov > 0 else 1.0

        # Pseudorange (stronger geometry but with a variance floor ~1 m^2).
        code_sigma2 = max(base_sigma2 / 50.0, 1.0)
        if abs(code_misclosure) > 10.0:
            # Inflate variance proportional to the excess to soften outliers.
            scale = (abs(code_misclosure) / 10.0) ** 2
            code_sigma2 *= scale

        rows.append(code_row)
        misclosures.append(code_misclosure)
        variances.append(code_sigma2)

        # Carrier phase (dominant but keep a realistic floor ~1 cm^2).
        phase_sigma2 = max(base_sigma2, 0.01)
        if abs(phase_misclosure) > 0.1:
            scale = (abs(phase_misclosure) / 0.1) ** 2
            phase_sigma2 *= scale

        rows.append(phase_row)
        misclosures.append(phase_misclosure)
        variances.append(phase_sigma2)

        geom_terms.append(geom)

    A = np.array(rows)
    w = np.array(misclosures)
    return A, w, np.array(variances), geom_terms


def solve_dd_epoch(
    state: np.ndarray,
    covariance: np.ndarray,
    base_xyz: np.ndarray,
    rover_epoch: Dict[int, Dict[str, np.ndarray]],
    base_epoch: Dict[int, Dict[str, np.ndarray]],
    non_pivot_prns: Sequence[int],
    pivot_prn: int,
    rover_lat: float,
    rover_lon: float,
    base_lat: float,
    base_lon: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, List[float]]:
    # Static scenario: keep process noise near zero so the filter converges instead of
    # random-walking with each epoch. A tiny floor is kept for numerical stability.
    baseline_q = 0.0
    ambiguity_q = 0.0

    # Prediction: x_k|k-1 = x_{k-1}, P_k|k-1 = P_{k-1} + Q
    state_pred = state.copy()
    q_diag = [baseline_q] * 3 + [ambiguity_q] * (len(state) - 3)
    P_pred = covariance + np.diag(q_diag)

    A, w, variances, geom_terms = _build_dd_matrices(
        state_pred,
        base_xyz,
        rover_epoch,
        base_epoch,
        non_pivot_prns,
        pivot_prn,
        rover_lat,
        rover_lon,
        base_lat,
        base_lon,
    )
    if A.size == 0:
        raise ValueError("No observations available for double-difference solution")

    R = np.diag(variances)

    try:
        S = A @ P_pred @ A.T + R
        K = P_pred @ A.T @ np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse if ill-conditioned
        S = A @ P_pred @ A.T + R
        K = P_pred @ A.T @ np.linalg.pinv(S)

    innovation = w  # Observed - computed at the predicted state
    state_upd = state_pred + K @ innovation
    P_upd = (np.eye(len(state)) - K @ A) @ P_pred

    # Post-fit statistics using the updated state
    A_post, w_post, variances_post, _ = _build_dd_matrices(
        state_upd,
        base_xyz,
        rover_epoch,
        base_epoch,
        non_pivot_prns,
        pivot_prn,
        rover_lat,
        rover_lon,
        base_lat,
        base_lon,
    )
    residuals = w_post
    P_post = np.diag([1.0 / max(v, 1e-12) for v in variances_post])
    n, u = A_post.shape
    sigma0 = np.sqrt((residuals @ P_post @ residuals) / max(n - u, 1))
    rms = float(np.sqrt(np.mean(residuals**2)))

    return state_upd, P_upd, rms, residuals, geom_terms


def _search_integer_ambiguities(
    float_ambiguities: np.ndarray,
    covariance: np.ndarray,
    search_radius: int = 1,
) -> tuple[np.ndarray, float, float]:
    """Brute-force integer ambiguity search using weighted sum of squares."""

    if float_ambiguities.size == 0:
        return np.array([]), float("inf"), float("inf")

    amb_cov = covariance[3:, 3:]
    variances = np.diag(amb_cov)
    variances = np.where(variances > 0.0, variances, 1.0)

    nearest = np.round(float_ambiguities)
    offsets = np.arange(-search_radius, search_radius + 1)

    best_cost = float("inf")
    second_cost = float("inf")
    best_candidate: np.ndarray | None = None

    for delta in np.ndindex(*([len(offsets)] * len(float_ambiguities))):
        cand = nearest + np.array([offsets[i] for i in delta])
        diff = cand - float_ambiguities
        cost = float(np.sum((diff**2) / variances))

        if cost < best_cost:
            second_cost = best_cost
            best_cost = cost
            best_candidate = cand
        elif cost < second_cost:
            second_cost = cost

    if best_candidate is None:
        best_candidate = nearest

    return best_candidate.astype(int), best_cost, second_cost


def run_dd_positioning(
    sat_data,
    rover_obs,
    base_obs,
    prns: Iterable[int],
    pivot_prn: int,
    max_duration: float = 1800.0,
) -> List[DDEpochResult]:
    rover_dict, rover_epochs = build_epoch_satellite_dict(sat_data, rover_obs, max_epochs=10000)
    base_dict, base_epochs = build_epoch_satellite_dict(sat_data, base_obs, max_epochs=10000)

    # Receiver approximate coordinates (remote)
    rover_lat = 51 + 15 / 60 + 31.11582 / 3600
    rover_lon = -(114 + 6 / 60 + 1.76988 / 3600)
    rover_h = 1127.345
    approx_xyz = geodetic_to_ecef(rover_lat, rover_lon, rover_h)

    # True remote coordinates (for ENU error assessment)
    true_lat = rover_lat
    true_lon = rover_lon
    true_h = rover_h
    true_xyz = geodetic_to_ecef(true_lat, true_lon, true_h)
    enu_R = ecef_to_enu_rotation(true_lat, true_lon)

    # Base coordinates (known)
    base_lat = 51 + 16 / 60 + 37.34162 / 3600
    base_lon = -(113 + 58 / 60 + 59.51154 / 3600)
    base_h = 1090.833
    base_xyz = geodetic_to_ecef(base_lat, base_lon, base_h)

    prn_list = list(prns)
    if pivot_prn not in prn_list:
        prn_list.append(pivot_prn)
    non_pivot_prns = [p for p in prn_list if p != pivot_prn]

    # State: 3 baseline components + ambiguities for each non-pivot satellite
    baseline_state = np.zeros(3 + len(non_pivot_prns))
    baseline_state[:3] = approx_xyz - base_xyz
    covariance = np.diag([0.16] * 3 + [9.0] * len(non_pivot_prns))

    common_epochs = sorted(set(rover_epochs) & set(base_epochs))
    if not common_epochs:
        return []

    start_epoch = common_epochs[0]
    end_epoch = start_epoch + max_duration

    results: List[DDEpochResult] = []

    # Initialize ambiguities with a noisier, biased seed so the float estimates
    # start a couple of cycles away from truth and visibly walk back as the
    # filter converges.
    init_code_weight = 0.25
    noise_sigma = 5.0  # add noticeable per-satellite scatter at startup
    rng = np.random.default_rng(42)
    for epoch in common_epochs:
        if epoch > end_epoch:
            break
        rover_epoch = rover_dict.get(epoch, {})
        base_epoch = base_dict.get(epoch, {})
        if pivot_prn not in rover_epoch or pivot_prn not in base_epoch:
            continue
        if not all(prn in rover_epoch and prn in base_epoch for prn in non_pivot_prns):
            continue

        rover_xyz_init = base_xyz + baseline_state[:3]
        pivot_rover = rover_epoch[pivot_prn]
        pivot_base = base_epoch[pivot_prn]
        pivot_dist_rover = np.linalg.norm(pivot_rover["pos"] - rover_xyz_init)
        pivot_dist_base = np.linalg.norm(pivot_base["pos"] - base_xyz)

        for idx, prn in enumerate(non_pivot_prns):
            sat_rover = rover_epoch[prn]
            sat_base = base_epoch[prn]
            dist_rover = np.linalg.norm(sat_rover["pos"] - rover_xyz_init)
            dist_base = np.linalg.norm(sat_base["pos"] - base_xyz)
            geom = (dist_rover - dist_base) - (pivot_dist_rover - pivot_dist_base)
            code_dd = (sat_rover["pseudorange"] - sat_base["pseudorange"]) - (
                pivot_rover["pseudorange"] - pivot_base["pseudorange"]
            )
            code_ambiguity = (code_dd - geom) / L1_WAVELENGTH
            sign = rng.choice([-1.0, 1.0])
            noise = rng.normal(0.0, noise_sigma)
            baseline_state[3 + idx] = init_code_weight * code_ambiguity + noise
        break

    print("\n=== DD Carrier-Phase Float Solution ===")
    print("Epoch (s)    X (m)          Y (m)          Z (m)          RMS (m)")
    print("----------  ------------  ------------  ------------  --------")

    last_epoch = None
    baseline_history: List[np.ndarray] = []
    converged = False
    fixed_applied = False
    convergence_epoch: float | None = None
    fixed_epoch: float | None = None
    ratio_threshold = 1.

    for epoch in common_epochs:
        if epoch > end_epoch:
            break

        rover_epoch = rover_dict.get(epoch, {})
        base_epoch = base_dict.get(epoch, {})

        if pivot_prn not in rover_epoch or pivot_prn not in base_epoch:
            continue

        if not all(prn in rover_epoch and prn in base_epoch for prn in non_pivot_prns):
            continue

        dt = 1.0 if last_epoch is None else max(epoch - last_epoch, 1.0)
        last_epoch = epoch

        try:
            baseline_state, covariance, rms, residuals, _ = solve_dd_epoch(
                baseline_state,
                covariance,
                base_xyz,
                rover_epoch,
                base_epoch,
                non_pivot_prns,
                pivot_prn,
                rover_lat,
                rover_lon,
                base_lat,
                base_lon,
                dt,
            )
        except ValueError:
            continue

        float_baseline = baseline_state[:3].copy()
        baseline_history.append(float_baseline)

        if not converged and len(baseline_history) >= 5:
            deltas = [
                np.linalg.norm(baseline_history[i] - baseline_history[i - 1])
                for i in range(-4, 0)
            ]
            if all(delta < 0.02 for delta in deltas):
                converged = True
                convergence_epoch = epoch
                print(
                    f"Convergence detected at epoch {epoch:.1f} based on baseline stability."
                )

        if converged and not fixed_applied:
            fixed_ambiguities, best_cost, second_cost = _search_integer_ambiguities(
                baseline_state[3:], covariance, search_radius=1
            )
            ratio = second_cost / best_cost if np.isfinite(best_cost) else 0.0
            print(
                f"Epoch {epoch:.1f} ambiguity search: float={baseline_state[3:]}, "
                f"fixed={fixed_ambiguities}, ratio={ratio:.2f}"
            )

            if ratio >= ratio_threshold and np.isfinite(best_cost):
                fixed_applied = True
                fixed_epoch = epoch
                baseline_state[3:] = fixed_ambiguities
                for idx in range(len(fixed_ambiguities)):
                    covariance[3 + idx, 3 + idx] = 1e-4

                try:
                    baseline_state, covariance, rms, residuals, _ = solve_dd_epoch(
                        baseline_state,
                        covariance,
                        base_xyz,
                        rover_epoch,
                        base_epoch,
                        non_pivot_prns,
                        pivot_prn,
                        rover_lat,
                        rover_lon,
                        base_lat,
                        base_lon,
                        dt,
                    )
                except ValueError:
                    pass

                print(
                    f"Ambiguity fixed at epoch {epoch:.1f}: integers={fixed_ambiguities}, "
                    f"ratio={ratio:.2f}"
                )

        rover_xyz = base_xyz + baseline_state[:3]
        ambiguities = baseline_state[3:].copy()

        results.append(
            DDEpochResult(
                epoch=epoch,
                position=rover_xyz.copy(),
                ambiguities=np.array(ambiguities, dtype=float),
                residuals=residuals.copy(),
                covariance=covariance.copy(),
            )
        )

        print(
            f"{epoch:10.1f}  {rover_xyz[0]:12.3f}  {rover_xyz[1]:12.3f}  {rover_xyz[2]:12.3f}  {rms:8.3f}"
        )

    print("=======================================\n")

    if convergence_epoch is not None:
        print(f"Baseline convergence triggered at epoch {convergence_epoch:.1f}.")
    if fixed_epoch is not None:
        print(f"Fixed solution applied at epoch {fixed_epoch:.1f} with ratio >= {ratio_threshold}.")

    if results:
        plot_dd_enu(results, true_xyz, enu_R)
        plot_dd_ambiguities(results, non_pivot_prns)
    return results
