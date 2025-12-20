# ====================================================
# Computing DOPs value
# ====================================================

import numpy as np
# ====================================================
# geodetic to ECEF
# ====================================================
def geodetic_to_ecef(lat, lon, h):
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f ** 2
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat_r) ** 2)
    X = (N + h) * np.cos(lat_r) * np.cos(lon_r)
    Y = (N + h) * np.cos(lat_r) * np.sin(lon_r)
    Z = (N * (1 - e2) + h) * np.sin(lat_r)
    return np.array([X, Y, Z])


# ====================================================
# ECEF to ENU
# ====================================================
def ecef_to_enu_rotation(lat, lon):
    slat, clat = np.sin(np.radians(lat)), np.cos(np.radians(lat))
    slon, clon = np.sin(np.radians(lon)), np.cos(np.radians(lon))
    R = np.array([
        [-slon,          clon,           0],
        [-slat*clon, -slat*slon,  clat],
        [ clat*clon,  clat*slon,  slat]
    ])
    return R

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
# Computing HDOP and VDOP
# ====================================================
def compute_dops(sat_dict, epochs, approx_rcv_xyz, lat, lon):
    R = ecef_to_enu_rotation(lat, lon)
    dop_results = []
    for t in epochs:
        prns = list(sat_dict[t].keys())
        if len(prns) < 4:
            dop_results.append((t, np.nan, np.nan, len(prns)))
            continue

        A = []
        for prn in prns:
            sat_xyz = sat_dict[t][prn]["pos"]
            vec = sat_xyz - approx_rcv_xyz
            rho = np.linalg.norm(vec)
            los = -vec / rho
            A.append([los[0], los[1], los[2], 1])
        A = np.array(A)

        Q = np.linalg.inv(A.T @ A)
        Q_xyz = Q[:3, :3]
        Q_enu = R @ Q_xyz @ R.T
        HDOP = np.sqrt(Q_enu[0, 0] + Q_enu[1, 1])
        VDOP = np.sqrt(Q_enu[2, 2])
        dop_results.append((t, HDOP, VDOP, len(prns)))
    return np.array(dop_results)