import numpy as np
import matplotlib.pyplot as plt
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