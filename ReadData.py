# ====================================================
# Reading satellites and observation files
# ====================================================

from __future__ import annotations
import numpy as np


def read_satellite_file(filename):
    """Reading satellites file (.sat)"""
    record_len = 8  # [PRN, time, X, Y, Z, Xdot, Ydot, Zdot]
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float64)
    n_records = data.size // record_len
    sat = data.reshape((n_records, record_len))
    return sat[sat[:, 0] != 0]


def read_obs_file(filename):
    """Reading observation files (.obs)"""
    record_len = 6  # [PRN, time, pseudorange, L1phase, Doppler, L2phase]
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float64)
    n_records = data.size // record_len
    obs = data.reshape((n_records, record_len))
    return obs[obs[:, 0] != 0]

# ====================================================
# Building time & PRN data structure
# ====================================================

def build_epoch_satellite_dict(sat_data, obs_data, max_epochs=300):
    """Time as the first dimension, then the PRN as the second dimension along with all the satellite parameters"""
    epochs = np.unique(sat_data[:, 1])
    epochs = epochs[:max_epochs]  # how many epochs

    sat_dict = {}
    for t in epochs:
        sat_dict[t] = {}
        sats_this_epoch = sat_data[sat_data[:, 1] == t]
        obs_this_epoch = obs_data[obs_data[:, 1] == t]

        for prn in np.unique(sats_this_epoch[:, 0]):
            sat_entry = sats_this_epoch[sats_this_epoch[:, 0] == prn][0]
            obs_entry = obs_this_epoch[obs_this_epoch[:, 0] == prn]
            # some PRN might lack some observations
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