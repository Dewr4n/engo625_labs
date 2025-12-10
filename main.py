from pathlib import Path
from ComputeDop import analyze_variations, compute_dops, geodetic_to_ecef
from ReadData import build_epoch_satellite_dict, read_obs_file, read_satellite_file
from DD import run_DD_positioning
from plottings import (
    plot_L1phase,
    plot_doppler,
    plot_dops,
    plot_pseudorange,
    plot_satellite_3D,
    plot_satellite_count,
)
from plsq import run_positioning

if __name__ == "__main__":
    RUN_PREVIOUS_LABS = False  # set True to regenerate Lab 1–3 figures

    data_dir = Path(__file__).resolve().parent
    sat_file = data_dir / "Satellites.sat"
    rover_obs_file = data_dir / "RemoteL1L2.obs"
    base_obs_file = data_dir / "BaseL1L2.obs"

    if not sat_file.exists():
        raise FileNotFoundError(f"Satellite file not found: {sat_file}")
    if not rover_obs_file.exists():
        raise FileNotFoundError(f"Observation file not found: {rover_obs_file}")
    if not base_obs_file.exists():
        raise FileNotFoundError(f"Observation file not found: {base_obs_file}")

    sat_data = read_satellite_file(sat_file)
    rover_obs_data = read_obs_file(rover_obs_file)
    base_obs_data = read_obs_file(base_obs_file)

    if RUN_PREVIOUS_LABS:
        sat_dict, epochs = build_epoch_satellite_dict(sat_data, rover_obs_data, max_epochs=3600)

        # Task 1 visuals
        plot_satellite_3D(sat_dict, epochs)
        plot_pseudorange(sat_dict, epochs)
        plot_L1phase(sat_dict, epochs)
        plot_doppler(sat_dict, epochs)
        analyze_variations(sat_dict, epochs)

        # Task 2
        # Remote receiver position (WGS-84)
        lat = 51 + 15 / 60 + 31.11582 / 3600  # 51.258643°
        lon = -(114 + 6 / 60 + 1.76988 / 3600)  # −114.100492°
        h = 1127.345
        approx_rcv_xyz = geodetic_to_ecef(lat, lon, h)
        print("Receiver ECEF position (m):", approx_rcv_xyz)

        dop_results = compute_dops(sat_dict, epochs, approx_rcv_xyz, lat, lon)
        plot_dops(dop_results, epochs)
        plot_satellite_count(dop_results, epochs)

        run_positioning(sat_data, rover_obs_data, base_obs_data, max_epochs=3600)

    # Lab 4: DD SLSQ
    run_DD_positioning(sat_data, rover_obs_data, base_obs_data, duration_s=1800)