#!/usr/bin/env python3

import os
import sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


def main(filename, v_ss):
    """
    Animated diagnostics for a frictional interface.

    Panels
    ------
    1. Mean cumulative slip versus time.
    2. Spatial spectrogram P(k,t) of the instantaneous interface slip velocity V(x,t).
    3. Instantaneous interface slip-velocity profile V(x,t).

    The spectrum is taken in SPACE, not time:

        V(x,t) --FFT in x--> V_hat(k,t)

    so the vertical axis of the spectrogram is wavenumber k [rad/m].
    """

    # ------------------------------------------------------------
    # Read files
    # ------------------------------------------------------------
    reader_file = (
        "/scratch/xyang/build-akantu/build/paraview/"
        + filename
        + ".pvd"
    )

    energy_file = (
        "/scratch/xyang/build-akantu/build/friction-energies-"
        + filename
        + ".csv"
    )

    reader = pv.get_reader(reader_file)
    df = pd.read_csv(energy_file)

    t_fin = df["time"].iloc[-1]

    if len(reader.time_values) == 0 or reader.time_values[-1] == 0:
        raise ValueError("The PVD file must contain a non-zero final time.")

    ratio = t_fin / reader.time_values[-1]

    # ------------------------------------------------------------
    # Construct interface mapping once
    # ------------------------------------------------------------
    reader.set_active_time_value(reader.time_values[0])

    mesh = reader.read()[0]

    mesh.point_data["orig_id"] = np.arange(mesh.n_points)

    connected = mesh.connectivity()

    region_ids = np.unique(connected.cell_data["RegionId"])

    lower_bodies = []
    upper_bodies = []

    for region_id in region_ids:

        body = connected.threshold(
            [region_id, region_id],
            scalars="RegionId",
            preference="cell",
        )

        if body.points[:, 1].mean() < 0:
            lower_bodies.append(body)
        else:
            upper_bodies.append(body)

    if not lower_bodies or not upper_bodies:
        raise ValueError("Could not identify both sides of the interface.")

    lower_block = lower_bodies[0]
    upper_block = upper_bodies[0]

    for body in lower_bodies[1:]:
        lower_block = lower_block.merge(body)

    for body in upper_bodies[1:]:
        upper_block = upper_block.merge(body)

    lower_points = lower_block.points
    upper_points = upper_block.points

    y_interface = 0.0
    tol = 1e-8

    lower_interface_ids = np.where(
        np.abs(lower_points[:, 1] - y_interface) < tol
    )[0]

    upper_interface_ids = np.where(
        np.abs(upper_points[:, 1] - y_interface) < tol
    )[0]

    if len(lower_interface_ids) == 0 or len(upper_interface_ids) == 0:
        raise ValueError("No interface points found at y = 0.")

    if len(lower_interface_ids) != len(upper_interface_ids):
        raise ValueError(
            "Upper and lower interface point counts differ."
        )

    # ------------------------------------------------------------
    # Pair points by x coordinate
    # ------------------------------------------------------------
    lower_x = lower_points[lower_interface_ids, 0]
    upper_x = upper_points[upper_interface_ids, 0]

    lower_order = np.argsort(lower_x)
    upper_order = np.argsort(upper_x)

    lower_x = lower_x[lower_order]
    upper_x = upper_x[upper_order]

    if not np.allclose(lower_x, upper_x, atol=tol, rtol=0):
        raise ValueError(
            "Upper and lower interface points cannot be paired by x."
        )

    lower_ids_global = np.asarray(
        lower_block.point_data["orig_id"][
            lower_interface_ids
        ][lower_order],
        dtype=int,
    )

    upper_ids_global = np.asarray(
        upper_block.point_data["orig_id"][
            upper_interface_ids
        ][upper_order],
        dtype=int,
    )

    # ------------------------------------------------------------
    # Check spatial spacing
    # ------------------------------------------------------------
    dx_values = np.diff(lower_x)
    dx = np.mean(dx_values)

    if not np.allclose(dx_values, dx, rtol=1e-4, atol=1e-12):
        raise ValueError(
            "Interface points are not uniformly spaced. "
            "np.fft assumes uniform spatial sampling."
        )

    # Spatial Fourier frequencies:
    #
    # cycles/m:
    spatial_frequency = np.fft.rfftfreq(
        len(lower_x),
        d=dx,
    )

    # angular wavenumber [rad/m]:
    k = 2.0 * np.pi * spatial_frequency

    # Nyquist wavenumber:
    k_nyquist = np.pi / dx

    print(f"Interface nodes      = {len(lower_x)}")
    print(f"dx                   = {dx:.6e} m")
    print(f"Nyquist k = pi/dx    = {k_nyquist:.6e} rad/m")

    # ------------------------------------------------------------
    # Read all timesteps once
    # ------------------------------------------------------------
    times = []
    cumulative_slip = []
    interface_velocity_history = []
    spatial_power_history = []

    # Spatial Hann window suppresses spectral leakage caused by the
    # finite interface interval.
    spatial_window = np.hanning(len(lower_x))

    # Normalization used for comparing frames.
    window_norm = np.sum(spatial_window**2)

    for frame, time_value in enumerate(reader.time_values):

        reader.set_active_time_value(time_value)

        mesh = reader.read()[0]

        displacement = mesh.point_data["displacement"]
        velocity = mesh.point_data["velocity"]

        u_lower = displacement[lower_ids_global, 0]
        u_upper = displacement[upper_ids_global, 0]

        v_lower = velocity[lower_ids_global, 0]
        v_upper = velocity[upper_ids_global, 0]

        # Keep the same convention as your original plot.
        v_rel = v_lower - v_upper

        # Mean cumulative relative slip.
        slip_mean = np.mean(-u_lower + u_upper)

        # --------------------------------------------------------
        # SPATIAL spectrum of V(x) at this instant
        # --------------------------------------------------------

        # Remove the k=0 / mean sliding component.
        v_fluct = v_rel - np.mean(v_rel)

        # Window in x to reduce finite-domain leakage.
        v_windowed = v_fluct * spatial_window

        fft_vals = np.fft.rfft(v_windowed)

        # Power versus spatial wavenumber.
        power = np.abs(fft_vals)**2 / window_norm

        # One-sided spectrum correction.
        if len(v_windowed) % 2 == 0:
            if len(power) > 2:
                power[1:-1] *= 2.0
        else:
            if len(power) > 1:
                power[1:] *= 2.0

        times.append(time_value * ratio)
        cumulative_slip.append(slip_mean)
        interface_velocity_history.append(v_rel.copy())
        spatial_power_history.append(power)

        print(
            f"Reading step {frame + 1}/{len(reader.time_values)}",
            end="\r",
        )

    print()

    times = np.asarray(times)
    cumulative_slip = np.asarray(cumulative_slip)
    interface_velocity_history = np.asarray(
        interface_velocity_history
    )
    spatial_power_history = np.asarray(
        spatial_power_history
    )

    # ------------------------------------------------------------
    # Dominant non-zero spatial wavenumber
    # ------------------------------------------------------------
    dominant_k = np.zeros(len(times))

    for i, power in enumerate(spatial_power_history):

        if len(power) > 1:
            peak_id = np.argmax(power[1:]) + 1
            dominant_k[i] = k[peak_id]

    # ------------------------------------------------------------
    # Save numerical results
    # ------------------------------------------------------------
    os.makedirs("Spectra", exist_ok=True)

    dominant_df = pd.DataFrame(
        {
            "time": times,
            "cumulative_slip": cumulative_slip,
            "dominant_k_rad_per_m": dominant_k,
            "dominant_wavelength_m": np.where(
                dominant_k > 0,
                2.0 * np.pi / dominant_k,
                np.nan,
            ),
        }
    )

    dominant_csv = os.path.join(
        "Spectra",
        f"dominant_spatial_frequency_{filename}.csv",
    )

    dominant_df.to_csv(
        dominant_csv,
        index=False,
    )

    np.savez(
        os.path.join(
            "Spectra",
            f"spatial_spectrogram_{filename}.npz",
        ),
        time=times,
        cumulative_slip=cumulative_slip,
        x=lower_x,
        k=k,
        power=spatial_power_history,
        dominant_k=dominant_k,
        dx=dx,
    )

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    fig, (
        ax_history,
        ax_spectrum,
        ax_interface,
    ) = plt.subplots(
        3,
        1,
        figsize=(12, 18),
        constrained_layout=True,
    )

    # ------------------------------------------------------------
    # Panel 1: cumulative slip
    # ------------------------------------------------------------
    ax_history.plot(
        times,
        cumulative_slip,
        linewidth=1.5,
    )

    slip_marker, = ax_history.plot(
        [times[0]],
        [cumulative_slip[0]],
        marker="o",
    )

    history_time_marker = ax_history.axvline(
        times[0],
        linestyle="--",
    )

    ax_history.set_xlabel("Time [s]")
    ax_history.set_ylabel("Mean cumulative slip")
    ax_history.set_title("Cumulative slip history")
    ax_history.set_xlim(times[0], times[-1])
    ax_history.grid(True)

    # ------------------------------------------------------------
    # Panel 2: spatial spectrogram
    # ------------------------------------------------------------
    #
    # Exclude k = 0 because we explicitly removed the spatial mean.
    positive_k = k > 0
    k_plot = k[positive_k]

    # Initialize with the first frame only.
    # The full x extent is kept fixed, while future columns are NaN.
    spectrogram_display = np.full(
        (len(k_plot), len(times)),
        np.nan,
    )

    power_db_all = 10.0 * np.log10(
        spatial_power_history[:, positive_k]
        + np.finfo(float).tiny
    )

    spectrogram_display[:, 0] = power_db_all[0]

    # Use a fixed color scale based on the full dataset so that colors
    # mean the same thing throughout the animation.
    finite_db = power_db_all[np.isfinite(power_db_all)]

    if len(finite_db) == 0:
        vmin = -100.0
        vmax = 0.0
    else:
        # Robust limits avoid one extreme value ruining contrast.
        vmin = np.percentile(finite_db, 5)
        vmax = np.percentile(finite_db, 99.5)

    spec_mesh = ax_spectrum.pcolormesh(
        times,
        k_plot,
        spectrogram_display,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    colorbar = fig.colorbar(
        spec_mesh,
        ax=ax_spectrum,
    )

    colorbar.set_label("Spatial spectral power [dB]")

    dominant_line, = ax_spectrum.plot(
        [times[0]],
        [dominant_k[0]],
        linewidth=1.5,
        label="Dominant k",
    )

    spectrum_time_marker = ax_spectrum.axvline(
        times[0],
        linestyle="--",
    )

    ax_spectrum.set_xlabel("Time [s]")
    ax_spectrum.set_ylabel("Wavenumber k [rad/m]")
    ax_spectrum.set_title(
        "Spatial spectrogram of interface slip velocity"
    )

    ax_spectrum.set_xlim(
        times[0],
        times[-1],
    )

    if len(k_plot) > 0:
        ax_spectrum.set_ylim(
            0,
            k_plot[-1],
        )

    ax_spectrum.legend()
    ax_spectrum.grid(True)

    # ------------------------------------------------------------
    # Panel 3: instantaneous V(x)
    # ------------------------------------------------------------
    interface_line, = ax_interface.plot(
        lower_x,
        interface_velocity_history[0],
        linewidth=1,
    )

    ax_interface.set_xlabel("x")
    ax_interface.set_ylabel("Slip velocity")
    ax_interface.set_title(
        f"Relative shear velocity   t = {times[0]:.6g}"
    )

    ax_interface.set_xlim(
        lower_x.min(),
        lower_x.max(),
    )

    ax_interface.set_ylim(
        -5.0 * v_ss,
        0.2 * v_ss,
    )

    ax_interface.grid(True)

    # ------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------
    def update(frame):

        current_t = times[frame]

        # Instantaneous interface velocity.
        interface_line.set_ydata(
            interface_velocity_history[frame]
        )

        ax_interface.set_title(
            f"Relative shear velocity   t = {current_t:.6g}"
        )

        # Cumulative slip marker.
        slip_marker.set_data(
            [current_t],
            [cumulative_slip[frame]],
        )

        history_time_marker.set_xdata(
            [current_t, current_t]
        )

        # Add all spectrum columns up to the current time.
        spectrogram_display[:, :frame + 1] = (
            power_db_all[:frame + 1].T
        )

        # pcolormesh stores the cell values flattened.
        spec_mesh.set_array(
            spectrogram_display.ravel()
        )

        # Dominant-k trajectory only up to the current time.
        dominant_line.set_data(
            times[:frame + 1],
            dominant_k[:frame + 1],
        )

        spectrum_time_marker.set_xdata(
            [current_t, current_t]
        )

        print(
            f"Animating step {frame + 1}/{len(times)}",
            end="\r",
        )

        return (
            interface_line,
            slip_marker,
            history_time_marker,
            spec_mesh,
            dominant_line,
            spectrum_time_marker,
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=range(len(times)),
        interval=50,
        blit=False,
    )

    # ------------------------------------------------------------
    # Save movie
    # ------------------------------------------------------------
    basename = os.path.splitext(
        os.path.basename(reader_file)
    )[0]

    outfile = (
        f"interface_spatial_spectrum_{basename}.mp4"
    )

    os.makedirs(
        "Movs",
        exist_ok=True,
    )

    print(f"\nSaving {outfile}...")

    ani.save(
        os.path.join(
            "Movs",
            outfile,
        ),
        writer="ffmpeg",
        fps=5,
    )

    print("Done.")
    print(
        f"Dominant-k history: {dominant_csv}"
    )
    print(
        "Spatial spectrogram data: "
        + os.path.join(
            "Spectra",
            f"spatial_spectrogram_{filename}.npz",
        )
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print("Usage:")
        print(
            "python interface_frequency_slip.py "
            "simulation_name steady_state_velocity"
        )

        sys.exit(1)

    main(
        sys.argv[1],
        float(sys.argv[2]),
    )
