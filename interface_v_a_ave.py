#!/usr/bin/env python3

import os
import sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


def main(filename, v_ss):

    # -----------------------------
    # Read data
    # -----------------------------
    reader_file = "/scratch/xyang/build-akantu/build/paraview/" + filename + ".pvd"
    pd_file = "/scratch/xyang/build-akantu/build/friction-energies-" + filename + ".csv"
    reader = pv.get_reader(reader_file)
    df1 = pd.read_csv(pd_file)
    t_fin = df1["time"][len(df1["time"])-1]
    if len(reader.time_values) == 0 or reader.time_values[-1] == 0:
        raise ValueError("The PVD file must contain a non-zero final time.")
    ratio = t_fin / reader.time_values[-1]
    
    reader.set_active_time_value(reader.time_values[0])
    mesh = reader.read()[0]
    mesh.point_data["orig_id"] = np.arange(mesh.n_points)
    connected = mesh.connectivity()
    region_ids = np.unique(connected.cell_data["RegionId"])
    
    lower_bodies = []
    upper_bodies = []
    stored_u = []
    times = []
    
    for region_id in region_ids:
        body = connected.threshold(
            [region_id, region_id],
            scalars="RegionId",
            preference="cell",
        )
    
        mean_y = body.points[:, 1].mean()
        if mean_y < 0:
            lower_bodies.append(body)
        else:
            upper_bodies.append(body)

    if not lower_bodies or not upper_bodies:
        raise ValueError("Could not identify both sides of the interface.")
    
    lower_block = lower_bodies[0]
    upper_block = upper_bodies[0]
    for k in lower_bodies[1:]:
        lower_block = lower_block.merge(k)
    for k in upper_bodies[1:]:
        upper_block = upper_block.merge(k)
    lower_points = lower_block.points
    lower_y = lower_points[:, 1]
    upper_points = upper_block.points
    upper_y = upper_points[:,1]

    y_interface = 0.0
    tol = 1e-8

    lower_interface_ids = np.where(np.abs(lower_y - y_interface) < tol)[0]
    upper_interface_ids = np.where(np.abs(upper_y - y_interface) < tol)[0]
    if len(lower_interface_ids) == 0 or len(upper_interface_ids) == 0:
        raise ValueError("No interface points found at y = 0.")
    if len(lower_interface_ids) != len(upper_interface_ids):
        raise ValueError("The upper and lower interface point counts differ.")

    # Cache mesh point IDs for the interface once.  These IDs are the pointers
    # used for every subsequent time step, so no connectivity/threshold work is
    # performed in the animation callback.
    lower_x = lower_points[lower_interface_ids, 0]
    upper_x = upper_points[upper_interface_ids, 0]
    lower_order = np.argsort(lower_x)
    upper_order = np.argsort(upper_x)
    lower_x = lower_x[lower_order]
    upper_x = upper_x[upper_order]
    if not np.allclose(lower_x, upper_x, atol=tol, rtol=0):
        raise ValueError("Upper and lower interface points cannot be paired by x.")

    lower_ids_global = np.asarray(
        lower_block.point_data["orig_id"][lower_interface_ids][lower_order], dtype=int
    )
    upper_ids_global = np.asarray(
        upper_block.point_data["orig_id"][upper_interface_ids][upper_order], dtype=int
    )

    displacement = mesh.point_data["displacement"]
    velocity = mesh.point_data["velocity"]
    stored_u.append(
        np.mean(displacement[lower_ids_global, 0] - displacement[upper_ids_global, 0])
    )
    times.append(reader.time_values[0] * ratio)

    # -----------------------------
    # Figure
    # -----------------------------
    fig, (ax_history, ax_interface) = plt.subplots(
    2, 1,
    figsize=(12,16),
    constrained_layout=True
    )

    plot = ax_interface.plot(
        lower_x,
        velocity[lower_ids_global, 0] - velocity[upper_ids_global, 0],
        linewidth=1,
    )[0]

    ax_interface.set_title("Shear velocity along interface")
    ax_interface.set_xlabel("x")
    ax_interface.set_ylabel("Slip velocity")
    ax_interface.grid(True)

    ax_interface.set_xlim(lower_x.min(), lower_x.max())
    ax_interface.set_ylim(-5*v_ss, 0.2*v_ss)

    scat = ax_history.scatter(times[0],stored_u[0])

    ax_history.set_xlabel("Time")
    ax_history.set_ylabel("Cumulative slip")
    ax_history.set_title("Slip history")
    ax_history.set_xlim(0, t_fin)
    ax_history.grid(True)
    
    # Vertical marker showing the current animation time
    time_marker = ax_history.axvline(
        times[0],
        linestyle="--"
    )


    # -----------------------------
    # Animation
    # -----------------------------
    def update(frame):

        reader.set_active_time_value(reader.time_values[frame])

        mesh = reader.read()[0]
        displacement = mesh.point_data["displacement"]
        u_lower = displacement[lower_ids_global, 0]
        u_upper = displacement[upper_ids_global, 0]
        stored_u.append(np.mean(-u_lower + u_upper))

        lower_velocity = mesh.point_data["velocity"][lower_ids_global]
        upper_velocity = mesh.point_data["velocity"][upper_ids_global]
        lower_v_t = lower_velocity[:, 0]
        upper_v_t = upper_velocity[:, 0]
        times.append(reader.time_values[frame]*ratio)

        data_cumu = np.stack([times, stored_u]).T
        plot.set_xdata(lower_x)
        plot.set_ydata(lower_v_t - upper_v_t)
        scat.set_offsets(data_cumu)
        ax_history.set_ylim(0, stored_u[-1])
        
        ax_interface.set_title(
            f"Relative shear velocity   t = {reader.time_values[frame]*ratio}"
        )
        time_marker.set_xdata([times[-1], times[-1]])
        print(f"Step {frame}", end = "\r")
        return plot,

    ani = FuncAnimation(
        fig,
        update,
        # Frame zero is already used to initialise the artists and history.
        frames=range(1, len(reader.time_values)),
        interval=50,
        blit=False,
    )

    basename = os.path.splitext(os.path.basename(reader_file))[0]
    outfile = f"interface_sync_{basename}.mp4"

    print(f"Saving {outfile}...")

    os.makedirs("Movs", exist_ok=True)
    ani.save(os.path.join("Movs", outfile), writer="ffmpeg", fps=5)

    print("Done.")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage:")
        print("python interface_v_a_ave.py simulation_name steady_state_velocity")
        sys.exit(1)

    main(sys.argv[1], float(sys.argv[2]))
