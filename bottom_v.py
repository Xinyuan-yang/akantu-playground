#!/usr/bin/env python3

import os
import sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main(filename):

    # -----------------------------
    # Read data
    # -----------------------------
    reader = pv.get_reader(filename)

    reader.set_active_time_value(reader.time_values[0])
    mesh = reader.read()[0]
    connected = mesh.connectivity()
    region_ids = np.unique(connected.cell_data["RegionId"])
    
    lower_bodies = []
    
    for region_id in region_ids:
        body = connected.threshold(
            [region_id, region_id],
            scalars="RegionId",
            preference="cell",
        )
    
        mean_y = body.points[:, 1].mean()
        if mean_y<0: 
            lower_bodies.append(body)
    
    lower_block = lower_bodies[0]
    for k in lower_bodies[1:]:
        lower_block = lower_block.merge(k)
    mesh = lower_block
    points = mesh.points
    y = points[:, 1]

    y_interface = -0.1
    tol = 1e-8

    interface_ids = np.where(np.abs(y - y_interface) < tol)[0]

    interface_points = points[interface_ids]

    velocity = mesh.point_data["displacement"][interface_ids]
    v_t = velocity[:, 0]
    print(np.mean(v_t))
    x = interface_points[:, 0]
    order = np.argsort(x)

    # -----------------------------
    # Figure
    # -----------------------------
    fig, ax = plt.subplots()

    scat = ax.scatter(x[order], v_t[order])

    ax.set_title("Shear velocity along interface")
    ax.set_xlabel("x")
    ax.set_ylabel("Slip velocity")
    ax.grid(True)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-2e-5, -1e-5)

    # -----------------------------
    # Animation
    # -----------------------------
    def update(frame):

        reader.set_active_time_value(reader.time_values[frame])

        mesh = reader.read()[0]
        connected = mesh.connectivity()
        region_ids = np.unique(connected.cell_data["RegionId"])
        
        lower_bodies = []
        
        for region_id in region_ids:
            body = connected.threshold(
                [region_id, region_id],
                scalars="RegionId",
                preference="cell",
            )
        
            mean_y = body.points[:, 1].mean()
            if mean_y<0: 
                lower_bodies.append(body)
        
        lower_block = lower_bodies[0]
        for k in lower_bodies[1:]:
            lower_block = lower_block.merge(k)
        mesh = lower_block
        points = mesh.points
        velocity = mesh.point_data["displacement"][interface_ids]

        v_t = velocity[:, 0]

        data = np.column_stack((x[order], v_t[order]))
        scat.set_offsets(data)

        ax.set_title(
            f"Shear velocity   t = {reader.time_values[frame]:.4e}"
        )

        return scat,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(reader.time_values),
        interval=50,
        blit=False,
    )

    basename = os.path.splitext(os.path.basename(filename))[0]
    outfile = f"bottom_velocity_{basename}.mp4"

    print(f"Saving {outfile}...")

    ani.save(outfile, writer="ffmpeg", fps=5)

    print("Done.")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python interface_velocity.py simulation.pvd")
        sys.exit(1)

    main(sys.argv[1])