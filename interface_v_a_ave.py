#!/usr/bin/env python3

import os
import sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


def main(filename,v_ss):

    # -----------------------------
    # Read data
    # -----------------------------
    reader_file = "/scratch/xyang/build-akantu/build/paraview/" + filename + ".pvd"
    pd_file = "/scratch/xyang/build-akantu/build/friction-energies-" + filename + ".csv"
    reader = pv.get_reader(reader_file)
    df1 = pd.read_csv(pd_file)
    t_fin = df1["time"][len(df1["time"])-1]
    ratio = round(t_fin / reader.time_values[-1])
    
    reader.set_active_time_value(reader.time_values[0])
    mesh = reader.read()[0]
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
        if mean_y<0: 
            lower_bodies.append(body)
        else:
            upper_bodies.append(body)
    
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
    print(len(lower_interface_ids))
    lower_interface_points = lower_points[lower_interface_ids]
    upper_interface_points = upper_points[upper_interface_ids]

    lower_velocity = lower_block.point_data["velocity"][lower_interface_ids]
    upper_velocity = upper_block.point_data["velocity"][upper_interface_ids]
    lower_v_t = lower_velocity[:, 0]
    upper_v_t = upper_velocity[:, 0]

    top_u = upper_block.point_data["displacement"][upper_interface_ids]
    bottom_u = lower_block.point_data["displacement"][lower_interface_ids]
    u_tu = top_u[:, 0]
    u_tl = bottom_u[:,0]
    
    x_l = lower_block.points[lower_interface_ids][:, 0]
    x_t = upper_block.points[upper_interface_ids][:, 0]

    lower_x = lower_interface_points[:, 0]
    upper_x = upper_interface_points[:, 0]
    order_l = np.argsort(lower_x)
    order_u = np.argsort(upper_x)
    u_tl = u_tl[order_l]
    u_tu = u_tu[order_u]
    stored_u.append(np.mean(u_tl-u_tu))
    times.append(reader.time_values[0] * ratio)

    # -----------------------------
    # Figure
    # -----------------------------
    fig, (ax_history, ax_interface) = plt.subplots(
    2, 1,
    figsize=(12,16),
    constrained_layout=True
    )

    plot = ax_interface.plot(lower_x[order_l], lower_v_t[order_l] - upper_v_t[order_u],linewidth=  1)[0]

    ax_interface.set_title("Shear velocity along interface")
    ax_interface.set_xlabel("x")
    ax_interface.set_ylabel("Slip velocity")
    ax_interface.grid(True)

    ax_interface.set_xlim(lower_x.min(), lower_x.max())
    ax_interface.set_ylim(-5*v_ss, 2e-2)

    scat = ax_history.scatter(times[0],stored_u[0])

    ax_history.set_xlabel("Time")
    ax_history.set_ylabel("Cumulative slip")
    ax_history.set_title("Slip history")
    ax_history.set_xlim(0, t_fin)
    ax_history.set_ylim(0,v_ss*t_fin)
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
    
            mean_y = body.points[:, 1].mean()
            if mean_y<0: 
                lower_bodies.append(body)
            else:
                upper_bodies.append(body)
    
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
        lower_interface_points = lower_points[lower_interface_ids]
        upper_interface_points = upper_points[upper_interface_ids]
    
        lower_velocity = lower_block.point_data["velocity"][lower_interface_ids]
        upper_velocity = upper_block.point_data["velocity"][upper_interface_ids]
        lower_v_t = lower_velocity[:, 0]
        upper_v_t = upper_velocity[:, 0]
        
        top_u = upper_block.point_data["displacement"][upper_interface_ids]
        bottom_u = lower_block.point_data["displacement"][lower_interface_ids]
        u_tu = top_u[:, 0]
        u_tl = bottom_u[:,0]
        
        x_l = lower_block.points[lower_interface_ids][:, 0]
        x_t = upper_block.points[upper_interface_ids][:, 0]
    
        lower_x = lower_interface_points[:, 0]
        upper_x = upper_interface_points[:, 0]
        order_l = np.argsort(lower_x)
        order_u = np.argsort(upper_x)
        u_tl = u_tl[order_l]
        u_tu = u_tu[order_u]
        stored_u.append(np.mean(u_tu-u_tl))
        times.append(reader.time_values[frame]*ratio)

        data = np.column_stack((lower_x[order_l], lower_v_t[order_l] - upper_v_t[order_u]))
        data_cumu = np.stack([times, stored_u]).T
        plot.set_xdata(lower_x[order_l])
        plot.set_ydata(lower_v_t[order_l] - upper_v_t[order_u])
        scat.set_offsets(data_cumu)
        
        ax_interface.set_title(
            f"Relative shear velocity   t = {reader.time_values[frame]*ratio}"
        )
        time_marker.set_xdata([times[-1], times[-1]])
        return plot,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(reader.time_values),
        interval=50,
        blit=False,
    )

    basename = os.path.splitext(os.path.basename(reader_file))[0]
    outfile = f"interface_sync_{basename}.mp4"

    print(f"Saving {outfile}...")

    ani.save("Movs/" + outfile, writer="ffmpeg", fps=5)

    print("Done.")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python interface_velocity.py simulation.pvd")
        sys.exit(1)

    main(sys.argv[1])