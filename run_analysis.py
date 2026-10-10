# run_analysis.py
import os

# Limit threaded BLAS/MKL inside each worker to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import multiprocessing as mp
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
import pickle

pv.OFF_SCREEN = True


# --------------------------------------------------------------------------- #
#  PASTE YOUR EXISTING split() FUNCTION HERE, OR IMPORT IT
# --------------------------------------------------------------------------- #
def split(con_mesh):
    region_ids = np.unique(con_mesh.cell_data["RegionId"])
    lower_bodies = []
    upper_bodies = []
    for region_id in region_ids:

        body = con_mesh.threshold(
            [region_id, region_id],
            scalars="RegionId",
            preference="cell",
        )

        mean_y = body.points[:, 1].mean()

        if mean_y < 0:
            lower_bodies.append(region_id)
        else:
            upper_bodies.append(region_id)

    lower_block = con_mesh.threshold(
        [lower_bodies[0], lower_bodies[0]],
        scalars="RegionId",
        preference="cell"
    )

    upper_block = con_mesh.threshold(
        [upper_bodies[0], upper_bodies[0]],
        scalars="RegionId",
        preference="cell"
    )
    for k in lower_bodies[1:]:
        temp = con_mesh.threshold(
            [k, k],
            scalars="RegionId",
            preference="cell"
        )
        lower_block = lower_block.merge(temp)

    for k in upper_bodies[1:]:
        temp = con_mesh.threshold(
            [k, k],
            scalars="RegionId",
            preference="cell"
        )
        upper_block = upper_block.merge(temp)

    y_interface = 0.0
    tol = 1e-8

    lower_ids_local = np.where(
        np.abs(lower_block.points[:, 1] - y_interface) < tol
    )[0]

    upper_ids_local = np.where(
        np.abs(upper_block.points[:, 1] - y_interface) < tol
    )[0]

    # Recover ORIGINAL mesh point IDs
    lower_ids_global = lower_block.point_data["orig_id"][lower_ids_local]
    upper_ids_global = upper_block.point_data["orig_id"][upper_ids_local]

    # Sort by x only once
    x_l = lower_block.points[lower_ids_local, 0]
    x_u = upper_block.points[upper_ids_local, 0]

    lower_ids_global = lower_ids_global[np.argsort(x_l)]
    upper_ids_global = upper_ids_global[np.argsort(x_u)]

    return lower_ids_global.astype(int), upper_ids_global.astype(int)

def process_timestep(args):
    """
    Worker that reads one timestep file and computes the average interface slip.
    """
    i, filepath, timestep, lower_interface_ids, upper_interface_ids, ratio = args

    reader = pv.get_reader(filepath)
    reader.disable_all_point_arrays()
    reader.disable_all_cell_arrays()
    reader.enable_point_array("displacement")

    mesh = reader.read()
    displacement = mesh.point_data["displacement"]

    u_lower = displacement[lower_interface_ids, 0]
    u_upper = displacement[upper_interface_ids, 0]

    avg_slip = float(np.mean(u_upper - u_lower))
    return i, float(timestep * ratio), avg_slip


# --------------------------------------------------------------------------- #
#  MAIN
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    # Use spawn to avoid potential VTK/PyVista deadlocks with fork()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    t0 = time.perf_counter()

    keys = [
        "SW_nh_peri_0.3_200_n_PARCOU",
       #"SW_nh_peri_0.3_300_n_COU",
       #"SW_nh_peri_0.3_400_n_COU",
       "SW_nh_peri_0.3_500_n_PARCOU",
       #"SW_nh_peri_0.3_600_n_COU",
       #"SW_nh_peri_0.3_700_n_COU",
       #"SW_nh_peri_0.3_800_n_COU",
       #"SW_nh_peri_0.3_900_n_COU",
       "SW_nh_peri_0.3_1000_n_PARCOU",
    ]

    v_ss = 2e-2
    results = {}
    t_fins = {}

    os.makedirs("Figs", exist_ok=True)

    fig_overlap, ax_overlap = plt.subplots(figsize=(7, 5))

    # Number of parallel workers: respect Slurm allocation if present
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"Running with {num_workers} workers")

    with mp.Pool(processes=num_workers) as pool:

        for key in keys:
            print(f"\nProcessing {key} ...")

            # --- read simulation metadata ---
            file = Path(
                "/scratch/xyang/build-akantu/build/paraview/" + key + ".pvd"
            )
            reader = pv.get_reader(file)

            tree = ET.parse(file)
            root = tree.getroot()
            datasets = root.findall(".//DataSet")

            df1 = pd.read_csv(
                "/scratch/xyang/build-akantu/build/friction-energies-"
                + key
                + ".csv"
            )
            t_fin = df1["time"].iloc[-1]
            t_fins[key] = t_fin

            time_values = np.asarray(reader.time_values)
            ratio = float(round(t_fin / time_values[-1]))

            # --- reference mesh / interface ids (computed once per key) ---
            reader.set_active_time_value(time_values[0])
            mesh0 = reader.read()[0]
            mesh0.point_data["orig_id"] = np.arange(mesh0.n_points)
            connected0 = mesh0.connectivity()
            lower_interface_ids, upper_interface_ids = split(connected0)

            # --- build task list for this key ---
            tasks = []
            for i, ds in enumerate(datasets):
                t = float(ds.attrib["timestep"])
                filepath = str(file.parent / ds.attrib["file"])
                tasks.append(
                    (
                        i,
                        filepath,
                        t,
                        lower_interface_ids,
                        upper_interface_ids,
                        ratio,
                    )
                )

            # --- parallel read/compute loop ---
            out = pool.map(process_timestep, tasks)

            # Restore chronological order
            out.sort(key=lambda x: x[0])
            times = np.array([x[1] for x in out])
            average_slip = np.array([x[2] for x in out])

            slip_velocity = np.gradient(average_slip, times)
            norm_velocity = slip_velocity / v_ss

            # --- store results ---
            results[key] = pd.DataFrame(
                {
                    "time": times,
                    "average_slip": average_slip,
                    "slip_velocity": slip_velocity,
                    "norm_velocity": norm_velocity,
                }
            )

            # --- individual average slip plot ---
            fig, ax = plt.subplots()
            ax.plot(times, average_slip, "k.", markersize=5)
            ax.set_xlim(times.min(), t_fin)
            ax.set_ylim(0, None)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Average interface slip [m]")
            ax.set_title("Average interfacial slip evolution")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.grid(True, alpha=0.3)

            fig.savefig(
                f"Figs/{key}_averageslip.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

            # --- add to overlapped velocity plot ---
            label = key.split("_")[4] + " elements"
            ax_overlap.plot(
                times,
                norm_velocity,
                label=label,
                linewidth=1.5,
            )

    # --- finalize overlap plot ---
    ax_overlap.set_xlabel("Time [s]")
    ax_overlap.set_ylabel("Normalized slip velocity")
    ax_overlap.legend(title="Mesh")
    ax_overlap.grid(True, alpha=0.3)
    fig_overlap.savefig(
        "Figs/velocity_overlap_SW.png",
        dpi=300,
        bbox_inches="tight",
    )

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved results to results.pkl")

    print("\nElapsed:", time.perf_counter() - t0)