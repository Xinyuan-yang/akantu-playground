#!/usr/bin/env python3
"""Plot a frictional slip front and its instantaneous propagation speed.

The front is defined from Akantu's friction-interface dump, rather than from a
velocity threshold in the volume output.  A point belongs to the ruptured
region when its tangential friction traction has reached its local
``frictional_strength``.  The latter is the strength used by the friction law
(for Coulomb friction, :math:`mu_d |p|`), so this remains valid when pressure
or the dynamic coefficient varies along the interface.

Only a PNG is written.

Example
-------
python slip_front.py SW_nh_trac_0.3_100_n_n \
    --directory build/paraview --output-directory figures
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Track a slip front where |friction traction| reaches the local "
            "dynamic frictional strength."
        )
    )
    parser.add_argument("base_name", help='Base PVD filename without ".pvd".')
    parser.add_argument(
        "legacy_average_slip_velocity_threshold",
        nargs="?",
        type=float,
        help=(
            "Deprecated and ignored. Kept so existing invocations using the "
            "old velocity-threshold argument continue to run."
        ),
    )
    parser.add_argument(
        "--directory",
        default="/scratch/xyang/build-akantu/build/paraview",
        help="Directory containing the PVD file.",
    )
    parser.add_argument(
        "--interface-info",
        type=Path,
        help=(
            "Path to the *_friction_interface.info file. By default it is "
            "looked up next to the PVD directory."
        ),
    )
    parser.add_argument(
        "--strength-ratio",
        type=float,
        default=0.99,
        help=(
            "A point is slipping when |traction| / frictional_strength is at "
            "least this value (default: 0.99)."
        ),
    )
    parser.add_argument(
        "--direction",
        choices=("right", "left"),
        default="right",
        help="Direction of the front to track.",
    )
    parser.add_argument(
        "--stop-position",
        type=float,
        default=2.5,
        help="Stop once the front reaches this x-position.",
    )
    parser.add_argument(
        "--output-directory",
        default=".",
        help="Directory for the PNG output.",
    )
    parser.add_argument(
        "--cs",
        default = 1600,
        help = "Shear wave speed",
    )
    return parser.parse_args()


def read_pvd_times(pvd_path):
    """Read PVD times without loading any VTK meshes."""
    root = ET.parse(pvd_path).getroot()
    times = [float(node.attrib["timestep"]) for node in root.iter("DataSet")]
    if not times:
        raise RuntimeError(f"No timesteps were found in {pvd_path}.")
    return np.asarray(times)


def find_interface_info(pvd_path, base_name, requested_path):
    if requested_path is not None:
        if not requested_path.is_file():
            raise FileNotFoundError(f"Interface info file not found: {requested_path}")
        return requested_path

    candidates = (
        pvd_path.parent / f"{base_name}_friction_interface.info",
        pvd_path.parent.parent / f"{base_name}_friction_interface.info",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find the friction-interface info file. Use --interface-info "
        "to specify it explicitly. Looked for: "
        + ", ".join(str(path) for path in candidates)
    )


def read_field_paths(info_path):
    """Resolve the text dump paths declared by an Akantu interface .info file."""
    field_description = None
    for line in info_path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0] == "field_description":
            field_description = parts[1]
            break
    if field_description is None:
        raise RuntimeError(f'No "field_description" entry in {info_path}.')

    fields_path = info_path.parent / field_description
    if not fields_path.is_file():
        raise FileNotFoundError(f"Interface fields file not found: {fields_path}")

    paths = {}
    for line in fields_path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 6:
            raise RuntimeError(f"Unexpected field description: {line}")
        name, components, relative_directory = parts[0], int(parts[4]), parts[5]
        paths[name] = (
            fields_path.parent / relative_directory / f"{info_path.stem}_{name}.out",
            components,
        )

    required = {"position", "friction_traction", "frictional_strength"}
    missing = required.difference(paths)
    if missing:
        raise RuntimeError(
            f"{fields_path} is missing required field(s): {', '.join(sorted(missing))}."
        )
    return paths


def interface_field_files(path):
    """Return the serial dump or all rank-local ``.procX.out`` dumps."""
    # In MPI runs Akantu writes, for example,
    # ``..._position.proc0.out`` instead of ``..._position.out``.  Prefer the
    # rank-local files if present: a leftover serial file must not be mixed
    # into a parallel result.
    parallel_files = sorted(path.parent.glob(f"{path.stem}.proc*.out"))
    if parallel_files:
        return parallel_files
    if path.is_file():
        return [path]
    raise FileNotFoundError(
        f"Neither serial nor parallel interface dump found for {path}."
    )


def read_snapshots(path, components, number_of_times):
    """Load serial or MPI rank-local records and align them by PVD timestep."""
    snapshots = []
    for field_path in interface_field_files(path):
        values = np.loadtxt(field_path, ndmin=2)
        if values.shape[1] != components:
            raise RuntimeError(
                f"{field_path} has {values.shape[1]} component(s), expected "
                f"{components}."
            )
        if values.shape[0] % number_of_times:
            raise RuntimeError(
                f"{field_path} has {values.shape[0]} rows, which cannot be "
                f"aligned with the {number_of_times} PVD timesteps. Specify "
                "the matching PVD/info pair."
            )
        snapshots.append(values.reshape(number_of_times, -1, components))

    # Each proc file contains every timestep. Concatenate *within* each
    # timestep, not the raw files end-to-end, which would scramble ranks and
    # times.
    return np.concatenate(snapshots, axis=1)


def slip_front_position(position, traction, strength, direction, strength_ratio):
    """Return the interpolated outer edge of the frictionally mobilized region."""
    order = np.argsort(position[:, 0])
    x = position[order, 0]
    traction = traction[order]
    strength = strength[order]
    strength = np.abs(strength)
    valid = strength > np.finfo(float).eps
    mobilization = np.full_like(strength, np.nan, dtype=float)
    mobilization[valid] = np.abs(traction[valid, 0]) / strength[valid]
    active = mobilization >= strength_ratio
    if not active.any():
        return np.nan

    # Interpolate |tau| / strength = strength_ratio between the last slipping
    # point and its non-slipping neighbour. This avoids quantising the front to
    # the interface mesh spacing.
    if direction == "right":
        inside = np.flatnonzero(active)[-1]
        outside = inside + 1
    else:
        inside = np.flatnonzero(active)[0]
        outside = inside - 1

    if 0 <= outside < len(x) and np.isfinite(mobilization[outside]):
        inside_value = mobilization[inside] - strength_ratio
        outside_value = mobilization[outside] - strength_ratio
        denominator = outside_value - inside_value
        if denominator != 0:
            fraction = np.clip(-inside_value / denominator, 0.0, 1.0)
            return x[inside] + fraction * (x[outside] - x[inside])
    return x[inside]


def main():
    args = parse_args()
    if not 0 < args.strength_ratio <= 1:
        raise ValueError("--strength-ratio must be in the interval (0, 1].")
    if args.legacy_average_slip_velocity_threshold is not None:
        print("Ignoring deprecated velocity-threshold argument; using frictional strength.")

    pvd_path = Path(args.directory) / f"{args.base_name}.pvd"
    if not pvd_path.is_file():
        raise FileNotFoundError(f"PVD file not found: {pvd_path}")
    info_path = find_interface_info(pvd_path, args.base_name, args.interface_info)
    field_paths = read_field_paths(info_path)
    times = read_pvd_times(pvd_path)

    print(f"Reading PVD times: {pvd_path}")
    print(f"Reading interface fields: {info_path}")
    positions = read_snapshots(*field_paths["position"], len(times))
    tractions = read_snapshots(*field_paths["friction_traction"], len(times))
    strengths = read_snapshots(*field_paths["frictional_strength"], len(times))[:, :, 0]

    front_positions = np.fromiter(
        (
            slip_front_position(position, traction, strength, args.direction, args.strength_ratio)
            for position, traction, strength in zip(positions, tractions, strengths)
        ),
        dtype=float,
        count=len(times),
    )
    valid = np.isfinite(front_positions)
    if not valid.any():
        raise RuntimeError("No interface point reached the requested friction-strength ratio.")

    times = times[valid]
    front_positions = front_positions[valid]
    reached_stop = (
        front_positions >= args.stop_position
        if args.direction == "right"
        else front_positions <= args.stop_position
    )
    if reached_stop.any():
        end = np.flatnonzero(reached_stop)[0] + 1
        times, front_positions = times[:end], front_positions[:end]

    # This is the local front speed, not a global linear fit.
    front_speed = (
        np.gradient(front_positions, times) if len(times) > 1 else np.full(1, np.nan)
    )

    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    cs = args.cs
    png_path = output_dir / f"{args.base_name}_slip_front.png"

    fig, (ax_position, ax_speed) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    ax_position.plot(times, front_positions, "o-", label="frictional-strength front")
    ax_position.set_ylabel("front position x")
    ax_position.set_title(f"Slip-front evolution: {args.base_name}")
    ax_position.grid(True)
    ax_position.legend()

    ax_speed.plot(times, front_speed, color="tab:orange", label="d x_front / d t")
    ax_speed.set_ylim(0,cs)
    ax_speed.set(xlabel="time", ylabel="front speed")
    ax_speed.grid(True)
    ax_speed.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to: {png_path}")


if __name__ == "__main__":
    main()
