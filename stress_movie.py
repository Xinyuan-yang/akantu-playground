#!/usr/bin/env python3

import argparse
import importlib
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyvista as pv

DEFAULT_FOLDER = "/scratch/xyang/build-akantu/build/paraview/"
DEFAULT_OUTPUT_FOLDER = "Stress/"
DEFAULT_KEY = "results"
DEFAULT_E = 7.98e9
DEFAULT_NU = 0.0
DEFAULT_WARP_FACTOR = 100.0
DEFAULT_FPS = 20
DEFAULT_DISP_NAME = "displacement"
DEFAULT_FIELD = "sigma_xy"
DEFAULT_PERCENTILE = 95.0
DEFAULT_RANGE_STRIDE = 10
DEFAULT_WINDOW = (1920, 1440)
DEFAULT_ZOOM = 1.4

AVAILABLE_FIELDS = (
    "von_mises",
    "sigma_vm",
    "sigma_xx",
    "sigma_yy",
    "sigma_xy",
    "epsilon_xx",
    "epsilon_yy",
    "epsilon_xy",
    "gamma_xy",
)

STRESS_FIELDS = {"sigma_xx", "sigma_yy", "sigma_xy", "von_mises"}


def canonical_field(name):
    return "von_mises" if name == "sigma_vm" else name


DISPLAY_NAMES = {
    "sigma_xy": r"$\tau_{xy}$",
    "sigma_xx": r"$\sigma_{xx}$",
    "sigma_yy": r"$\sigma_{yy}$",
    "epsilon_xx": r"$\varepsilon_{xx}$",
    "epsilon_yy": r"$\varepsilon_{yy}$",
    "epsilon_xy": r"$\varepsilon_{xy}$",
    "gamma_xy": r"$\gamma_{xy}$",
    "von_mises": "von Mises",
}

PLAIN_NAMES = {
    "sigma_xy": "tau_xy",
    "sigma_xx": "sigma_xx",
    "sigma_yy": "sigma_yy",
    "epsilon_xx": "epsilon_xx",
    "epsilon_yy": "epsilon_yy",
    "epsilon_xy": "epsilon_xy",
    "gamma_xy": "gamma_xy",
    "von_mises": "von Mises",
}


def setup_mathtext_font():
    """
    Enable Times-like mathtext rendering (STIX) for VTK scalar-bar titles.
    VTK renders mathtext titles through matplotlib when available.
    Returns True if mathtext titles (e.g. $\\tau_{xy}$) can be used.
    """
    try:
        import matplotlib
    except ImportError:
        return False

    matplotlib.rcParams["mathtext.fontset"] = "stix"
    return True


def display_name(field, mathtext_ok=True):
    mapping = DISPLAY_NAMES if mathtext_ok else PLAIN_NAMES
    return mapping.get(field, field)


def display_scale(field):
    """Display-only scale factor. Stress fields are shown in MPa (data stays in Pa)."""
    return 1e6 if field in STRESS_FIELDS else 1.0


def unit_suffix(field):
    """Unit label for titles/prints. Strains are dimensionless."""
    return " [MPa]" if field in STRESS_FIELDS else ""


def get_requested_cpus():
    """Return the number of CPUs Slurm allocated for this job/task."""
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus:
        try:
            return int(cpus)
        except ValueError:
            pass
    job_cpus = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if job_cpus:
        first = job_cpus.split()[0].split("(")[0]
        try:
            return int(first)
        except ValueError:
            pass
    return os.cpu_count() or 1


# Per-worker globals for parallel cache building and range sampling.
_CACHE_READER = None
_CACHE_FIELD = None
_CACHE_E = None
_CACHE_NU = None
_CACHE_DISP_NAME = None

_RANGE_READER = None
_RANGE_FIELD = None
_RANGE_E = None
_RANGE_NU = None
_RANGE_DISP_NAME = None


def _init_cache_worker(source_pvd, field, E, nu, disp_name):
    global _CACHE_READER, _CACHE_FIELD, _CACHE_E, _CACHE_NU, _CACHE_DISP_NAME
    os.environ["OMP_NUM_THREADS"] = "1"
    _CACHE_READER = pv.get_reader(str(source_pvd))
    _CACHE_FIELD = field
    _CACHE_E = E
    _CACHE_NU = nu
    _CACHE_DISP_NAME = disp_name


def _cache_frame(task):
    i, t, target_path = task
    target_path = Path(target_path)

    # Prefer existing enriched frame so previously cached fields are preserved.
    if target_path.exists():
        mesh = pv.read(target_path)
    else:
        mesh = load_mesh_at_time(_CACHE_READER, t)

    if not has_field(mesh, _CACHE_FIELD):
        mesh = compute_requested_field(
            mesh,
            E=_CACHE_E,
            nu=_CACHE_NU,
            disp_name=_CACHE_DISP_NAME,
            field=_CACHE_FIELD,
        )
        mesh.save(target_path)

    return i


def _init_range_worker(active_pvd, field, E, nu, disp_name):
    global _RANGE_READER, _RANGE_FIELD, _RANGE_E, _RANGE_NU, _RANGE_DISP_NAME
    os.environ["OMP_NUM_THREADS"] = "1"
    _RANGE_READER = pv.get_reader(str(active_pvd))
    _RANGE_FIELD = field
    _RANGE_E = E
    _RANGE_NU = nu
    _RANGE_DISP_NAME = disp_name


def _range_sample(task):
    i, t = task
    mesh = load_mesh_at_time(_RANGE_READER, t)

    if not has_field(mesh, _RANGE_FIELD):
        mesh = compute_requested_field(
            mesh,
            E=_RANGE_E,
            nu=_RANGE_NU,
            disp_name=_RANGE_DISP_NAME,
            field=_RANGE_FIELD,
        )

    vals = np.asarray(mesh[_RANGE_FIELD])
    vals = vals[np.isfinite(vals)]
    return i, vals


def add_scalar_bar_background(plotter):
    """White background box behind the scalar bar so it stays visible when
    overlapping the mesh."""
    scalar_bar = plotter.scalar_bar
    if hasattr(scalar_bar, "SetDrawBackground"):
        scalar_bar.SetDrawBackground(True)
    prop = scalar_bar.GetBackgroundProperty()
    prop.SetColor(1.0, 1.0, 1.0)
    prop.SetOpacity(0.85)


COLORBAR_RESERVE = 0.16


def compute_camera(mesh, zoom, window_size, reserve=COLORBAR_RESERVE):
    """Shared camera replicating view_xy + reset_camera + zoom, computed from
    mesh bounds only (no render window, safe to call before forking workers).
    The scene is shifted left and, if needed, pulled back so the right-hand
    strip reserved for the vertical colorbar stays clear of the mesh."""
    b = np.asarray(mesh.bounds, dtype=float)
    center = np.array(
        [(b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0, (b[4] + b[5]) / 2.0]
    )
    diag = float(
        np.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)
    )
    radius = diag / 2.0 if diag > 0.0 else 1.0
    width = float(b[1] - b[0])

    window_w, window_h = window_size
    aspect = window_w / window_h if window_h else 1.0
    tan_half_v = np.tan(np.radians(30.0) / 2.0)
    tan_half_h = tan_half_v * aspect

    distance = radius / np.sin(np.radians(30.0) / 2.0) / zoom
    shift = 0.0
    if width > 0.0 and reserve > 0.0:
        mesh_fraction = width / (2.0 * distance * tan_half_h)
        if mesh_fraction > 1.0 - reserve:
            distance *= mesh_fraction / (1.0 - reserve)
        shift = (reserve / 2.0) * 2.0 * distance * tan_half_h

    position = center + distance * np.array([0.0, 0.0, 1.0])
    position[0] += shift
    focal = center.copy()
    focal[0] += shift
    return (
        position.tolist(),
        focal.tolist(),
        [0.0, 1.0, 0.0],
    )


def has_headless_render_support():
    try:
        module = importlib.import_module("vtkmodules.vtkRenderingOpenGL2")
    except ImportError:
        return False
    return any(
        hasattr(module, name)
        for name in ("vtkOSOpenGLRenderWindow", "vtkEGLRenderWindow")
    )


def _print_render_backend_hint():
    print(
        "WARNING: headless node and no usable render backend.\n"
        "Install an OSMesa build of VTK in this environment:\n"
        "  pip uninstall -y vtk\n"
        "  pip install --extra-index-url https://wheels.vtk.org vtk-osmesa\n"
        "or make 'xvfb' available (e.g. 'module load xvfb')."
    )


_XVFB_DISPLAY = None


def ensure_render_backend():
    """Off-screen rendering on a headless node needs Xvfb or an OSMesa/EGL VTK
    build. Start Xvfb when possible and no display is available."""
    global _XVFB_DISPLAY

    if os.environ.get("DISPLAY") or has_headless_render_support():
        return

    if hasattr(pv, "start_xvfb"):
        try:
            pv.start_xvfb()
            print("No DISPLAY found: started Xvfb via pyvista.start_xvfb().")
            return
        except (ImportError, RuntimeError, OSError):
            pass

    if shutil.which("Xvfb") is None:
        _print_render_backend_hint()
        return

    try:
        from pyvirtualdisplay import Display
        _XVFB_DISPLAY = Display(visible=False, size=(1920, 1440))
        _XVFB_DISPLAY.start()
        print("No DISPLAY found: started Xvfb via pyvirtualdisplay.")
    except (ImportError, RuntimeError, OSError):
        _print_render_backend_hint()


_MOVIE_SETTINGS = None


def _init_movie_worker(settings):
    global _MOVIE_SETTINGS
    os.environ["OMP_NUM_THREADS"] = "1"
    _MOVIE_SETTINGS = settings


def _render_chunk(task):
    part_path, chunk_times = task
    s = _MOVIE_SETTINGS
    reader = pv.get_reader(s["active_pvd"])

    plotter = pv.Plotter(
        off_screen=True,
        window_size=list(s["window_size"]),
    )

    warped = None

    for i, t in enumerate(chunk_times):
        mesh = load_mesh_at_time(reader, t)

        if not has_field(mesh, s["field"]):
            mesh = compute_requested_field(
                mesh,
                E=s["E"],
                nu=s["nu"],
                disp_name=s["disp_name"],
                field=s["field"],
            )

        new_warped = mesh.warp_by_vector(s["disp_name"], factor=s["warp_factor"])

        if warped is None:
            warped = new_warped
            warped.point_data[s["field"]] = np.asarray(mesh[s["field"]]) / s["scale"]
            plotter.add_mesh(
                warped,
                scalars=s["field"],
                cmap=s["cmap"],
                clim=s["clim"],
                show_edges=False,
                scalar_bar_args=s["scalar_bar_args"],
            )
            add_scalar_bar_background(plotter)
            plotter.open_movie(part_path, framerate=s["fps"])
        else:
            warped.points = new_warped.points.copy()
            warped.point_data[s["field"]] = np.asarray(mesh[s["field"]]) / s["scale"]
            warped.GetPoints().Modified()
            warped.GetPointData().Modified()
            warped.Modified()

        plotter.add_text(
            f"Time = {t:.6e} s",
            name="time_text",
            font_size=20,
            font="times",
        )

        plotter.camera_position = s["camera_position"]
        plotter.write_frame()

    plotter.close()
    return len(chunk_times)


def _get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _concat_video(part_paths, movie_file):
    ffmpeg = _get_ffmpeg()
    list_file = movie_file.parent / f"{movie_file.stem}_concat.txt"

    with open(list_file, "w") as handle:
        handle.writelines(f"file '{part.resolve()}'\n" for part in part_paths)

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(movie_file),
        ],
        check=True,
    )

    list_file.unlink(missing_ok=True)


def parse_window_size(text):
    try:
        width_str, height_str = text.lower().split("x")
        width, height = int(width_str), int(height_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid window size '{text}'. Expected WIDTHxHEIGHT, e.g. 1920x1440."
        ) from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Window size must be positive.")

    return width, height


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a warped stress/strain animation from a PVD file. "
            "Requested derived fields are cached in an enriched PVD/VTU dataset "
            "so later movies can reuse them without recomputing gradients."
        )
    )

    parser.add_argument("--folder", type=str, default=DEFAULT_FOLDER)
    parser.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help="Movie output folder. Default: Stress/.",
    )
    parser.add_argument("--key", type=str, default=DEFAULT_KEY)
    parser.add_argument("--E", type=float, default=DEFAULT_E)
    parser.add_argument("--nu", type=float, default=DEFAULT_NU)
    parser.add_argument("--warp-factor", type=float, default=DEFAULT_WARP_FACTOR)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--disp-name", type=str, default=DEFAULT_DISP_NAME)

    parser.add_argument(
        "--field",
        type=str,
        choices=AVAILABLE_FIELDS,
        default=DEFAULT_FIELD,
    )

    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help="Global color clipping percentile. Default: 95. Use 100 for full range.",
    )
    parser.add_argument(
        "--range-stride",
        type=int,
        default=DEFAULT_RANGE_STRIDE,
        help=(
            "Use every Nth frame when estimating the global color range. "
            "Default: 10. Use 1 to inspect every frame."
        ),
    )

    parser.add_argument(
        "--window-size",
        type=parse_window_size,
        default=DEFAULT_WINDOW,
        help="Movie window size WIDTHxHEIGHT. Default: 1920x1440.",
    )

    parser.add_argument(
        "--zoom",
        type=float,
        default=DEFAULT_ZOOM,
        help="Camera zoom applied after reset_camera. Default: 1.4.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of parallel workers for cache building and range sampling. "
            "0 = auto (Slurm allocation or cpu count)."
        ),
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not create/use enriched cached VTU/PVD files.",
    )

    return parser.parse_args()


def load_mesh_at_time(reader, time_value):
    reader.set_active_time_value(time_value)
    mesh = reader.read()

    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()

    return mesh


def has_field(mesh, field):
    return field in mesh.point_data


def compute_requested_field(mesh, E, nu, disp_name, field):
    """
    Add only the requested derived field (plus no unnecessary outputs)
    unless it already exists.
    """
    field = canonical_field(field)
    mesh = mesh.copy()

    if has_field(mesh, field):
        return mesh

    if disp_name not in mesh.point_data:
        raise KeyError(
            f"Could not find displacement field '{disp_name}'.\n"
            f"Available point fields: {list(mesh.point_data.keys())}"
        )

    grad_mesh = mesh.compute_derivative(
        scalars=disp_name,
        gradient="disp_gradient",
        preference="point",
    )

    grad = grad_mesh["disp_gradient"].reshape((-1, 3, 3))

    dux_dx = grad[:, 0, 0]
    dux_dy = grad[:, 0, 1]
    duy_dx = grad[:, 1, 0]
    duy_dy = grad[:, 1, 1]

    eps_xx = dux_dx
    eps_yy = duy_dy
    eps_xy = 0.5 * (dux_dy + duy_dx)
    gamma_xy = dux_dy + duy_dx

    if field == "epsilon_xx":
        values = eps_xx

    elif field == "epsilon_yy":
        values = eps_yy

    elif field == "epsilon_xy":
        values = eps_xy

    elif field == "gamma_xy":
        values = gamma_xy

    else:
        C11 = E / (1.0 - nu**2)
        C12 = nu * E / (1.0 - nu**2)
        C66 = E / (2.0 * (1.0 + nu))

        sigma_xx = C11 * eps_xx + C12 * eps_yy
        sigma_yy = C12 * eps_xx + C11 * eps_yy
        sigma_xy = C66 * gamma_xy

        if field == "sigma_xx":
            values = sigma_xx

        elif field == "sigma_yy":
            values = sigma_yy

        elif field == "sigma_xy":
            values = sigma_xy

        elif field == "von_mises":
            values = np.sqrt(
                sigma_xx**2
                - sigma_xx * sigma_yy
                + sigma_yy**2
                + 3.0 * sigma_xy**2
            )

        else:
            raise ValueError(f"Unknown field: {field}")

    mesh.point_data[field] = values
    return mesh


def write_pvd(pvd_path, times, filenames):
    root = ET.Element(
        "VTKFile",
        type="Collection",
        version="0.1",
        byte_order="LittleEndian",
    )
    collection = ET.SubElement(root, "Collection")

    for t, filename in zip(times, filenames):
        ET.SubElement(
            collection,
            "DataSet",
            timestep=f"{float(t):.17g}",
            group="",
            part="0",
            file=filename,
        )

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    tree.write(pvd_path, encoding="utf-8", xml_declaration=True)


def compute_color_limits(values, field, percentile):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("No finite values found for color scaling.")

    if not (0.0 < percentile <= 100.0):
        raise ValueError("--percentile must be in (0, 100].")

    field = canonical_field(field)

    if field == "von_mises":
        return 0.0, np.percentile(values, percentile)

    [limlow, limhi] = np.percentile(values, [100 - percentile, percentile] )
    print(limlow, limhi)
    return 0.05e6, 0.25e6


def cache_paths(folder, key):
    cache_dir = folder / f"{key}_enriched"
    cache_pvd = folder / f"{key}_enriched.pvd"
    return cache_dir, cache_pvd


def build_or_extend_cache(
    source_reader,
    source_times,
    source_pvd,
    folder,
    key,
    E,
    nu,
    disp_name,
    field,
    workers,
):
    """
    Creates/extends:
        <folder>/<key>_enriched.pvd
        <folder>/<key>_enriched/frame_000000.vtu ...

    If the enriched dataset already exists but lacks the requested field,
    the cached VTUs are updated in place and retain previously cached fields.
    """
    field = canonical_field(field)
    cache_dir, cache_pvd = cache_paths(folder, key)
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing_reader = None
    existing_times = None

    if cache_pvd.exists():
        try:
            existing_reader = pv.get_reader(str(cache_pvd))
            existing_times = np.asarray(existing_reader.time_values)
            if len(existing_times) != len(source_times):
                existing_reader = None
        except Exception:
            existing_reader = None

    # Fast check: does cached first frame already have this field?
    if existing_reader is not None:
        first_cached = load_mesh_at_time(existing_reader, existing_times[0])
        if has_field(first_cached, field):
            print(f"Cached field '{field}' already exists.")
            return cache_pvd

    print(f"Caching derived field '{field}' for future reuse...")

    filenames = [
        f"{cache_dir.name}/frame_{i:06d}.vtu" for i in range(len(source_times))
    ]

    if workers > 1:
        tasks = [
            (i, t, str(cache_dir / filename))
            for i, (t, filename) in enumerate(zip(source_times, filenames))
        ]

        done = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_cache_worker,
            initargs=(str(source_pvd), field, E, nu, disp_name),
        ) as pool:
            futures = {pool.submit(_cache_frame, task) for task in tasks}
            for future in as_completed(futures):
                future.result()
                done += 1
                print(
                    f"\rCaching: {done}/{len(tasks)}",
                    end="",
                    flush=True,
                )

        print()
    else:
        _init_cache_worker(str(source_pvd), field, E, nu, disp_name)

        for i, t in enumerate(source_times):
            _cache_frame((i, t, str(cache_dir / filenames[i])))

            if i % 10 == 0 or i == len(source_times) - 1:
                print(
                    f"\rCaching: {i + 1}/{len(source_times)}",
                    end="",
                    flush=True,
                )

        print()

    write_pvd(cache_pvd, source_times, filenames)
    print(f"Enriched cache: {cache_pvd}")

    return cache_pvd


def main():
    args = parse_args()

    if args.range_stride < 1:
        raise ValueError("--range-stride must be >= 1.")

    ensure_render_backend()

    field = canonical_field(args.field)

    folder = Path(args.folder)
    output_folder = Path(args.output_folder) if args.output_folder else folder
    output_folder.mkdir(parents=True, exist_ok=True)

    source_pvd = folder / f"{args.key}.pvd"

    if not source_pvd.exists():
        raise FileNotFoundError(f"PVD file does not exist:\n{source_pvd}")

    movie_file = output_folder / f"{args.key}_{field}.mp4"

    n_workers = args.workers if args.workers > 0 else get_requested_cpus()
    n_workers = max(1, n_workers)

    print("=" * 68)
    print(f"Input         : {source_pvd}")
    print(f"Output        : {movie_file}")
    print(f"Field         : {field}")
    print(f"E             : {args.E:.6e} Pa")
    print(f"nu            : {args.nu}")
    print(f"Warp factor   : {args.warp_factor}")
    print(f"FPS           : {args.fps}")
    print(f"Disp field    : {args.disp_name}")
    print(f"Percentile    : {args.percentile:.2f}")
    print(f"Range stride  : {args.range_stride}")
    print(f"Cache enabled : {not args.no_cache}")
    print(f"Window size   : {args.window_size[0]}x{args.window_size[1]}")
    print(f"Zoom          : {args.zoom}")
    print(f"Workers       : {n_workers}")
    print("=" * 68)

    mathtext_ok = setup_mathtext_font()
    if not mathtext_ok:
        print("matplotlib not available: using plain-text titles (no mathtext).")
    title = display_name(field, mathtext_ok) + unit_suffix(field)
    scale = display_scale(field)

    source_reader = pv.get_reader(str(source_pvd))
    source_times = np.asarray(source_reader.time_values)

    print(f"Number of timesteps: {len(source_times)}")
    print(f"Time range: {source_times[0]:.6e} -> {source_times[-1]:.6e} s")

    # --------------------------------------------------------
    # Determine whether field already exists in original PVD.
    # --------------------------------------------------------

    first_mesh = load_mesh_at_time(source_reader, source_times[0])
    field_in_original = has_field(first_mesh, field)

    if field_in_original:
        print(f"\nField '{field}' already exists in original PVD. No recomputation needed.")
        active_pvd = source_pvd

    elif args.no_cache:
        print(f"\nField '{field}' not found. It will be computed on the fly.")
        active_pvd = source_pvd

    else:
        cache_pvd = build_or_extend_cache(
            source_reader=source_reader,
            source_times=source_times,
            source_pvd=source_pvd,
            folder=folder,
            key=args.key,
            E=args.E,
            nu=args.nu,
            disp_name=args.disp_name,
            field=field,
            workers=n_workers,
        )
        active_pvd = cache_pvd

    reader = pv.get_reader(str(active_pvd))
    time_values = np.asarray(reader.time_values)

    # --------------------------------------------------------
    # RANGE PASS — sampled every range_stride frames
    # --------------------------------------------------------

    print(f"\nEstimating global color range using every {args.range_stride} frame(s)...")

    sampled_values = []

    indices = list(range(0, len(time_values), args.range_stride))
    if indices[-1] != len(time_values) - 1:
        indices.append(len(time_values) - 1)

    tasks = [(i, time_values[i]) for i in indices]

    if n_workers > 1:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_range_worker,
            initargs=(str(active_pvd), field, args.E, args.nu, args.disp_name),
        ) as pool:
            for j, (i, vals) in enumerate(pool.map(_range_sample, tasks)):
                if vals.size:
                    sampled_values.append(vals)

                if j % 10 == 0 or j == len(tasks) - 1:
                    print(
                        f"\rRange samples: {j + 1}/{len(tasks)}",
                        end="",
                        flush=True,
                    )

        print()
    else:
        _init_range_worker(str(active_pvd), field, args.E, args.nu, args.disp_name)

        for j, i in enumerate(indices):
            i, vals = _range_sample(tasks[j])

            if vals.size:
                sampled_values.append(vals)

            if j % 10 == 0 or j == len(indices) - 1:
                print(
                    f"\rRange samples: {j + 1}/{len(indices)}",
                    end="",
                    flush=True,
                )

        print()

    if not sampled_values:
        raise ValueError(f"No finite data found for field '{field}'.")

    sampled_values = np.concatenate(sampled_values)

    clim_min, clim_max = compute_color_limits(
        sampled_values,
        field=field,
        percentile=args.percentile,
    )

    print(f"Color limits: {clim_min / scale:.6g} -> {clim_max / scale:.6g}{unit_suffix(field)}")

    # --------------------------------------------------------
    # INITIAL FRAME + SHARED RENDER SETTINGS
    # --------------------------------------------------------

    mesh = load_mesh_at_time(reader, time_values[0])

    if not has_field(mesh, field):
        mesh = compute_requested_field(
            mesh,
            E=args.E,
            nu=args.nu,
            disp_name=args.disp_name,
            field=field,
        )

    warped = mesh.warp_by_vector(
        args.disp_name,
        factor=args.warp_factor,
    )

    warped.point_data[field] = np.asarray(mesh[field]) / scale

    camera_position = compute_camera(warped, args.zoom, args.window_size)

    scalar_bar_args = {
        "title": title,
        "vertical": True,
        "position_x": 0.87,
        "position_y": 0.25,
        "width": 0.07,
        "height": 0.5,
        "title_font_size": 36,
        "label_font_size": 28,
        "font_family": "times",
    }

    # --------------------------------------------------------
    # MOVIE
    # --------------------------------------------------------

    print("\nWriting movie...")

    if n_workers > 1:
        settings = {
            "active_pvd": str(active_pvd),
            "field": field,
            "disp_name": args.disp_name,
            "E": args.E,
            "nu": args.nu,
            "warp_factor": args.warp_factor,
            "scale": scale,
            "cmap": "RdBu_r",
            "clim": [clim_min / scale, clim_max / scale],
            "scalar_bar_args": scalar_bar_args,
            "window_size": args.window_size,
            "fps": args.fps,
            "camera_position": camera_position,
        }

        n_chunks = min(n_workers, len(time_values))
        bounds = np.linspace(0, len(time_values), n_chunks + 1).astype(int)
        parts_dir = Path(
            tempfile.mkdtemp(prefix="stress_movie_parts_", dir=str(output_folder))
        )
        tasks = [
            (str(parts_dir / f"part_{k:04d}.mp4"), time_values[bounds[k]:bounds[k + 1]])
            for k in range(n_chunks)
        ]

        try:
            done = 0
            mp_context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=n_chunks,
                mp_context=mp_context,
                initializer=_init_movie_worker,
                initargs=(settings,),
            ) as pool:
                futures = {pool.submit(_render_chunk, task) for task in tasks}
                for future in as_completed(futures):
                    future.result()
                    done += 1
                    print(
                        f"\rRendered chunks: {done}/{len(tasks)}",
                        end="",
                        flush=True,
                    )

            print()
            _concat_video([Path(task[0]) for task in tasks], movie_file)
        finally:
            shutil.rmtree(parts_dir, ignore_errors=True)
    else:
        plotter = pv.Plotter(
            off_screen=True,
            window_size=list(args.window_size),
        )

        plotter.add_mesh(
            warped,
            scalars=field,
            cmap="RdBu_r",
            clim=[clim_min / scale, clim_max / scale],
            show_edges=False,
            scalar_bar_args=scalar_bar_args,
        )

        add_scalar_bar_background(plotter)

        plotter.add_text(
            f"Time = {time_values[0]:.6e} s",
            name="time_text",
            font_size=28,
            font="times",
        )

        plotter.camera_position = camera_position

        plotter.open_movie(str(movie_file), framerate=args.fps)
        plotter.write_frame()

        for i, t in enumerate(time_values[1:], start=1):
            mesh = load_mesh_at_time(reader, t)

            if not has_field(mesh, field):
                mesh = compute_requested_field(
                    mesh,
                    E=args.E,
                    nu=args.nu,
                    disp_name=args.disp_name,
                    field=field,
                )

            warped_new = mesh.warp_by_vector(
                args.disp_name,
                factor=args.warp_factor,
            )

            warped.points = warped_new.points.copy()
            warped.point_data[field] = np.asarray(mesh[field]) / scale

            warped.GetPoints().Modified()
            warped.GetPointData().Modified()
            warped.Modified()

            plotter.add_text(
                f"time = {t:.6e} s",
                name="time_text",
                font_size=20,
                font="times",
            )

            plotter.camera_position = camera_position
            plotter.write_frame()

            if i % 10 == 0 or i == len(time_values) - 1:
                print(
                    f"\rMovie frames: {i + 1}/{len(time_values)}",
                    end="",
                    flush=True,
                )

        print()
        plotter.close()

    print("\nDone.")
    print(f"Movie saved to: {movie_file}")


if __name__ == "__main__":
    main()
