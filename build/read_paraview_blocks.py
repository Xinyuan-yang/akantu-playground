#!/usr/bin/env python3
"""Summarize Akantu/ParaView VTU or PVTU output by connected block.

This reader uses only the Python standard library. It understands the inline
base64 binary VTK XML files written in this project and reports per-block
statistics for point fields such as displacement and cell fields such as stress.
"""

from __future__ import annotations

import argparse
import base64
import csv
import math
import re
import struct
import xml.etree.ElementTree as ET
from array import array
from collections import defaultdict
from pathlib import Path
from statistics import fmean


VTK_TYPE_CODES = {
    "Float64": "d",
    "Float32": "f",
    "Int64": "q",
    "UInt64": "Q",
    "Int32": "i",
    "UInt32": "I",
    "Int16": "h",
    "UInt16": "H",
    "Int8": "b",
    "UInt8": "B",
}


def _strip_namespace(tag):
    return tag.rsplit("}", 1)[-1]


def _children(parent, tag):
    return [child for child in parent if _strip_namespace(child.tag) == tag]


def _first(parent, tag):
    matches = _children(parent, tag)
    if not matches:
        raise ValueError(f"missing <{tag}> in {parent.tag}")
    return matches[0]


def _decode_data_array(node):
    vtk_type = node.attrib.get("type")
    if vtk_type not in VTK_TYPE_CODES:
        raise ValueError(f"unsupported VTK data type {vtk_type!r}")
    if node.attrib.get("format") != "binary":
        raise ValueError("this reader expects binary DataArray entries")

    text = "".join((node.text or "").split())
    raw = base64.b64decode(text)
    if len(raw) < 4:
        return []

    payload_size = struct.unpack("<I", raw[:4])[0]
    payload = raw[4 : 4 + payload_size]

    values = array(VTK_TYPE_CODES[vtk_type])
    values.frombytes(payload)
    if values.itemsize > 1:
        values.byteswap() if struct.pack("=H", 1) == struct.pack(">H", 1) else None
    return values.tolist()


def _component_count(node):
    return int(node.attrib.get("NumberOfComponents", "1"))


def _named_arrays(parent):
    arrays = {}
    components = {}
    for node in _children(parent, "DataArray"):
        name = node.attrib.get("Name")
        if not name:
            continue
        arrays[name] = _decode_data_array(node)
        components[name] = _component_count(node)
    return arrays, components


def read_vtu(path):
    """Read one .vtu piece and return points, cells, point_data, cell_data."""
    path = Path(path)
    root = ET.parse(path).getroot()
    piece = _first(_first(root, "UnstructuredGrid"), "Piece")

    points_node = _first(_first(piece, "Points"), "DataArray")
    points_flat = _decode_data_array(points_node)
    point_components = _component_count(points_node)
    points = [
        tuple(points_flat[i : i + point_components])
        for i in range(0, len(points_flat), point_components)
    ]

    cells_node = _first(piece, "Cells")
    cell_arrays = {
        node.attrib["Name"]: _decode_data_array(node)
        for node in _children(cells_node, "DataArray")
        if "Name" in node.attrib
    }
    connectivity = [int(v) for v in cell_arrays["connectivity"]]
    offsets = [int(v) for v in cell_arrays["offsets"]]
    cells = []
    start = 0
    for end in offsets:
        cells.append([int(v) for v in connectivity[start:end]])
        start = end

    point_data, point_components_by_name = _named_arrays(_first(piece, "PointData"))
    cell_data, cell_components_by_name = _named_arrays(_first(piece, "CellData"))

    return {
        "points": points,
        "cells": cells,
        "point_data": point_data,
        "cell_data": cell_data,
        "point_components": point_components_by_name,
        "cell_components": cell_components_by_name,
    }


def read_pvtu(path):
    """Read a .pvtu file by loading and merging its referenced .vtu pieces."""
    path = Path(path)
    root = ET.parse(path).getroot()
    grid = _first(root, "PUnstructuredGrid")
    sources = [piece.attrib["Source"] for piece in _children(grid, "Piece")]
    if not sources:
        raise ValueError(f"{path} does not reference any VTU pieces")

    merged = None
    point_offset = 0
    for source in sources:
        part = read_vtu(path.parent / source)
        if merged is None:
            merged = {
                "points": [],
                "cells": [],
                "point_data": {name: [] for name in part["point_data"]},
                "cell_data": {name: [] for name in part["cell_data"]},
                "point_components": part["point_components"],
                "cell_components": part["cell_components"],
            }

        merged["points"].extend(part["points"])
        merged["cells"].extend(
            [[point + point_offset for point in cell] for cell in part["cells"]]
        )
        for name, values in part["point_data"].items():
            merged["point_data"].setdefault(name, []).extend(values)
        for name, values in part["cell_data"].items():
            merged["cell_data"].setdefault(name, []).extend(values)
        point_offset += len(part["points"])

    return merged


def read_vtk_xml(path):
    path = Path(path)
    if path.suffix == ".pvtu":
        return read_pvtu(path)
    if path.suffix == ".vtu":
        return read_vtu(path)
    raise ValueError(f"expected .pvtu or .vtu file, got {path}")


class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def connected_cell_blocks(cells):
    """Return cell-index lists for components connected by shared point ids."""
    dsu = DisjointSet(len(cells))
    first_cell_for_point = {}
    for cell_id, cell in enumerate(cells):
        for point_id in cell:
            previous = first_cell_for_point.setdefault(point_id, cell_id)
            dsu.union(previous, cell_id)

    groups = defaultdict(list)
    for cell_id in range(len(cells)):
        groups[dsu.find(cell_id)].append(cell_id)

    return sorted(groups.values(), key=lambda ids: (min(ids), len(ids)))


def _chunks(values, components):
    for i in range(0, len(values), components):
        yield values[i : i + components]


def _field_stats(rows):
    if not rows:
        return {}
    components = len(rows[0])
    stats = {}
    for c in range(components):
        values = [row[c] for row in rows]
        stats[f"c{c}_mean"] = fmean(values)
        stats[f"c{c}_min"] = min(values)
        stats[f"c{c}_max"] = max(values)
    if components > 1:
        magnitudes = [math.sqrt(sum(v * v for v in row)) for row in rows]
        stats["norm_mean"] = fmean(magnitudes)
        stats["norm_min"] = min(magnitudes)
        stats["norm_max"] = max(magnitudes)
    return stats


def summarize_by_block(dataset, fields=None):
    cells = dataset["cells"]
    blocks = connected_cell_blocks(cells)
    point_to_cells = defaultdict(list)
    for cell_id, cell in enumerate(cells):
        for point_id in cell:
            point_to_cells[point_id].append(cell_id)

    cell_to_block = {}
    for block_id, cell_ids in enumerate(blocks):
        for cell_id in cell_ids:
            cell_to_block[cell_id] = block_id

    block_points = [set() for _ in blocks]
    for point_id, cell_ids in point_to_cells.items():
        for cell_id in cell_ids:
            block_points[cell_to_block[cell_id]].add(point_id)

    requested = set(fields or [])
    rows = []
    for block_id, cell_ids in enumerate(blocks):
        point_ids = sorted(block_points[block_id])
        block_row = {
            "block": block_id,
            "n_cells": len(cell_ids),
            "n_points": len(point_ids),
        }

        for name, values in dataset["point_data"].items():
            if requested and name not in requested:
                continue
            components = dataset["point_components"][name]
            rows_for_field = list(_chunks(values, components))
            stats = _field_stats([rows_for_field[i] for i in point_ids])
            for key, value in stats.items():
                block_row[f"{name}.{key}"] = value

        for name, values in dataset["cell_data"].items():
            if requested and name not in requested:
                continue
            components = dataset["cell_components"][name]
            rows_for_field = list(_chunks(values, components))
            stats = _field_stats([rows_for_field[i] for i in cell_ids])
            for key, value in stats.items():
                block_row[f"{name}.{key}"] = value

        rows.append(block_row)

    return rows


def latest_file(paraview_dir, pattern):
    files = sorted(Path(paraview_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files match {Path(paraview_dir) / pattern}")

    def step_number(path):
        match = re.search(r"_(\d+)\.p?vtu$", path.name)
        return int(match.group(1)) if match else -1

    return max(files, key=step_number)


def write_csv(rows, path):
    keys = sorted({key for row in rows for key in row})
    first = ["block", "n_cells", "n_points"]
    fieldnames = first + [key for key in keys if key not in first]
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows, max_columns=12):
    if not rows:
        print("No rows.")
        return
    preferred = [
        "block",
        "n_cells",
        "n_points",
        "displacement.c0_mean",
        "displacement.c1_mean",
        "displacement.c2_mean",
        "displacement.norm_mean",
        "stress.c0_mean",
        "stress.c1_mean",
        "stress.c2_mean",
        "stress.c3_mean",
    ]
    columns = [column for column in preferred if column in rows[0]]
    columns += [key for key in rows[0] if key not in columns]
    columns = columns[:max_columns]
    widths = {
        column: max(len(column), *(len(f"{row.get(column, ''):.6g}") if isinstance(row.get(column), float) else len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    print("  ".join(column.rjust(widths[column]) for column in columns))
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.6g}".rjust(widths[column]) if isinstance(value, float) else str(value).rjust(widths[column]))
        print("  ".join(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", help="VTU/PVTU file to read")
    parser.add_argument(
        "--paraview-dir",
        default=Path(__file__).resolve().parent / "paraview",
        type=Path,
        help="directory used when --latest is selected",
    )
    parser.add_argument(
        "--pattern",
        default="bloc_friction_test1_*.pvtu",
        help="glob pattern used with --latest",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="read the highest numbered file matching --pattern",
    )
    parser.add_argument(
        "--field",
        action="append",
        dest="fields",
        help="field to include; repeatable. Defaults to all fields.",
    )
    parser.add_argument("--csv", type=Path, help="optional CSV output path")
    args = parser.parse_args()

    path = latest_file(args.paraview_dir, args.pattern) if args.latest else args.file
    if path is None:
        path = latest_file(args.paraview_dir, args.pattern)
    path = Path(path)

    dataset = read_vtk_xml(path)
    rows = summarize_by_block(dataset, args.fields)
    print(f"Read {path}")
    print(
        f"{len(dataset['points'])} points, {len(dataset['cells'])} cells, "
        f"{len(rows)} connected block(s)"
    )
    print("Point fields:", ", ".join(dataset["point_data"]))
    print("Cell fields:", ", ".join(dataset["cell_data"]))
    print_table(rows)

    if args.csv:
        write_csv(rows, args.csv)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
