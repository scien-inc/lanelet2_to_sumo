from __future__ import annotations

import xml.etree.ElementTree as ET

from ll2sumo.geometry import distance_2d, polyline_length
from ll2sumo.model import Point3D


def id_sort_key(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def shape_string(points: tuple[Point3D, ...]) -> str:
    return " ".join(f"{point.x:.3f},{point.y:.3f},{point.z:.3f}" for point in points)


def parse_shape_points(shape: str) -> tuple[Point3D, ...]:
    points: list[Point3D] = []
    for token in shape.split():
        parts = token.split(",")
        if len(parts) < 2:
            continue
        points.append(
            Point3D(
                x=float(parts[0]),
                y=float(parts[1]),
                z=float(parts[2]) if len(parts) >= 3 else 0.0,
            )
        )
    return tuple(points)


def polyline_length_2d(points: tuple[Point3D, ...] | list[Point3D]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(distance_2d(points[index], points[index + 1]) for index in range(len(points) - 1))


def dedupe_consecutive_shape_points(
    points: tuple[Point3D, ...] | list[Point3D],
    min_distance_m: float = 0.01,
) -> tuple[Point3D, ...]:
    deduped: list[Point3D] = []
    for point in points:
        if deduped and distance_2d(deduped[-1], point) < min_distance_m:
            continue
        deduped.append(point)
    return tuple(deduped)


def usable_connection_shape(
    points: tuple[Point3D, ...] | list[Point3D],
    min_length_m: float = 0.01,
) -> tuple[Point3D, ...] | None:
    deduped = dedupe_consecutive_shape_points(points, min_length_m)
    if len(deduped) < 2:
        return None
    if polyline_length(deduped) < min_length_m:
        return None
    return deduped


def is_internal_edge(edge_element: ET.Element) -> bool:
    edge_id = edge_element.attrib.get("id", "")
    return edge_id.startswith(":") or edge_element.attrib.get("function") == "internal"


def net_lane_id(edge_id: str, lane_index: str) -> str:
    return f"{edge_id}_{lane_index}"
