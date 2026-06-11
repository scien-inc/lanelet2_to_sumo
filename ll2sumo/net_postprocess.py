from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ll2sumo.geometry import distance_2d, heading_deg, polyline_length
from ll2sumo.geometry import first_nonzero_segment as _first_nonzero_segment
from ll2sumo.geometry import last_nonzero_segment as _last_nonzero_segment
from ll2sumo.geometry import point_along_direction as _point_along_direction
from ll2sumo.model import Point3D
from ll2sumo.sumo_xml import id_sort_key as _sort_key
from ll2sumo.sumo_xml import is_internal_edge as _is_internal_edge
from ll2sumo.sumo_xml import net_lane_id as _net_lane_id
from ll2sumo.sumo_xml import parse_shape_points as _parse_shape_points
from ll2sumo.sumo_xml import polyline_length_2d as _polyline_length_2d
from ll2sumo.sumo_xml import shape_string as _shape_string
from ll2sumo.sumo_xml import usable_connection_shape as _usable_connection_shape

MIN_SUMO_LANE_LENGTH_M, DEGENERATE_INTERNAL_LANE_XY_LENGTH_M = 0.1, 0.01
REPAIRED_INTERNAL_LANE_FALLBACK_LENGTH_M, MAX_INTERNAL_CONNECTION_ALIGN_EXAMPLES = 0.25, 20
JP_TLS_GREEN_TIME_S, JP_TLS_RIGHT_TURN_TIME_S, JP_TLS_YELLOW_TIME_S, JP_TLS_ALL_RED_TIME_S = 35, 8, 3, 2
JP_TLS_AXIS_CLUSTER_THRESHOLD_DEG = 35.0

@dataclass(frozen=True)
class TLSLinkInfo:
    index: int
    incoming_heading_deg: float
    directions: tuple[str, ...]

    @property
    def axis_deg(self) -> float:
        return self.incoming_heading_deg % 180.0

    @property
    def has_right_turn(self) -> bool:
        return any(direction in {"r", "R"} for direction in self.directions)

    @property
    def has_non_right_turn(self) -> bool:
        return any(direction not in {"r", "R"} for direction in self.directions)

def _summarize_net_tls(net_path: Path) -> dict[str, object]:
    root = ET.parse(net_path).getroot()
    tls_ids = sorted((element.attrib["id"] for element in root.findall("tlLogic")), key=_sort_key)
    tls_id_set = set(tls_ids)
    vehicle_tls_link_count = sum(
        1
        for connection_element in root.findall("connection")
        if connection_element.attrib.get("tl") in tls_id_set
        and not connection_element.attrib.get("from", "").startswith(":")
    )
    signalized_junction_count = sum(
        1
        for junction_element in root.findall("junction")
        if junction_element.attrib.get("type", "").startswith("traffic_light")
    )
    return {
        "tls_cluster_count": len(tls_ids),
        "vehicle_tls_link_count": vehicle_tls_link_count,
        "sumo_signalized_junction_count": signalized_junction_count,
        "sumo_tls_ids": tls_ids,
    }

def _replacement_internal_shape(
    from_points: tuple[Point3D, ...],
    to_points: tuple[Point3D, ...],
) -> tuple[Point3D, Point3D] | None:
    if not from_points or not to_points:
        return None
    start = from_points[-1]
    end = to_points[0]
    if distance_2d(start, end) >= DEGENERATE_INTERNAL_LANE_XY_LENGTH_M:
        return start, end

    downstream_segment = _first_nonzero_segment(to_points, DEGENERATE_INTERNAL_LANE_XY_LENGTH_M)
    if downstream_segment is not None:
        fallback_end = _point_along_direction(
            start,
            downstream_segment[0],
            downstream_segment[1],
            REPAIRED_INTERNAL_LANE_FALLBACK_LENGTH_M,
        )
        if fallback_end is not None:
            return start, fallback_end

    incoming_segment = _last_nonzero_segment(from_points, DEGENERATE_INTERNAL_LANE_XY_LENGTH_M)
    if incoming_segment is not None:
        fallback_end = _point_along_direction(
            start,
            incoming_segment[0],
            incoming_segment[1],
            REPAIRED_INTERNAL_LANE_FALLBACK_LENGTH_M,
        )
        if fallback_end is not None:
            return start, fallback_end
    return None

def _interpolate_point(start: Point3D, end: Point3D, ratio: float) -> Point3D:
    return Point3D(
        x=start.x + (end.x - start.x) * ratio,
        y=start.y + (end.y - start.y) * ratio,
        z=start.z + (end.z - start.z) * ratio,
    )

def _project_point_on_polyline(points: tuple[Point3D, ...], point: Point3D) -> tuple[float, float, Point3D] | None:
    if len(points) < 2:
        return None
    best_distance = math.inf
    best_along = 0.0
    best_point: Point3D | None = None
    along_before = 0.0
    for start, end in zip(points, points[1:]):
        segment_length = distance_2d(start, end)
        if segment_length <= 0.0:
            continue
        dx = end.x - start.x
        dy = end.y - start.y
        ratio = ((point.x - start.x) * dx + (point.y - start.y) * dy) / (segment_length * segment_length)
        ratio = max(0.0, min(1.0, ratio))
        projected = _interpolate_point(start, end, ratio)
        projected_distance = distance_2d(point, projected)
        projected_along = along_before + segment_length * ratio
        if projected_distance < best_distance:
            best_distance = projected_distance
            best_along = projected_along
            best_point = projected
        along_before += segment_length
    if best_point is None:
        return None
    return best_distance, best_along, best_point

def _slice_polyline_between(
    points: tuple[Point3D, ...],
    start_along: float,
    start_point: Point3D,
    end_along: float,
    end_point: Point3D,
) -> tuple[Point3D, ...] | None:
    if end_along <= start_along + DEGENERATE_INTERNAL_LANE_XY_LENGTH_M:
        return None
    sliced: list[Point3D] = [start_point]
    along = 0.0
    for start, end in zip(points, points[1:]):
        segment_length = distance_2d(start, end)
        if segment_length <= 0.0:
            continue
        along += segment_length
        if start_along + DEGENERATE_INTERNAL_LANE_XY_LENGTH_M < along < end_along - DEGENERATE_INTERNAL_LANE_XY_LENGTH_M:
            sliced.append(end)
    sliced.append(end_point)
    return _usable_connection_shape(sliced)

def _aligned_internal_connection_shape(
    connection_points: tuple[Point3D, ...],
    from_points: tuple[Point3D, ...],
    to_points: tuple[Point3D, ...],
) -> tuple[tuple[Point3D, ...], str] | None:
    if not from_points or not to_points:
        return None
    start = from_points[-1]
    end = to_points[0]
    if distance_2d(start, end) < REPAIRED_INTERNAL_LANE_FALLBACK_LENGTH_M:
        downstream_segment = _first_nonzero_segment(to_points, DEGENERATE_INTERNAL_LANE_XY_LENGTH_M)
        if downstream_segment is not None:
            fallback_end = _point_along_direction(
                start,
                downstream_segment[0],
                downstream_segment[1],
                REPAIRED_INTERNAL_LANE_FALLBACK_LENGTH_M,
            )
            if fallback_end is not None:
                fallback_shape = _usable_connection_shape((start, fallback_end))
                if fallback_shape is not None:
                    return fallback_shape, "tangent_fallback"
        if distance_2d(start, end) >= DEGENERATE_INTERNAL_LANE_XY_LENGTH_M:
            fallback_end = _point_along_direction(
                start,
                start,
                end,
                REPAIRED_INTERNAL_LANE_FALLBACK_LENGTH_M,
            )
            if fallback_end is not None:
                fallback_shape = _usable_connection_shape((start, fallback_end))
                if fallback_shape is not None:
                    return fallback_shape, "direct_fallback"

    if len(connection_points) >= 2:
        start_projection = _project_point_on_polyline(connection_points, start)
        end_projection = _project_point_on_polyline(connection_points, end)
        if start_projection is not None and end_projection is not None:
            _, start_along, _ = start_projection
            _, end_along, _ = end_projection
            sliced_shape = _slice_polyline_between(connection_points, start_along, start, end_along, end)
            if sliced_shape is not None:
                return sliced_shape, "trimmed"

    direct_shape = _usable_connection_shape((start, end))
    if direct_shape is not None:
        return direct_shape, "direct_fallback"

    fallback_shape = _replacement_internal_shape(from_points, to_points)
    if fallback_shape is None:
        return None
    usable_fallback_shape = _usable_connection_shape(fallback_shape)
    if usable_fallback_shape is None:
        return None
    return usable_fallback_shape, "tangent_fallback"

def _audit_degenerate_internal_lane_shapes(net_path: str | Path) -> dict[str, object]:
    root = ET.parse(net_path).getroot()
    scanned_count = 0
    degenerate_count = 0
    examples: list[dict[str, object]] = []
    for edge_element in root.findall("edge"):
        if not _is_internal_edge(edge_element):
            continue
        for lane_element in edge_element.findall("lane"):
            scanned_count += 1
            points = _parse_shape_points(lane_element.attrib.get("shape", ""))
            xy_length = _polyline_length_2d(points)
            if xy_length >= DEGENERATE_INTERNAL_LANE_XY_LENGTH_M:
                continue
            degenerate_count += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "lane_id": lane_element.attrib.get("id"),
                        "xy_length_m": round(xy_length, 6),
                        "length": lane_element.attrib.get("length"),
                        "shape": lane_element.attrib.get("shape"),
                    }
                )
    return {
        "scanned_internal_lane_count": scanned_count,
        "degenerate_internal_lane_count": degenerate_count,
        "examples": examples,
    }

def _sync_internal_lane_shapes_from_connection_shapes(net_path: str | Path) -> dict[str, object]:
    path = Path(net_path)
    tree = ET.parse(path)
    root = tree.getroot()
    lanes_by_id: dict[str, ET.Element] = {
        lane_element.attrib["id"]: lane_element
        for edge_element in root.findall("edge")
        for lane_element in edge_element.findall("lane")
        if "id" in lane_element.attrib
    }
    scanned_via_count = 0
    synced_count = 0
    missing_shape_count = 0
    unusable_shape_count = 0
    examples: list[dict[str, object]] = []

    for connection_element in root.findall("connection"):
        via_lane_id = connection_element.attrib.get("via")
        if not via_lane_id:
            continue
        lane_element = lanes_by_id.get(via_lane_id)
        if lane_element is None:
            continue
        scanned_via_count += 1
        connection_shape = connection_element.attrib.get("shape")
        if not connection_shape:
            missing_shape_count += 1
            continue
        connection_points = _parse_shape_points(connection_shape)
        usable_shape = _usable_connection_shape(connection_points)
        if usable_shape is None:
            unusable_shape_count += 1
            continue
        replacement_shape = _shape_string(usable_shape)
        if lane_element.attrib.get("shape") == replacement_shape:
            continue
        lane_element.set("shape", replacement_shape)
        synced_count += 1
        if len(examples) < 20:
            examples.append(
                {
                    "lane_id": via_lane_id,
                    "from": connection_element.attrib.get("from"),
                    "to": connection_element.attrib.get("to"),
                    "xy_length_m": round(_polyline_length_2d(usable_shape), 6),
                }
            )

    if synced_count:
        ET.indent(tree, space="    ")
        tree.write(path, encoding="utf-8", xml_declaration=True)

    return {
        "scanned_via_connection_count": scanned_via_count,
        "synced_internal_lane_count": synced_count,
        "missing_connection_shape_count": missing_shape_count,
        "unusable_connection_shape_count": unusable_shape_count,
        "examples": examples,
    }

def _align_internal_connection_shapes_to_net_lanes(net_path: str | Path) -> dict[str, object]:
    path = Path(net_path)
    tree = ET.parse(path)
    root = tree.getroot()
    lanes_by_id: dict[str, ET.Element] = {
        lane_element.attrib["id"]: lane_element
        for edge_element in root.findall("edge")
        for lane_element in edge_element.findall("lane")
        if "id" in lane_element.attrib
    }
    scanned_count = 0
    aligned_count = 0
    trimmed_count = 0
    direct_fallback_count = 0
    tangent_fallback_count = 0
    unrepaired_count = 0
    max_endpoint_gap_before = 0.0
    max_endpoint_gap_after = 0.0
    examples: list[dict[str, object]] = []

    for connection_element in root.findall("connection"):
        from_edge_id = connection_element.attrib.get("from")
        to_edge_id = connection_element.attrib.get("to")
        from_lane_index = connection_element.attrib.get("fromLane")
        to_lane_index = connection_element.attrib.get("toLane")
        if from_edge_id is None or to_edge_id is None or from_lane_index is None or to_lane_index is None:
            continue
        from_lane_element = lanes_by_id.get(_net_lane_id(from_edge_id, from_lane_index))
        to_lane_element = lanes_by_id.get(_net_lane_id(to_edge_id, to_lane_index))
        if from_lane_element is None or to_lane_element is None:
            continue
        from_points = _parse_shape_points(from_lane_element.attrib.get("shape", ""))
        to_points = _parse_shape_points(to_lane_element.attrib.get("shape", ""))
        if not from_points or not to_points:
            continue
        scanned_count += 1
        connection_points = _parse_shape_points(connection_element.attrib.get("shape", ""))
        if len(connection_points) >= 2:
            before_gap = max(
                distance_2d(connection_points[0], from_points[-1]),
                distance_2d(connection_points[-1], to_points[0]),
            )
            max_endpoint_gap_before = max(max_endpoint_gap_before, before_gap)
        aligned_shape = _aligned_internal_connection_shape(connection_points, from_points, to_points)
        if aligned_shape is None:
            unrepaired_count += 1
            if len(examples) < MAX_INTERNAL_CONNECTION_ALIGN_EXAMPLES:
                examples.append(
                    {
                        "from": from_edge_id,
                        "to": to_edge_id,
                        "fromLane": from_lane_index,
                        "toLane": to_lane_index,
                        "reason": "unshapeable",
                    }
                )
            continue

        replacement_points, source = aligned_shape
        replacement_shape = _shape_string(replacement_points)
        replacement_length = f"{polyline_length(replacement_points):.3f}"
        after_gap = max(
            distance_2d(replacement_points[0], from_points[-1]),
            distance_2d(replacement_points[-1], to_points[0]),
        )
        max_endpoint_gap_after = max(max_endpoint_gap_after, after_gap)
        shape_changed = connection_element.attrib.get("shape") != replacement_shape
        length_changed = connection_element.attrib.get("length") != replacement_length
        via_lane_id = connection_element.attrib.get("via")
        via_lane_element = lanes_by_id.get(via_lane_id) if via_lane_id else None
        via_changed = (
            via_lane_element is not None
            and (
                via_lane_element.attrib.get("shape") != replacement_shape
                or via_lane_element.attrib.get("length") != replacement_length
            )
        )
        if not shape_changed and not length_changed and not via_changed:
            continue

        connection_element.set("shape", replacement_shape)
        connection_element.set("length", replacement_length)
        if via_lane_element is not None:
            via_lane_element.set("shape", replacement_shape)
            via_lane_element.set("length", replacement_length)
        aligned_count += 1
        if source == "trimmed":
            trimmed_count += 1
        elif source == "direct_fallback":
            direct_fallback_count += 1
        elif source == "tangent_fallback":
            tangent_fallback_count += 1
        if len(examples) < MAX_INTERNAL_CONNECTION_ALIGN_EXAMPLES:
            examples.append(
                {
                    "from": from_edge_id,
                    "to": to_edge_id,
                    "fromLane": from_lane_index,
                    "toLane": to_lane_index,
                    "via": via_lane_id,
                    "source": source,
                    "length_m": round(polyline_length(replacement_points), 6),
                }
            )

    if aligned_count:
        ET.indent(tree, space="    ")
        tree.write(path, encoding="utf-8", xml_declaration=True)

    return {
        "scanned_connection_count": scanned_count,
        "aligned_connection_count": aligned_count,
        "trimmed_connection_shape_count": trimmed_count,
        "direct_fallback_shape_count": direct_fallback_count,
        "tangent_fallback_shape_count": tangent_fallback_count,
        "unrepaired_connection_count": unrepaired_count,
        "max_endpoint_gap_before_m": round(max_endpoint_gap_before, 6),
        "max_endpoint_gap_after_m": round(max_endpoint_gap_after, 6),
        "examples": examples,
    }

def _repair_degenerate_internal_lane_shapes(net_path: str | Path) -> dict[str, object]:
    path = Path(net_path)
    tree = ET.parse(path)
    root = tree.getroot()
    lanes_by_id: dict[str, ET.Element] = {}
    lane_shapes: dict[str, tuple[Point3D, ...]] = {}
    internal_lane_ids: set[str] = set()

    for edge_element in root.findall("edge"):
        is_internal = _is_internal_edge(edge_element)
        for lane_element in edge_element.findall("lane"):
            lane_id = lane_element.attrib.get("id")
            shape = lane_element.attrib.get("shape")
            if lane_id is None or shape is None:
                continue
            lanes_by_id[lane_id] = lane_element
            lane_shapes[lane_id] = _parse_shape_points(shape)
            if is_internal:
                internal_lane_ids.add(lane_id)

    connection_by_via: dict[str, ET.Element] = {}
    for connection_element in root.findall("connection"):
        via_lane_id = connection_element.attrib.get("via")
        if via_lane_id:
            connection_by_via[via_lane_id] = connection_element

    degenerate_ids: list[str] = []
    repaired_ids: list[str] = []
    unrepaired_ids: list[str] = []
    examples: list[dict[str, object]] = []

    for lane_id in sorted(internal_lane_ids, key=_sort_key):
        points = lane_shapes.get(lane_id, tuple())
        if _polyline_length_2d(points) >= DEGENERATE_INTERNAL_LANE_XY_LENGTH_M:
            continue
        degenerate_ids.append(lane_id)
        connection = connection_by_via.get(lane_id)
        if connection is None:
            unrepaired_ids.append(lane_id)
            if len(examples) < 20:
                examples.append({"lane_id": lane_id, "reason": "missing_via_connection"})
            continue
        from_edge = connection.attrib.get("from")
        to_edge = connection.attrib.get("to")
        from_lane = connection.attrib.get("fromLane")
        to_lane = connection.attrib.get("toLane")
        if from_edge is None or to_edge is None or from_lane is None or to_lane is None:
            unrepaired_ids.append(lane_id)
            if len(examples) < 20:
                examples.append({"lane_id": lane_id, "reason": "incomplete_connection"})
            continue
        from_points = lane_shapes.get(_net_lane_id(from_edge, from_lane), tuple())
        to_points = lane_shapes.get(_net_lane_id(to_edge, to_lane), tuple())
        replacement_shape = _replacement_internal_shape(from_points, to_points)
        if replacement_shape is None:
            unrepaired_ids.append(lane_id)
            if len(examples) < 20:
                examples.append({"lane_id": lane_id, "reason": "missing_or_degenerate_neighbor_shapes"})
            continue
        lane_element = lanes_by_id[lane_id]
        lane_element.set("shape", _shape_string(replacement_shape))
        lane_shapes[lane_id] = replacement_shape
        repaired_ids.append(lane_id)
        if len(examples) < 20:
            examples.append(
                {
                    "lane_id": lane_id,
                    "from": from_edge,
                    "fromLane": from_lane,
                    "to": to_edge,
                    "toLane": to_lane,
                    "repaired_xy_length_m": round(_polyline_length_2d(replacement_shape), 6),
                }
            )

    if repaired_ids:
        ET.indent(tree, space="    ")
        tree.write(path, encoding="utf-8", xml_declaration=True)

    return {
        "scanned_internal_lane_count": len(internal_lane_ids),
        "degenerate_internal_lane_count": len(degenerate_ids),
        "repaired_internal_lane_count": len(repaired_ids),
        "unrepaired_internal_lane_count": len(unrepaired_ids),
        "examples": examples,
    }

def _patch_net_lane_lengths_to_shape(net_path: str | Path) -> dict[str, object]:
    path = Path(net_path)
    tree = ET.parse(path)
    root = tree.getroot()
    patched_count = 0
    patched_internal_count = 0
    patched_normal_count = 0
    max_abs_diff = 0.0
    max_abs_diff_lane_id: str | None = None

    for edge_element in root.findall("edge"):
        is_internal_edge = _is_internal_edge(edge_element)
        for lane_element in edge_element.findall("lane"):
            shape = lane_element.attrib.get("shape")
            length = lane_element.attrib.get("length")
            if shape is None or length is None:
                continue
            points = _parse_shape_points(shape)
            if len(points) < 2:
                continue
            shape_length = max(polyline_length(points), MIN_SUMO_LANE_LENGTH_M)
            current_length = float(length)
            abs_diff = abs(current_length - shape_length)
            if abs_diff > max_abs_diff:
                max_abs_diff = abs_diff
                max_abs_diff_lane_id = lane_element.attrib.get("id")
            rounded_length = f"{shape_length:.3f}"
            if lane_element.attrib.get("length") == rounded_length:
                continue
            lane_element.set("length", rounded_length)
            patched_count += 1
            if is_internal_edge:
                patched_internal_count += 1
            else:
                patched_normal_count += 1

    if patched_count:
        ET.indent(tree, space="    ")
        tree.write(path, encoding="utf-8", xml_declaration=True)

    return {
        "patched_lane_count": patched_count,
        "patched_normal_lane_count": patched_normal_count,
        "patched_internal_lane_count": patched_internal_count,
        "max_abs_diff_before_m": round(max_abs_diff, 6),
        "max_abs_diff_lane_id": max_abs_diff_lane_id,
    }

def _angle_axis_diff_deg(lhs: float, rhs: float) -> float:
    diff = abs((lhs - rhs) % 180.0)
    return min(diff, 180.0 - diff)

def _mean_axis_deg(axes: list[float]) -> float:
    if not axes:
        return 0.0
    x = sum(math.cos(math.radians(axis * 2.0)) for axis in axes)
    y = sum(math.sin(math.radians(axis * 2.0)) for axis in axes)
    return (math.degrees(math.atan2(y, x)) / 2.0) % 180.0

def _axis_clusters(link_infos: list[TLSLinkInfo]) -> list[tuple[float, set[int]]]:
    clusters: list[tuple[float, set[int]]] = []
    axes_by_index = {link.index: link.axis_deg for link in link_infos}
    for link in sorted(link_infos, key=lambda item: (item.axis_deg, item.index)):
        best_index: int | None = None
        best_diff = math.inf
        for cluster_index, (axis, _) in enumerate(clusters):
            diff = _angle_axis_diff_deg(link.axis_deg, axis)
            if diff < best_diff:
                best_diff = diff
                best_index = cluster_index
        if best_index is None or best_diff > JP_TLS_AXIS_CLUSTER_THRESHOLD_DEG:
            clusters.append((link.axis_deg, {link.index}))
            continue
        _, member_indices = clusters[best_index]
        member_indices.add(link.index)
        clusters[best_index] = (
            _mean_axis_deg([axes_by_index[index] for index in member_indices]),
            member_indices,
        )
    return sorted(clusters, key=lambda item: item[0])

def _edge_lane_shapes(root: ET.Element) -> dict[str, tuple[Point3D, ...]]:
    edge_shapes: dict[str, tuple[Point3D, ...]] = {}
    for edge_element in root.findall("edge"):
        edge_id = edge_element.attrib.get("id")
        if not edge_id or edge_id.startswith(":"):
            continue
        lane_element = edge_element.find("lane")
        shape = lane_element.attrib.get("shape") if lane_element is not None else edge_element.attrib.get("shape")
        if not shape:
            continue
        points = _parse_shape_points(shape)
        if len(points) >= 2:
            edge_shapes[edge_id] = points
    return edge_shapes

def _incoming_heading_from_shape(points: tuple[Point3D, ...]) -> float | None:
    for start, end in zip(reversed(points[:-1]), reversed(points[1:])):
        if distance_2d(start, end) > 0.01:
            return heading_deg(start, end)
    return None

def _tls_link_infos(root: ET.Element) -> dict[str, list[TLSLinkInfo]]:
    edge_shapes = _edge_lane_shapes(root)
    raw_links: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for connection_element in root.findall("connection"):
        tl_id = connection_element.attrib.get("tl")
        link_index = connection_element.attrib.get("linkIndex")
        from_edge_id = connection_element.attrib.get("from")
        if tl_id is None or link_index is None or from_edge_id is None or from_edge_id.startswith(":"):
            continue
        shape = edge_shapes.get(from_edge_id)
        if shape is None:
            continue
        incoming_heading = _incoming_heading_from_shape(shape)
        if incoming_heading is None:
            continue
        try:
            index = int(link_index)
        except ValueError:
            continue
        link_entry = raw_links[tl_id].setdefault(
            index,
            {
                "headings": [],
                "directions": set(),
            },
        )
        link_entry["headings"].append(incoming_heading)
        link_entry["directions"].add(connection_element.attrib.get("dir", "s"))

    result: dict[str, list[TLSLinkInfo]] = {}
    for tl_id, links_by_index in raw_links.items():
        link_infos: list[TLSLinkInfo] = []
        for index, link_entry in links_by_index.items():
            headings = link_entry["headings"]
            if not headings:
                continue
            heading_x = sum(math.cos(math.radians(heading)) for heading in headings)
            heading_y = sum(math.sin(math.radians(heading)) for heading in headings)
            incoming_heading = math.degrees(math.atan2(heading_y, heading_x)) % 360.0
            link_infos.append(
                TLSLinkInfo(
                    index=index,
                    incoming_heading_deg=incoming_heading,
                    directions=tuple(sorted(link_entry["directions"])),
                )
            )
        if link_infos:
            result[tl_id] = sorted(link_infos, key=lambda item: item.index)
    return result

def _phase_state(length: int, active_indices: set[int], right_turn_indices: set[int] | None = None) -> str:
    right_turn_indices = right_turn_indices or set()
    chars: list[str] = []
    for index in range(length):
        if index not in active_indices:
            chars.append("r")
        elif index in right_turn_indices:
            chars.append("g")
        else:
            chars.append("G")
    return "".join(chars)

def _japanese_tls_phases(link_infos: list[TLSLinkInfo], state_length: int) -> list[tuple[int, str]] | None:
    if not link_infos:
        return None
    clusters = _axis_clusters(link_infos)
    if len(clusters) <= 1:
        return None

    links_by_index = {link.index: link for link in link_infos}
    phases: list[tuple[int, str]] = []
    green_time = JP_TLS_GREEN_TIME_S if len(clusters) <= 2 else max(20, JP_TLS_GREEN_TIME_S - 10)
    all_red = "r" * state_length
    for _, active_indices in clusters:
        valid_active_indices = {index for index in active_indices if 0 <= index < state_length}
        if not valid_active_indices:
            continue
        right_turn_indices = {
            index
            for index in valid_active_indices
            if links_by_index[index].has_right_turn
        }
        through_indices = {
            index
            for index in valid_active_indices
            if links_by_index[index].has_non_right_turn
        }
        phases.append((green_time, _phase_state(state_length, valid_active_indices, right_turn_indices)))
        phases.append((JP_TLS_YELLOW_TIME_S, _phase_state(state_length, valid_active_indices).replace("G", "y")))
        phases.append((JP_TLS_ALL_RED_TIME_S, all_red))
        if right_turn_indices and through_indices:
            phases.append((JP_TLS_RIGHT_TURN_TIME_S, _phase_state(state_length, right_turn_indices)))
            phases.append((JP_TLS_YELLOW_TIME_S, _phase_state(state_length, right_turn_indices).replace("G", "y")))
            phases.append((JP_TLS_ALL_RED_TIME_S, all_red))

    return phases or None

def _joined_intersection_area_tls_ids(net_path: str | Path) -> set[str]:
    root = ET.parse(net_path).getroot()
    tls_ids: set[str] = set()
    for connection_element in root.findall("connection"):
        tl_id = connection_element.attrib.get("tl")
        via_lane_id = connection_element.attrib.get("via", "")
        if tl_id and via_lane_id.startswith(":ia_"):
            tls_ids.add(tl_id)
    return tls_ids

def _patch_net_japanese_tls_phases(
    net_path: str | Path,
    excluded_tls_ids: set[str] | None = None,
) -> dict[str, object]:
    path = Path(net_path)
    tree = ET.parse(path)
    root = tree.getroot()
    link_infos_by_tls = _tls_link_infos(root)
    excluded_tls_ids = excluded_tls_ids or set()
    patched_tls_ids: list[str] = []
    skipped_excluded_tls_ids: list[str] = []
    skipped_single_axis_tls_ids: list[str] = []
    max_phase_count = 0

    for tl_logic_element in root.findall("tlLogic"):
        tl_id = tl_logic_element.attrib.get("id")
        if tl_id is None:
            continue
        if tl_id in excluded_tls_ids:
            skipped_excluded_tls_ids.append(tl_id)
            continue
        link_infos = link_infos_by_tls.get(tl_id, [])
        if not link_infos:
            continue
        existing_phases = tl_logic_element.findall("phase")
        existing_state_length = max(
            [len(phase.attrib.get("state", "")) for phase in existing_phases] + [0]
        )
        state_length = max(existing_state_length, max(link.index for link in link_infos) + 1)
        phases = _japanese_tls_phases(link_infos, state_length)
        if phases is None:
            skipped_single_axis_tls_ids.append(tl_id)
            continue
        for phase_element in existing_phases:
            tl_logic_element.remove(phase_element)
        for duration, state in phases:
            ET.SubElement(
                tl_logic_element,
                "phase",
                {
                    "duration": str(duration),
                    "state": state,
                },
            )
        patched_tls_ids.append(tl_id)
        max_phase_count = max(max_phase_count, len(phases))

    if patched_tls_ids:
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)

    return {
        "patched_tls_count": len(patched_tls_ids),
        "patched_tls_ids": patched_tls_ids,
        "skipped_excluded_tls_count": len(skipped_excluded_tls_ids),
        "skipped_excluded_tls_ids": skipped_excluded_tls_ids,
        "skipped_single_axis_tls_count": len(skipped_single_axis_tls_ids),
        "max_phase_count": max_phase_count,
    }

def _is_normal_net_edge(edge_element: ET.Element) -> bool:
    edge_id = edge_element.attrib.get("id", "")
    if not edge_id or edge_id.startswith(":"):
        return False
    return edge_element.attrib.get("function", "") not in {"internal", "crossing", "walkingarea"}

def _normal_net_edge_ids(root: ET.Element) -> set[str]:
    return {
        edge_element.attrib["id"]
        for edge_element in root.findall("edge")
        if _is_normal_net_edge(edge_element) and "id" in edge_element.attrib
    }

def _normal_net_edge_lengths(root: ET.Element, edge_ids: set[str]) -> dict[str, float]:
    lengths: dict[str, float] = {}
    for edge_element in root.findall("edge"):
        edge_id = edge_element.attrib.get("id")
        if edge_id not in edge_ids:
            continue
        lane_lengths: list[float] = []
        for lane_element in edge_element.findall("lane"):
            length = lane_element.attrib.get("length")
            if length is not None:
                lane_lengths.append(float(length))
                continue
            shape = lane_element.attrib.get("shape")
            if shape:
                lane_lengths.append(polyline_length(_parse_shape_points(shape)))
        lengths[edge_id] = max(lane_lengths) if lane_lengths else 0.0
    return lengths

def _write_randomtrips_safe_weights(
    out_dir: Path,
    edge_ids: set[str],
    eligible_edge_ids: set[str],
    edge_lengths: dict[str, float],
) -> dict[str, object]:
    prefix = out_dir / "randomtrips.safe"
    suffixes = {
        "src": ".src.xml",
        "dst": ".dst.xml",
        "via": ".via.xml",
    }
    paths = {interval_id: Path(str(prefix) + suffix) for interval_id, suffix in suffixes.items()}
    for interval_id, suffix in suffixes.items():
        root = ET.Element("edgedata")
        interval = ET.SubElement(root, "interval", {"id": interval_id, "begin": "0", "end": "1"})
        for edge_id in sorted(edge_ids, key=_sort_key):
            value = max(edge_lengths.get(edge_id, 1.0), 0.001) if edge_id in eligible_edge_ids else 0.0
            ET.SubElement(interval, "edge", {"id": edge_id, "value": f"{value:.3f}"})
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(paths[interval_id], encoding="utf-8", xml_declaration=True)

    return {
        "weights_prefix": str(prefix),
        "src_path": str(paths["src"]),
        "dst_path": str(paths["dst"]),
        "via_path": str(paths["via"]),
    }

def _summarize_net_connectivity_and_write_safe_weights(net_path: str | Path, out_dir: str | Path) -> dict[str, object]:
    path = Path(net_path)
    root = ET.parse(path).getroot()
    normal_edge_ids = _normal_net_edge_ids(root)
    incoming_counts: dict[str, int] = {edge_id: 0 for edge_id in normal_edge_ids}
    outgoing_counts: dict[str, int] = {edge_id: 0 for edge_id in normal_edge_ids}
    for connection_element in root.findall("connection"):
        from_edge = connection_element.attrib.get("from")
        to_edge = connection_element.attrib.get("to")
        if from_edge in outgoing_counts:
            outgoing_counts[from_edge] += 1
        if to_edge in incoming_counts:
            incoming_counts[to_edge] += 1

    no_outgoing_edge_ids = sorted(
        (edge_id for edge_id, count in outgoing_counts.items() if count == 0),
        key=_sort_key,
    )
    no_incoming_edge_ids = sorted(
        (edge_id for edge_id, count in incoming_counts.items() if count == 0),
        key=_sort_key,
    )
    eligible_edge_ids = {
        edge_id
        for edge_id in normal_edge_ids
        if incoming_counts[edge_id] > 0 and outgoing_counts[edge_id] > 0
    }
    weights_summary = _write_randomtrips_safe_weights(
        Path(out_dir),
        normal_edge_ids,
        eligible_edge_ids,
        _normal_net_edge_lengths(root, normal_edge_ids),
    )

    return {
        "normal_edge_count": len(normal_edge_ids),
        "safe_randomtrips_edge_count": len(eligible_edge_ids),
        "no_outgoing_edge_count": len(no_outgoing_edge_ids),
        "no_incoming_edge_count": len(no_incoming_edge_ids),
        "no_outgoing_edge_ids": no_outgoing_edge_ids,
        "no_incoming_edge_ids": no_incoming_edge_ids,
        "safe_randomtrips_weights": weights_summary,
    }
