from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ll2sumo.geometry import distance_2d, heading_deg, polyline_length
from ll2sumo.geometry import first_nonzero_segment as _first_nonzero_segment
from ll2sumo.geometry import last_nonzero_segment as _last_nonzero_segment
from ll2sumo.model import Point3D
from ll2sumo.sumo_xml import id_sort_key as _sort_key
from ll2sumo.sumo_xml import is_internal_edge as _is_internal_edge
from ll2sumo.sumo_xml import net_lane_id as _net_lane_id
from ll2sumo.sumo_xml import parse_shape_points as _parse_shape_points
from ll2sumo.sumo_xml import polyline_length_2d as _polyline_length_2d
from ll2sumo.sumo_xml import shape_string as _shape_string
from ll2sumo.sumo_xml import usable_connection_shape as _usable_connection_shape

MIN_SUMO_LANE_LENGTH_M, DEGENERATE_INTERNAL_LANE_XY_LENGTH_M = 0.1, 0.01
REPAIRED_INTERNAL_LANE_FALLBACK_LENGTH_M = 0.25
MAX_JOINED_UNMAPPED_CONNECTION_EXAMPLES = 20
JP_TLS_GREEN_TIME_S, JP_TLS_YELLOW_TIME_S, JP_TLS_ALL_RED_TIME_S = 35, 3, 2
JP_TLS_AXIS_CLUSTER_THRESHOLD_DEG = 35.0
MAX_TLS_PHASE_SYNC_EXAMPLES = 20

@dataclass(frozen=True)
class TLSLinkInfo:
    index: int
    incoming_heading_deg: float
    directions: tuple[str, ...]
    incoming_lane_keys: tuple[tuple[str, str], ...] = tuple()
    outgoing_lane_keys: tuple[tuple[str, str], ...] = tuple()

    @property
    def axis_deg(self) -> float:
        return self.incoming_heading_deg % 180.0

    @property
    def has_right_turn(self) -> bool:
        return any(direction in {"r", "R"} for direction in self.directions)

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
    ambiguous = False
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
        if abs(projected_distance - best_distance) <= 1e-9 and abs(projected_along - best_along) > 0.01:
            ambiguous = True
        if projected_distance < best_distance - 1e-9:
            ambiguous = False
            best_distance = projected_distance
            best_along = projected_along
            best_point = projected
        along_before += segment_length
    if best_point is None or ambiguous:
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
    if len(sliced) > 1 and distance_2d(sliced[-1], end_point) < 0.01:
        sliced[-1] = end_point
    else:
        sliced.append(end_point)
    return _usable_connection_shape(sliced)

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
    lane_shapes = {
        lane.attrib["id"]: _parse_shape_points(lane.get("shape", ""))
        for edge in root.findall("edge") for lane in edge.findall("lane")
    }
    discontinuities = []
    for lane_id, points in lane_shapes.items():
        turn = _max_shape_turn(points)
        if turn > 45.0:
            discontinuities.append({"lane_id": lane_id, "reason": "lane_shape_turn", "angle_deg": round(turn, 6)})
    for connection in root.findall("connection"):
        key = _connection_key(connection)
        if key is None:
            continue
        from_id = _net_lane_id(key[0], key[2])
        to_id = connection.get("via") or _net_lane_id(key[1], key[3])
        incoming, outgoing = lane_shapes.get(from_id, ()), lane_shapes.get(to_id, ())
        if not incoming or not outgoing:
            continue
        gap = distance_2d(incoming[-1], outgoing[0])
        dz = abs(incoming[-1].z - outgoing[0].z)
        if gap > 0.001 or dz > 0.001:
            discontinuities.append({"from_lane": from_id, "to_lane": to_id, "reason": "endpoint_gap", "gap_m": round(gap, 6), "z_gap_m": round(dz, 6)})
        turn = _max_shape_turn((*incoming[-2:], *outgoing[:2]))
        if turn > 45.0:
            discontinuities.append({"from_lane": from_id, "to_lane": to_id, "reason": "connection_boundary_turn", "angle_deg": round(turn, 6)})
    return {
        "scanned_internal_lane_count": scanned_count,
        "degenerate_internal_lane_count": degenerate_count,
        "examples": examples,
        "discontinuity_counts": dict(Counter(item["reason"] for item in discontinuities)),
        "discontinuities": discontinuities,
    }

def _plain_connection_shapes(connections_path: str | Path | None) -> dict[tuple[str, str, str, str], tuple[Point3D, ...]]:
    if connections_path is None:
        return {}
    path = Path(connections_path)
    if not path.exists():
        return {}
    root = ET.parse(path).getroot()
    shapes: dict[tuple[str, str, str, str], tuple[Point3D, ...]] = {}
    for connection_element in root.findall("connection"):
        connection_shape = connection_element.attrib.get("shape")
        if not connection_shape:
            continue
        from_edge_id = connection_element.attrib.get("from")
        to_edge_id = connection_element.attrib.get("to")
        from_lane_index = connection_element.attrib.get("fromLane")
        to_lane_index = connection_element.attrib.get("toLane")
        if from_edge_id is None or to_edge_id is None or from_lane_index is None or to_lane_index is None:
            continue
        points = _parse_shape_points(connection_shape)
        usable_shape = _usable_connection_shape(points)
        if usable_shape is None:
            continue
        shapes[(from_edge_id, to_edge_id, from_lane_index, to_lane_index)] = usable_shape
    return shapes

def _connection_key(connection_element: ET.Element) -> tuple[str, str, str, str] | None:
    from_edge_id = connection_element.attrib.get("from")
    to_edge_id = connection_element.attrib.get("to")
    from_lane_index = connection_element.attrib.get("fromLane")
    to_lane_index = connection_element.attrib.get("toLane")
    if from_edge_id is None or to_edge_id is None or from_lane_index is None or to_lane_index is None:
        return None
    return from_edge_id, to_edge_id, from_lane_index, to_lane_index

def _plain_connection_keys(connections_path: str | Path) -> set[tuple[str, str, str, str]]:
    root = ET.parse(connections_path).getroot()
    keys: set[tuple[str, str, str, str]] = set()
    for connection_element in root.findall("connection"):
        connection_key = _connection_key(connection_element)
        if connection_key is not None:
            keys.add(connection_key)
    return keys

def _joined_unmapped_connection_keys(
    net_path: str | Path,
    plain_connections_path: str | Path,
) -> list[tuple[str, str, str, str]]:
    plain_keys = _plain_connection_keys(plain_connections_path)
    root = ET.parse(net_path).getroot()
    keys: set[tuple[str, str, str, str]] = set()
    for connection_element in root.findall("connection"):
        via_lane_id = connection_element.attrib.get("via", "")
        if not via_lane_id.startswith(":ia_"):
            continue
        connection_key = _connection_key(connection_element)
        if connection_key is None:
            continue
        from_edge_id, to_edge_id, _, _ = connection_key
        if from_edge_id.startswith(":") or to_edge_id.startswith(":"):
            continue
        if connection_key in plain_keys:
            continue
        keys.add(connection_key)
    return sorted(keys, key=lambda key: (_sort_key(key[0]), _sort_key(key[1]), int(key[2]), int(key[3])))

def _joined_unmapped_connection_examples(
    keys: list[tuple[str, str, str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "from": from_edge_id,
            "to": to_edge_id,
            "fromLane": from_lane_index,
            "toLane": to_lane_index,
        }
        for from_edge_id, to_edge_id, from_lane_index, to_lane_index in keys[:MAX_JOINED_UNMAPPED_CONNECTION_EXAMPLES]
    ]

def _summarize_joined_unmapped_connections(
    net_path: str | Path,
    plain_connections_path: str | Path,
) -> dict[str, object]:
    keys = _joined_unmapped_connection_keys(net_path, plain_connections_path)
    return {
        "joined_unmapped_connection_count": len(keys),
        "examples": _joined_unmapped_connection_examples(keys),
    }

def _write_joined_unmapped_connection_deletions(
    net_path: str | Path,
    plain_connections_path: str | Path,
    delete_connections_path: str | Path,
) -> dict[str, object]:
    keys = _joined_unmapped_connection_keys(net_path, plain_connections_path)
    path = Path(delete_connections_path)
    root = ET.Element("connections")
    for from_edge_id, to_edge_id, from_lane_index, to_lane_index in keys:
        ET.SubElement(
            root,
            "delete",
            {
                "from": from_edge_id,
                "to": to_edge_id,
                "fromLane": from_lane_index,
                "toLane": to_lane_index,
            },
        )
    if keys:
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(path, encoding="utf-8", xml_declaration=True)
    elif path.exists():
        path.unlink()
    return {
        "joined_unmapped_connection_count_before": len(keys),
        "deleted_joined_unmapped_connection_count": len(keys),
        "delete_connections_path": str(path) if keys else None,
        "examples": _joined_unmapped_connection_examples(keys),
    }

def _split_shape_at(points: tuple[Point3D, ...], along: float) -> tuple[tuple[Point3D, ...], tuple[Point3D, ...]] | None:
    total = _polyline_length_2d(points)
    if not 0.01 < along < total - 0.01:
        return None
    distance = 0.0
    for first, second in zip(points, points[1:]):
        length = distance_2d(first, second)
        if length and distance + length >= along:
            point = _interpolate_point(first, second, (along - distance) / length)
            before = _slice_polyline_between(points, 0.0, points[0], along, point)
            after = _slice_polyline_between(points, along, point, total, points[-1])
            return (before, after) if before and after else None
        distance += length
    return None


def _max_shape_turn(points: tuple[Point3D, ...]) -> float:
    headings = [heading_deg(a, b) for a, b in zip(points, points[1:]) if distance_2d(a, b) >= 0.01]
    return max((abs((b - a + 180.0) % 360.0 - 180.0) for a, b in zip(headings, headings[1:])), default=0.0)


def _internal_lane_chain(owner: ET.Element, outgoing: dict[str, list[ET.Element]]) -> tuple[list[str], list[ET.Element]]:
    """Follow actual via links; an owner shape may span several internal lanes."""
    lanes: list[str] = []
    links: list[ET.Element] = []
    via = owner.get("via")
    while via:
        if via in lanes:
            return [], []
        lanes.append(via)
        downstream = outgoing.get(via, [])
        if len(downstream) != 1:
            return [], []
        link = downstream[0]
        if (link.get("to"), link.get("toLane")) != (owner.get("to"), owner.get("toLane")):
            return [], []
        links.append(link)
        via = link.get("via")
    return lanes, links


def _source_aligned_internal_shapes(
    shapes: list[tuple[Point3D, ...]], reference: tuple[Point3D, ...],
) -> list[tuple[Point3D, ...]]:
    """Resolve netconvert's Y reflection using the source curve, not Y's sign.

    Verify every vertex, height, and projection order before correcting the
    coordinate frame; a failed alignment may safely retain this source geometry.
    """
    if len(reference) < 2 or any(len(shape) < 2 for shape in shapes):
        return shapes
    for reflect in (False, True):
        candidate = [tuple(Point3D(p.x, -p.y, p.z) for p in shape) for shape in shapes] if reflect else shapes
        points = [point for shape in candidate for point in shape]
        projections = [_project_point_on_polyline(reference, point) for point in points]
        if all(projection is not None and projection[0] <= 0.001
               and abs(point.z - projection[2].z) <= 0.001
               for point, projection in zip(points, projections)) and all(
            after[1] >= before[1] - 1e-9 for before, after in zip(projections, projections[1:])
        ):
            return candidate
    return shapes


def _align_internal_connection_shapes_to_net_lanes(
    net_path: str | Path,
    plain_connections_path: str | Path | None = None,
    plain_edges_path: str | Path | None = None,
) -> dict[str, object]:
    """Restore source curves and allocate junction stubs once per successor.

    Correct source-verified Y reflections before testing alignment proposals.
    If a movement cannot be reconstructed, keep its corrected successor group
    and adjoining endpoints, then recompute the remaining proposals.
    """
    path = Path(net_path)
    tree = ET.parse(path)
    root = tree.getroot()
    lanes = {lane.attrib["id"]: lane for edge in root.findall("edge") for lane in edge.findall("lane")}
    original = {key: _parse_shape_points(lane.get("shape", "")) for key, lane in lanes.items()}
    normal_ids = {lane.attrib["id"] for edge in root.findall("edge") if not _is_internal_edge(edge) for lane in edge.findall("lane")}
    source = {key: original[key] for key in normal_ids}
    if plain_edges_path is not None:
        for edge in ET.parse(plain_edges_path).getroot().findall("edge"):
            for lane in edge.findall("lane"):
                key = _net_lane_id(edge.attrib["id"], lane.attrib["index"])
                if key in normal_ids:
                    source[key] = _parse_shape_points(lane.get("shape", ""))
    plain = _plain_connection_shapes(plain_connections_path)
    outgoing: dict[str, list[ET.Element]] = defaultdict(list)
    owners: list[ET.Element] = []
    for connection in root.findall("connection"):
        key = _connection_key(connection)
        if key is None:
            continue
        outgoing[_net_lane_id(key[0], key[2])].append(connection)
        if key[0].startswith(":") or key[1].startswith(":") or not connection.get("via"):
            continue
        owners.append(connection)
    keys = [_connection_key(owner) for owner in owners]
    chains = [_internal_lane_chain(owner, outgoing) for owner in owners]
    owners_by_via: dict[str, list[int]] = defaultdict(list)
    for index, (chain, _) in enumerate(chains):
        for via in chain or [owners[index].get("via")]:
            owners_by_via[via].append(index)
    reflected: dict[str, tuple[Point3D, ...]] = {}
    for index, (chain, links) in enumerate(chains):
        if not chain or any(via not in original or len(owners_by_via[via]) != 1 for via in chain):
            continue
        reference = plain.get(keys[index], ())
        shapes = _source_aligned_internal_shapes([original[via] for via in chain], reference)
        corrections = {via: shape for via, shape in zip(chain, shapes) if shape != original[via]}
        if corrections:
            reflected.update(corrections)
            for connection in [owners[index], *links]:
                shape = _parse_shape_points(connection.get("shape", ""))
                corrected = _source_aligned_internal_shapes([shape], reference)[0]
                if corrected != shape:
                    connection.set("shape", _shape_string(corrected))
    internal_shapes = {**original, **reflected}
    from_ids = [_net_lane_id(key[0], key[2]) for key in keys]
    to_ids = [_net_lane_id(key[1], key[3]) for key in keys]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, lane_id in enumerate(to_ids):
        groups[lane_id].append(index)
    failures: dict[int, str] = {}
    frozen: set[str] = set()
    proposals: dict[int, tuple[tuple[Point3D, ...], list[tuple[Point3D, ...]]]] = {}
    prefixes: dict[str, tuple[Point3D, ...]] = {}

    while True:
        normal = {key: original[key] if key in frozen else points for key, points in source.items()}
        prefixes = {}
        problems: dict[int, str] = {}
        for target, indices in groups.items():
            if any(index in failures for index in indices):
                continue
            points = normal.get(target, ())
            needs_stub = any(
                normal.get(from_ids[i]) and points
                and distance_2d(normal[from_ids[i]][-1], points[0]) < 0.01
                and not (plain.get(keys[i]) and distance_2d(plain[keys[i]][-1], points[0]) <= 0.001)
                for i in indices
            )
            if not needs_stub:
                continue
            split = _split_shape_at(points, REPAIRED_INTERNAL_LANE_FALLBACK_LENGTH_M)
            if split is None or distance_2d(split[1][0], split[1][1]) < 0.05:
                problems[indices[0]] = "successor_too_short_for_stub"
                continue
            if abs(split[0][-1].z - points[0].z) > 0.05:
                problems[indices[0]] = "stub_height_change_exceeds_limit"
                continue
            prefixes[target], normal[target] = split

        proposals = {}
        via_proposals: dict[str, tuple[Point3D, ...]] = {}
        for index, owner in enumerate(owners):
            if index in failures or index in problems:
                continue
            chain, _ = chains[index]
            incoming, successor = normal.get(from_ids[index], ()), normal.get(to_ids[index], ())
            if not chain or any(lane_id not in original for lane_id in chain):
                problems[index] = "non_unique_or_missing_via_chain"
                continue
            if len(incoming) < 2 or len(successor) < 2:
                problems[index] = "missing_normal_lane_shape"
                continue
            start, end = incoming[-1], successor[0]
            curve = plain.get(keys[index], _parse_shape_points(owner.get("shape", "")))
            source_end = source.get(to_ids[index], successor)[0]
            # Tangent fallback shapes extend into the next lane; they are not
            # source intersection curves and must not be copied over that lane.
            if curve and distance_2d(curve[-1], source_end) <= 0.001:
                # An unchanged adjoining lane may already be cut by netconvert.
                # Include the omitted source interval, then clip the whole curve.
                for lane_id, endpoint, at_start in ((from_ids[index], start, True), (to_ids[index], end, False)):
                    reference = source.get(lane_id, ())
                    projection = _project_point_on_polyline(reference, endpoint)
                    if projection and projection[0] <= 0.001:
                        if at_start:
                            segment = _slice_polyline_between(
                                reference, projection[1], endpoint, _polyline_length_2d(reference), reference[-1],
                            )
                        else:
                            segment = _slice_polyline_between(reference, 0.0, reference[0], projection[1], endpoint)
                        if segment:
                            curve = (*segment, *curve) if at_start else (*curve, *segment)
            else:
                curve = (start, *prefixes.get(to_ids[index], ()), end)
            first = _project_point_on_polyline(curve, start)
            last = _project_point_on_polyline(curve, end)
            points = None
            if first and last and first[0] <= 0.001 and last[0] <= 0.001:
                points = _slice_polyline_between(curve, first[1], start, last[1], end)
            if points is None:
                problems[index] = "source_curve_endpoint_mismatch"
                continue
            points = _parse_shape_points(_shape_string(points))
            entry, exit_segment = _last_nonzero_segment(incoming), _first_nonzero_segment(successor)
            if entry is None or exit_segment is None:
                problems[index] = "degenerate_normal_lane_shape"
                continue
            if _max_shape_turn((*entry, *points, *exit_segment)) > 45.0 + 1e-6:
                problems[index] = "heading_change_exceeds_limit"
                continue
            split_shapes = [internal_shapes[lane_id] for lane_id in chain]
            cuts = [(0.0, points[0])]
            for before, after in zip(split_shapes, split_shapes[1:]):
                if not before or not after:
                    break
                boundary = _interpolate_point(before[-1], after[0], 0.5)
                projection = _project_point_on_polyline(points, boundary)
                if projection is None or projection[1] <= cuts[-1][0] + 0.01:
                    break
                cuts.append((projection[1], projection[2]))
            cuts.append((_polyline_length_2d(points), points[-1]))
            pieces = [_slice_polyline_between(points, a, p, b, q) for (a, p), (b, q) in zip(cuts, cuts[1:])]
            if len(pieces) != len(chain) or any(piece is None for piece in pieces):
                problems[index] = "ambiguous_internal_split"
                continue
            pieces = [_parse_shape_points(_shape_string(piece)) for piece in pieces]
            if _polyline_length_2d(points) <= 0.251:
                lengths = [distance_2d(*entry), distance_2d(*exit_segment)]
                lengths += [distance_2d(a, b) for piece in pieces for a, b in zip(piece, piece[1:])]
                if min(lengths) < 0.05 - 1e-9:
                    problems[index] = "stub_segment_too_short"
                    continue
                boundaries = [(from_ids[index], incoming, (-1,)), (to_ids[index], successor, (0,))]
                boundaries += [(via, piece, (0, -1)) for via, piece in zip(chain, pieces)]
                if any(not original[lane_id] or any(abs(shape[pos].z - original[lane_id][pos].z) > 0.05
                       for pos in positions) for lane_id, shape, positions in boundaries):
                    problems[index] = "stub_height_change_exceeds_limit"
                    continue
            if any(lane_id in via_proposals and via_proposals[lane_id] != piece for lane_id, piece in zip(chain, pieces)):
                problems[index] = "shared_via_shape_conflict"
                continue
            via_proposals.update(zip(chain, pieces))
            proposals[index] = (points, pieces)
        if not problems:
            break
        while problems:
            index, reason = problems.popitem()
            if index in failures:
                continue
            failures[index] = reason
            frozen.update((from_ids[index], to_ids[index]))
            for other in groups[to_ids[index]]:
                if other not in failures:
                    problems.setdefault(other, "ambiguous_successor_group")
            for via in chains[index][0] or [owners[index].get("via")]:
                for other in owners_by_via[via]:
                    if other not in failures:
                        problems.setdefault(other, "shared_ambiguous_via")
        # Each pass excludes at least one additional movement, so this converges.

    replacements = {**reflected, **normal}
    for index, (points, pieces) in proposals.items():
        owner = owners[index]
        owner.set("shape", _shape_string(points))
        chain, links = chains[index]
        replacements.update(zip(chain, pieces))
        # Continuation connections own only the remaining via chain, never the
        # complete external-to-external movement. A direct exit has no shape.
        for offset, link in enumerate(links):
            if offset + 1 == len(pieces):
                link.attrib.pop("shape", None)
                link.attrib.pop("length", None)
            else:
                tail = tuple(point for piece in pieces[offset + 1:] for point in piece)
                link.set("shape", _shape_string(tail))
                link.set("length", f"{sum(max(polyline_length(piece), MIN_SUMO_LANE_LENGTH_M) for piece in pieces[offset + 1:]):.3f}")
        owner.set("length", f"{sum(max(polyline_length(piece), MIN_SUMO_LANE_LENGTH_M) for piece in pieces):.3f}")
    changed = {key for key, points in replacements.items() if original[key] != points}
    for key in changed:
        lanes[key].set("shape", _shape_string(replacements[key]))
    ET.indent(tree, space="    ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    unresolved = [
        dict(zip(("from", "to", "fromLane", "toLane"), keys[i]), via=owners[i].get("via"), reason=reason)
        for i, reason in sorted(failures.items())
    ]
    repaired_degenerate = sum(key not in normal_ids and _polyline_length_2d(original[key]) < 0.01 for key in changed)
    return {
        "scanned_connection_count": len(owners),
        "aligned_connection_count": len(proposals),
        "restored_normal_lane_count": len(changed & normal_ids),
        "repaired_internal_lane_count": len(changed - normal_ids),
        "repaired_degenerate_internal_lane_count": repaired_degenerate,
        "allocated_successor_stub_count": len(prefixes),
        "split_connection_count": sum(len(chains[i][0]) > 1 for i in proposals),
        "corrected_y_reflection_lane_count": len(reflected),
        "unrepaired_connection_count": len(failures),
        "unrepaired_connections": unresolved,
        "reason_counts": dict(Counter(failures.values())),
        "max_endpoint_gap_after_m": max(
            (max(distance_2d(points[0], normal[from_ids[i]][-1]), distance_2d(points[-1], normal[to_ids[i]][0]))
             for i, (points, _) in proposals.items()), default=0.0,
        ),
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

def _angle_diff_deg(lhs: float, rhs: float) -> float:
    diff = abs((lhs - rhs) % 360.0)
    return min(diff, 360.0 - diff)

def _mean_heading_deg(headings: list[float]) -> float:
    if not headings:
        return 0.0
    x = sum(math.cos(math.radians(heading)) for heading in headings)
    y = sum(math.sin(math.radians(heading)) for heading in headings)
    return math.degrees(math.atan2(y, x)) % 360.0

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

def _approach_clusters(link_infos: list[TLSLinkInfo]) -> list[tuple[float, set[int]]]:
    clusters: list[tuple[float, set[int]]] = []
    headings_by_index = {link.index: link.incoming_heading_deg for link in link_infos}
    for link in sorted(link_infos, key=lambda item: (item.incoming_heading_deg, item.index)):
        best_index: int | None = None
        best_diff = math.inf
        for cluster_index, (heading, _) in enumerate(clusters):
            diff = _angle_diff_deg(link.incoming_heading_deg, heading)
            if diff < best_diff:
                best_diff = diff
                best_index = cluster_index
        if best_index is None or best_diff > JP_TLS_AXIS_CLUSTER_THRESHOLD_DEG:
            clusters.append((link.incoming_heading_deg, {link.index}))
            continue
        _, member_indices = clusters[best_index]
        member_indices.add(link.index)
        clusters[best_index] = (
            _mean_heading_deg([headings_by_index[index] for index in member_indices]),
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

def _normal_lane_shapes(root: ET.Element) -> dict[tuple[str, str], tuple[Point3D, ...]]:
    lane_shapes: dict[tuple[str, str], tuple[Point3D, ...]] = {}
    for edge_element in root.findall("edge"):
        edge_id = edge_element.attrib.get("id")
        if not edge_id or edge_id.startswith(":"):
            continue
        for lane_element in edge_element.findall("lane"):
            lane_index = lane_element.attrib.get("index")
            shape = lane_element.attrib.get("shape")
            if lane_index is None or not shape:
                continue
            points = _parse_shape_points(shape)
            if len(points) >= 2:
                lane_shapes[(edge_id, lane_index)] = points
    return lane_shapes

def _phase_signal_color(state: str, index: int) -> str | None:
    if index >= len(state):
        return None
    signal = state[index]
    if signal in {"G", "g"}:
        return "green"
    if signal == "r":
        return "red"
    return None

def _incoming_heading_from_shape(points: tuple[Point3D, ...]) -> float | None:
    for start, end in zip(reversed(points[:-1]), reversed(points[1:])):
        if distance_2d(start, end) > 0.01:
            return heading_deg(start, end)
    return None

def _tls_link_infos(root: ET.Element) -> dict[str, list[TLSLinkInfo]]:
    edge_shapes = _edge_lane_shapes(root)
    lane_shapes = _normal_lane_shapes(root)
    raw_links: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for connection_element in root.findall("connection"):
        tl_id = connection_element.attrib.get("tl")
        link_index = connection_element.attrib.get("linkIndex")
        from_edge_id = connection_element.attrib.get("from")
        from_lane_index = connection_element.attrib.get("fromLane")
        to_edge_id = connection_element.attrib.get("to")
        to_lane_index = connection_element.attrib.get("toLane")
        if (
            tl_id is None
            or link_index is None
            or from_edge_id is None
            or from_lane_index is None
            or to_edge_id is None
            or to_lane_index is None
            or from_edge_id.startswith(":")
        ):
            continue
        shape = lane_shapes.get((from_edge_id, from_lane_index)) or edge_shapes.get(from_edge_id)
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
                "incoming_lane_keys": set(),
                "outgoing_lane_keys": set(),
            },
        )
        link_entry["headings"].append(incoming_heading)
        link_entry["directions"].add(connection_element.attrib.get("dir", "s"))
        link_entry["incoming_lane_keys"].add((from_edge_id, from_lane_index))
        link_entry["outgoing_lane_keys"].add((to_edge_id, to_lane_index))

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
                    incoming_lane_keys=tuple(sorted(link_entry["incoming_lane_keys"])),
                    outgoing_lane_keys=tuple(sorted(link_entry["outgoing_lane_keys"])),
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

def _shared_target_lane_indices(active_indices: set[int], links_by_index: dict[int, TLSLinkInfo]) -> set[int]:
    indices_by_target: dict[tuple[str, str], set[int]] = defaultdict(set)
    for index in active_indices:
        link = links_by_index.get(index)
        if link is None:
            continue
        for target_lane_key in link.outgoing_lane_keys:
            indices_by_target[target_lane_key].add(index)
    return {
        index
        for indices in indices_by_target.values()
        if len(indices) > 1
        for index in indices
    }

def _tls_phase_sync_audit(
    tl_logic_elements: list[ET.Element],
    link_infos_by_tls: dict[str, list[TLSLinkInfo]],
) -> dict[str, object]:
    mixed_lane_count = 0
    mixed_approach_count = 0
    examples: list[dict[str, object]] = []

    for tl_logic_element in tl_logic_elements:
        tl_id = tl_logic_element.attrib.get("id")
        if tl_id is None:
            continue
        link_infos = link_infos_by_tls.get(tl_id, [])
        if not link_infos:
            continue
        lane_keys_by_index = {
            link.index: link.incoming_lane_keys
            for link in link_infos
        }
        approach_indices = [
            member_indices
            for _, member_indices in _approach_clusters(link_infos)
        ]

        for phase_index, phase_element in enumerate(tl_logic_element.findall("phase")):
            state = phase_element.attrib.get("state", "")
            colors_by_lane: dict[tuple[str, str], set[str]] = defaultdict(set)
            for link in link_infos:
                color = _phase_signal_color(state, link.index)
                if color is None:
                    continue
                for lane_key in lane_keys_by_index.get(link.index, tuple()):
                    colors_by_lane[lane_key].add(color)
            for (from_edge_id, from_lane_index), colors in sorted(colors_by_lane.items(), key=lambda item: (_sort_key(item[0][0]), int(item[0][1]))):
                if colors != {"red", "green"}:
                    continue
                mixed_lane_count += 1
                if len(examples) < MAX_TLS_PHASE_SYNC_EXAMPLES:
                    examples.append(
                        {
                            "type": "incoming_lane",
                            "tl": tl_id,
                            "phase_index": phase_index,
                            "from": from_edge_id,
                            "fromLane": from_lane_index,
                            "state": state,
                        }
                    )

            for approach_index, member_indices in enumerate(approach_indices):
                colors = {
                    color
                    for link_index in member_indices
                    for color in [_phase_signal_color(state, link_index)]
                    if color is not None
                }
                if colors != {"red", "green"}:
                    continue
                mixed_approach_count += 1
                if len(examples) < MAX_TLS_PHASE_SYNC_EXAMPLES:
                    examples.append(
                        {
                            "type": "approach",
                            "tl": tl_id,
                            "phase_index": phase_index,
                            "approach_index": approach_index,
                            "state": state,
                        }
                    )

    return {
        "mixed_same_incoming_lane_phase_count": mixed_lane_count,
        "mixed_same_approach_phase_count": mixed_approach_count,
        "examples": examples,
    }

def _summarize_tls_phase_sync(net_path: str | Path) -> dict[str, object]:
    root = ET.parse(net_path).getroot()
    link_infos_by_tls = _tls_link_infos(root)
    return _tls_phase_sync_audit(root.findall("tlLogic"), link_infos_by_tls)

def _japanese_tls_phases(link_infos: list[TLSLinkInfo], state_length: int) -> list[tuple[int, str]] | None:
    if not link_infos:
        return None
    clusters = _axis_clusters(link_infos)
    if not clusters:
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
        permissive_indices = right_turn_indices | _shared_target_lane_indices(valid_active_indices, links_by_index)
        phases.append((green_time, _phase_state(state_length, valid_active_indices, permissive_indices)))
        phases.append((JP_TLS_YELLOW_TIME_S, _phase_state(state_length, valid_active_indices).replace("G", "y")))
        phases.append((JP_TLS_ALL_RED_TIME_S, all_red))

    return phases or None

def _patch_net_japanese_tls_phases(net_path: str | Path) -> dict[str, object]:
    path = Path(net_path)
    tree = ET.parse(path)
    root = tree.getroot()
    link_infos_by_tls = _tls_link_infos(root)
    tl_logic_elements = root.findall("tlLogic")
    before_audit = _tls_phase_sync_audit(tl_logic_elements, link_infos_by_tls)
    patched_tls_ids: list[str] = []
    skipped_single_axis_tls_ids: list[str] = []
    max_phase_count = 0

    for tl_logic_element in tl_logic_elements:
        tl_id = tl_logic_element.attrib.get("id")
        if tl_id is None:
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

    after_audit = _tls_phase_sync_audit(tl_logic_elements, link_infos_by_tls)
    if patched_tls_ids:
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)

    return {
        "patched_tls_count": len(patched_tls_ids),
        "patched_tls_ids": patched_tls_ids,
        "approach_synchronized_tls_count": len(patched_tls_ids),
        "skipped_single_axis_tls_count": len(skipped_single_axis_tls_ids),
        "max_phase_count": max_phase_count,
        "mixed_same_incoming_lane_phase_count_before": before_audit["mixed_same_incoming_lane_phase_count"],
        "mixed_same_incoming_lane_phase_count_after": after_audit["mixed_same_incoming_lane_phase_count"],
        "mixed_same_approach_phase_count_before": before_audit["mixed_same_approach_phase_count"],
        "mixed_same_approach_phase_count_after": after_audit["mixed_same_approach_phase_count"],
        "phase_sync_examples_before": before_audit["examples"],
        "phase_sync_examples_after": after_audit["examples"],
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
