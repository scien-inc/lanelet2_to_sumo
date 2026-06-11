from __future__ import annotations

import argparse
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ll2sumo.georeference import patch_net_location
from ll2sumo.geometry import (
    angle_diff_deg,
    average_polyline_many,
    distance_2d,
    first_nonzero_segment as _first_nonzero_segment,
    heading_deg,
    last_nonzero_segment as _last_nonzero_segment,
    midpoint,
    parallel_polylines_compatible,
    point_along_direction as _point_along_direction,
    polyline_length,
    signed_lateral_offset,
)
from ll2sumo import net_postprocess
from ll2sumo.lane_change import analyze_lane_changes, blocked_change_permissions
from ll2sumo.model import GeoReference, Lanelet, LaneletMap, Point3D, RegulatoryElement
from ll2sumo.parser import parse_lanelet_map
from ll2sumo.sumo_xml import (
    id_sort_key as _sort_key,
    shape_string as _shape_string,
    usable_connection_shape as _usable_connection_shape,
)


MIN_EXPORT_EDGE_LENGTH_M = 1.0
MIN_FOLDBACK_CHECK_LENGTH_M = 20.0
MIN_FOLDBACK_CHORD_RATIO = 0.25
MAX_PARALLEL_GROUP_LENGTH_RATIO = 1.35
MAX_PARALLEL_GROUP_MEAN_GAP_M = 8.0
MAX_PARALLEL_GROUP_SAMPLE_GAP_M = 12.0
MAX_PARALLEL_GROUP_SEGMENT_HEADING_DIFF_DEG = 45.0
STOPLINE_ENDPOINT_TOLERANCE_M = 1.0
TANGENT_CONNECTOR_MIN_LENGTH_M = 0.25
TANGENT_CONNECTOR_MAX_LENGTH_M = 3.0
TANGENT_CONNECTOR_LENGTH_RATIO = 0.25
TANGENT_CONNECTOR_STRAIGHT_THRESHOLD_DEG = 20.0
MAX_ENDPOINT_BRIDGE_CONNECTION_LENGTH_M = 50.0
MAX_CONNECTION_SHAPE_EXAMPLES = 20


@dataclass(frozen=True)
class LaneGroup:
    group_id: str
    edge_id: str
    lanelet_paths: tuple[tuple[str, ...], ...]
    start: Point3D
    end: Point3D
    centerline: tuple[Point3D, ...]

    @property
    def ordered_lanelet_ids(self) -> tuple[str, ...]:
        return tuple(path[0] for path in self.lanelet_paths)


@dataclass(frozen=True)
class IntersectionCluster:
    intersection_area_id: str
    lanelet_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    incoming_lanelet_ids: tuple[str, ...]
    outgoing_lanelet_ids: tuple[str, ...]
    movement_lanelet_pairs: tuple[tuple[str, str], ...]
    centroid: Point3D


@dataclass(frozen=True)
class StopLinePlacement:
    endpoint: str
    stop_offset_m: float | None = None


@dataclass(frozen=True)
class ConnectionShape:
    points: tuple[Point3D, ...]
    source: str


@dataclass(frozen=True)
class IntersectionAreaNodeJoin:
    intersection_area_id: str
    join_id: str
    node_ids: tuple[str, ...]
    point: Point3D
    shape: tuple[Point3D, ...]


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item in self.parent:
            return
        self.parent[item] = item
        self.rank[item] = 0

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, lhs: str, rhs: str) -> None:
        lhs_root = self.find(lhs)
        rhs_root = self.find(rhs)
        if lhs_root == rhs_root:
            return
        lhs_rank = self.rank[lhs_root]
        rhs_rank = self.rank[rhs_root]
        if lhs_rank < rhs_rank:
            lhs_root, rhs_root = rhs_root, lhs_root
        self.parent[rhs_root] = lhs_root
        if lhs_rank == rhs_rank:
            self.rank[lhs_root] += 1

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            groups[self.find(item)].append(item)
        return groups


def _speed_mps(lanelet: Lanelet) -> float:
    speed_kph = float(lanelet.tags.get("speed_limit", "50"))
    return speed_kph / 3.6


def _priority_for_group(group_lanelets: list[Lanelet]) -> str:
    mean_speed_kph = sum(float(lanelet.tags.get("speed_limit", "50")) for lanelet in group_lanelets) / len(group_lanelets)
    if mean_speed_kph >= 50.0:
        return "4"
    if mean_speed_kph >= 30.0:
        return "3"
    return "2"


def _point_average(points: list[Point3D]) -> Point3D:
    return Point3D(
        x=sum(point.x for point in points) / len(points),
        y=sum(point.y for point in points) / len(points),
        z=sum(point.z for point in points) / len(points),
    )


def _geo_reference_dict(geo_reference: GeoReference | None, patched_net_location: bool = False) -> dict[str, object] | None:
    if geo_reference is None:
        return None
    return {
        "proj_parameter": geo_reference.proj_parameter,
        "utm_zone": geo_reference.utm_zone,
        "hemisphere": geo_reference.hemisphere,
        "local_to_projected_offset": {
            "x": round(geo_reference.local_to_projected_offset_x, 6),
            "y": round(geo_reference.local_to_projected_offset_y, 6),
        },
        "sumo_net_offset": {
            "x": round(geo_reference.net_offset_x, 6),
            "y": round(geo_reference.net_offset_y, 6),
        },
        "sample_count": geo_reference.sample_count,
        "max_error_m": round(geo_reference.max_error_m, 6),
        "mean_error_m": round(geo_reference.mean_error_m, 6),
        "patched_net_location": patched_net_location,
    }


def _concat_polylines(polylines: list[tuple[Point3D, ...]]) -> tuple[Point3D, ...]:
    merged: list[Point3D] = []
    for polyline in polylines:
        if not polyline:
            continue
        if merged and merged[-1] == polyline[0]:
            merged.extend(polyline[1:])
            continue
        merged.extend(polyline)
    return tuple(merged)


def _intersection_area_id(lanelet: Lanelet) -> str | None:
    return lanelet.tags.get("intersection_area")


def _group_from_lanelet_paths(
    group_id: str,
    edge_id: str,
    lanelet_paths: tuple[tuple[str, ...], ...],
    road_lanelets: dict[str, Lanelet],
) -> LaneGroup:
    path_centerlines = [
        _concat_polylines([road_lanelets[lanelet_id].centerline for lanelet_id in lanelet_path])
        for lanelet_path in lanelet_paths
    ]
    start = _point_average([path_centerline[0] for path_centerline in path_centerlines])
    end = _point_average([path_centerline[-1] for path_centerline in path_centerlines])
    group_centerline = average_polyline_many(path_centerlines)
    group_heading = heading_deg(start, end)
    origin = midpoint(group_centerline)
    ordered_paths = sorted(
        zip(lanelet_paths, path_centerlines),
        key=lambda item: (
            signed_lateral_offset(midpoint(item[1]), origin, group_heading),
            _sort_key(item[0][0]),
        ),
    )
    return LaneGroup(
        group_id=group_id,
        edge_id=edge_id,
        lanelet_paths=tuple(path for path, _ in ordered_paths),
        start=start,
        end=end,
        centerline=group_centerline,
    )


def _successor_candidates(road_lanelets: dict[str, Lanelet]) -> dict[str, list[str]]:
    starts_by_endpoint: dict[tuple[str, str], list[str]] = defaultdict(list)
    for lanelet in road_lanelets.values():
        starts_by_endpoint[(lanelet.left_node_ids[0], lanelet.right_node_ids[0])].append(lanelet.id)

    successors: dict[str, set[str]] = defaultdict(set)
    for lanelet in road_lanelets.values():
        end_key = (lanelet.left_node_ids[-1], lanelet.right_node_ids[-1])
        for candidate_id in starts_by_endpoint.get(end_key, []):
            if candidate_id == lanelet.id:
                continue
            successors[lanelet.id].add(candidate_id)

    return {lanelet_id: sorted(candidate_ids, key=_sort_key) for lanelet_id, candidate_ids in successors.items()}


def _build_lane_groups(road_lanelets: dict[str, Lanelet], lane_change_analysis) -> tuple[list[LaneGroup], dict[str, str]]:
    components = DisjointSet()
    for lanelet_id in road_lanelets:
        components.add(lanelet_id)
    for neighbor in lane_change_analysis.neighbors:
        if not neighbor.bundle_eligible:
            continue
        components.union(neighbor.left_lanelet_id, neighbor.right_lanelet_id)

    lanelet_to_group: dict[str, str] = {}
    lane_groups: list[LaneGroup] = []
    for group_index, lanelet_ids in enumerate(sorted(components.components().values(), key=lambda ids: [_sort_key(value) for value in ids])):
        group = _group_from_lanelet_paths(
            group_id=f"group_{group_index}",
            edge_id=f"edge_{group_index}",
            lanelet_paths=tuple((lanelet_id,) for lanelet_id in sorted(lanelet_ids, key=_sort_key)),
            road_lanelets=road_lanelets,
        )
        lane_groups.append(group)
        for lanelet_id in group.ordered_lanelet_ids:
            lanelet_to_group[lanelet_id] = group.group_id

    return lane_groups, lanelet_to_group


def _rebuild_lanelet_to_group(lane_groups: list[LaneGroup]) -> dict[str, str]:
    lanelet_to_group: dict[str, str] = {}
    for group in lane_groups:
        for lanelet_path in group.lanelet_paths:
            for lanelet_id in lanelet_path:
                lanelet_to_group[lanelet_id] = group.group_id
    return lanelet_to_group


def _build_predecessors(successors: dict[str, list[str]]) -> dict[str, list[str]]:
    predecessors: dict[str, list[str]] = defaultdict(list)
    for source_lanelet_id, target_lanelet_ids in successors.items():
        for target_lanelet_id in target_lanelet_ids:
            predecessors[target_lanelet_id].append(source_lanelet_id)
    return {lanelet_id: sorted(source_lanelet_ids, key=_sort_key) for lanelet_id, source_lanelet_ids in predecessors.items()}


def _lane_change_signature(
    lanelet_id: str,
    lanelet_ids: tuple[str, ...],
    lane_change_analysis,
) -> tuple[str, str]:
    lane_index = lanelet_ids.index(lanelet_id)
    left_neighbor = lanelet_ids[lane_index - 1] if lane_index > 0 else None
    right_neighbor = lanelet_ids[lane_index + 1] if lane_index + 1 < len(lanelet_ids) else None

    def status(target_lanelet_id: str | None) -> str:
        if target_lanelet_id is None:
            return "none"
        decision = lane_change_analysis.decisions.get((lanelet_id, target_lanelet_id))
        if decision is None:
            return "blocked:no_shared_boundary"
        return ("allowed:" if decision.allowed else "blocked:") + decision.reason

    return status(left_neighbor), status(right_neighbor)


def _lane_change_state(signature: str) -> str:
    if signature == "none":
        return "none"
    if signature.startswith("allowed:"):
        return "allowed"
    return "blocked"


def _lane_change_export_signature(signature: tuple[str, str]) -> tuple[str, str]:
    return tuple(_lane_change_state(status) for status in signature)


def _target_is_short_restrictive_tail(
    source_group: LaneGroup,
    target_group: LaneGroup,
    source_lanelet_ids: tuple[str, ...],
    target_lanelet_ids: tuple[str, ...],
    lane_change_analysis,
    max_length_m: float = 5.0,
) -> bool:
    if polyline_length(target_group.centerline) > max_length_m:
        return False

    for source_lanelet_id, target_lanelet_id in zip(source_lanelet_ids, target_lanelet_ids):
        source_signature = _lane_change_signature(source_lanelet_id, source_lanelet_ids, lane_change_analysis)
        target_signature = _lane_change_signature(target_lanelet_id, target_lanelet_ids, lane_change_analysis)
        for source_status, target_status in zip(source_signature, target_signature):
            source_state = _lane_change_state(source_status)
            target_state = _lane_change_state(target_status)
            if source_state == target_state:
                continue
            if source_state == "allowed" and target_state == "blocked":
                continue
            return False

    return True


def _lanelet_path_centerline(lanelet_path: tuple[str, ...], road_lanelets: dict[str, Lanelet]) -> tuple[Point3D, ...]:
    return _concat_polylines([road_lanelets[lanelet_id].centerline for lanelet_id in lanelet_path])


def _would_create_foldback_edge(
    source_group: LaneGroup,
    target_group: LaneGroup,
    road_lanelets: dict[str, Lanelet],
) -> bool:
    for source_lanelet_path, target_lanelet_path in zip(source_group.lanelet_paths, target_group.lanelet_paths):
        merged_centerline = _lanelet_path_centerline(source_lanelet_path + target_lanelet_path, road_lanelets)
        length = polyline_length(merged_centerline)
        if length < MIN_FOLDBACK_CHECK_LENGTH_M:
            continue
        chord = distance_2d(merged_centerline[0], merged_centerline[-1])
        if chord / length < MIN_FOLDBACK_CHORD_RATIO:
            return True
    return False


def _blocks_serial_merge_as_source(lanelet: Lanelet, lanelet_map: LaneletMap) -> bool:
    return bool(lanelet.tags.get("intersection_area")) or "end" in _traffic_light_stopline_endpoints(
        lanelet_map,
        lanelet,
    )


def _blocks_serial_merge_as_target(lanelet: Lanelet, lanelet_map: LaneletMap) -> bool:
    return bool(lanelet.tags.get("intersection_area")) or "start" in _traffic_light_stopline_endpoints(
        lanelet_map,
        lanelet,
    )


def _groups_are_mergeable(
    source_group: LaneGroup,
    target_group: LaneGroup,
    road_lanelets: dict[str, Lanelet],
    lanelet_map: LaneletMap,
    lanelet_to_group: dict[str, str],
    successors: dict[str, list[str]],
    predecessors: dict[str, list[str]],
    lane_change_analysis,
) -> bool:
    if len(source_group.lanelet_paths) != len(target_group.lanelet_paths):
        return False

    source_lanelet_ids = tuple(path[-1] for path in source_group.lanelet_paths)
    target_lanelet_ids = tuple(path[0] for path in target_group.lanelet_paths)

    for source_lanelet_id, target_lanelet_id in zip(source_lanelet_ids, target_lanelet_ids):
        if _blocks_serial_merge_as_source(road_lanelets[source_lanelet_id], lanelet_map):
            return False
        if _blocks_serial_merge_as_target(road_lanelets[target_lanelet_id], lanelet_map):
            return False

    source_successor_groups = {
        lanelet_to_group[target_lanelet_id]
        for source_lanelet_id in source_lanelet_ids
        for target_lanelet_id in successors.get(source_lanelet_id, [])
        if lanelet_to_group[target_lanelet_id] != source_group.group_id
    }
    target_predecessor_groups = {
        lanelet_to_group[source_lanelet_id]
        for target_lanelet_id in target_lanelet_ids
        for source_lanelet_id in predecessors.get(target_lanelet_id, [])
        if lanelet_to_group[source_lanelet_id] != target_group.group_id
    }
    if source_successor_groups != {target_group.group_id} or target_predecessor_groups != {source_group.group_id}:
        return False
    if _would_create_foldback_edge(source_group, target_group, road_lanelets):
        return False

    for source_lanelet_id, target_lanelet_id in zip(source_lanelet_ids, target_lanelet_ids):
        outgoing_lanelets = [
            candidate_id
            for candidate_id in successors.get(source_lanelet_id, [])
            if lanelet_to_group[candidate_id] == target_group.group_id
        ]
        incoming_lanelets = [
            candidate_id
            for candidate_id in predecessors.get(target_lanelet_id, [])
            if lanelet_to_group[candidate_id] == source_group.group_id
        ]
        if outgoing_lanelets != [target_lanelet_id] or incoming_lanelets != [source_lanelet_id]:
            return False
        source_signature = _lane_change_signature(source_lanelet_id, source_lanelet_ids, lane_change_analysis)
        target_signature = _lane_change_signature(
            target_lanelet_id,
            target_lanelet_ids,
            lane_change_analysis,
        )
        if source_signature == target_signature:
            continue
        if _lane_change_export_signature(source_signature) == _lane_change_export_signature(target_signature):
            continue
        if _target_is_short_restrictive_tail(
            source_group,
            target_group,
            source_lanelet_ids,
            target_lanelet_ids,
            lane_change_analysis,
        ):
            continue
        return False

    return True


def _merge_two_groups(source_group: LaneGroup, target_group: LaneGroup, road_lanelets: dict[str, Lanelet]) -> LaneGroup:
    merged_paths = tuple(
        source_lanelet_path + target_lanelet_path
        for source_lanelet_path, target_lanelet_path in zip(source_group.lanelet_paths, target_group.lanelet_paths)
    )
    return _group_from_lanelet_paths(
        group_id=source_group.group_id,
        edge_id=source_group.edge_id,
        lanelet_paths=merged_paths,
        road_lanelets=road_lanelets,
    )


def _group_contains_intersection_area(group: LaneGroup, road_lanelets: dict[str, Lanelet]) -> bool:
    return any(_intersection_area_id(road_lanelets[lanelet_id]) for lanelet_path in group.lanelet_paths for lanelet_id in lanelet_path)


def _collapsed_intersection_group_area_ids(
    lane_groups: list[LaneGroup],
    road_lanelets: dict[str, Lanelet],
    excluded_area_ids: set[str] | None = None,
) -> dict[str, str]:
    excluded_area_ids = excluded_area_ids or set()
    collapsed_group_area_ids: dict[str, str] = {}
    for group in lane_groups:
        area_ids = {
            _intersection_area_id(road_lanelets[lanelet_id])
            for lanelet_path in group.lanelet_paths
            for lanelet_id in lanelet_path
        }
        if len(area_ids) != 1 or None in area_ids:
            continue
        intersection_area_id = next(iter(area_ids))
        if intersection_area_id in excluded_area_ids:
            continue
        collapsed_group_area_ids[group.group_id] = intersection_area_id
    return collapsed_group_area_ids


def _reachable_intersection_area_signature(
    lanelet_ids: tuple[str, ...],
    road_lanelets: dict[str, Lanelet],
    successors: dict[str, list[str]],
) -> tuple[str, ...]:
    reachable_area_ids: set[str] = set()
    stack = list(lanelet_ids)
    visited: set[str] = set()

    while stack:
        lanelet_id = stack.pop()
        if lanelet_id in visited:
            continue
        visited.add(lanelet_id)
        for target_lanelet_id in successors.get(lanelet_id, []):
            target_area_id = _intersection_area_id(road_lanelets[target_lanelet_id])
            if target_area_id is not None:
                reachable_area_ids.add(target_area_id)
                continue
            stack.append(target_lanelet_id)

    return tuple(sorted(reachable_area_ids, key=_sort_key))


def _traffic_light_regulatory_elements(lanelet_map: LaneletMap) -> dict[str, RegulatoryElement]:
    return {
        regulatory_element_id: regulatory_element
        for regulatory_element_id, regulatory_element in lanelet_map.regulatory_elements.items()
        if regulatory_element.subtype == "traffic_light"
    }


def _lanelets_by_regulatory_element(lanelet_map: LaneletMap) -> dict[str, list[str]]:
    lanelets_by_regulatory_element: dict[str, list[str]] = defaultdict(list)
    for lanelet in lanelet_map.lanelets.values():
        for regulatory_element_id in lanelet.regulatory_ids:
            lanelets_by_regulatory_element[regulatory_element_id].append(lanelet.id)
    return {
        regulatory_element_id: sorted(lanelet_ids, key=_sort_key)
        for regulatory_element_id, lanelet_ids in lanelets_by_regulatory_element.items()
    }


def _is_vehicle_traffic_light_regulatory_element(
    lanelet_map: LaneletMap,
    regulatory_element: RegulatoryElement,
) -> bool:
    refers_way_ids = tuple(sorted(regulatory_element.members_by_role.get("refers", ()), key=_sort_key))
    if not refers_way_ids:
        return False
    if {
        lanelet_map.ways[way_id].tags.get("subtype", "")
        for way_id in refers_way_ids
        if way_id in lanelet_map.ways
    } == {"red_green"}:
        return False
    return True


def _point_to_segment_distance_2d(point: Point3D, start: Point3D, end: Point3D) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    segment_length_squared = dx * dx + dy * dy
    if segment_length_squared <= 0.0:
        return distance_2d(point, start)
    t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / segment_length_squared
    t = min(max(t, 0.0), 1.0)
    projected_x = start.x + t * dx
    projected_y = start.y + t * dy
    return math.hypot(point.x - projected_x, point.y - projected_y)


def _project_point_to_polyline_distance_2d(point: Point3D, polyline: tuple[Point3D, ...]) -> tuple[float, float] | None:
    if not polyline:
        return None
    if len(polyline) == 1:
        return distance_2d(point, polyline[0]), 0.0

    best_distance = math.inf
    best_along = 0.0
    cumulative = 0.0
    for start, end in zip(polyline, polyline[1:]):
        dx = end.x - start.x
        dy = end.y - start.y
        segment_length = math.hypot(dx, dy)
        segment_length_squared = dx * dx + dy * dy
        if segment_length_squared <= 0.0:
            distance = distance_2d(point, start)
            along = cumulative
        else:
            t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / segment_length_squared
            t = min(max(t, 0.0), 1.0)
            projected_x = start.x + t * dx
            projected_y = start.y + t * dy
            distance = math.hypot(point.x - projected_x, point.y - projected_y)
            along = cumulative + t * segment_length
        if distance < best_distance:
            best_distance = distance
            best_along = along
        cumulative += segment_length

    return best_distance, best_along


def _linework_sample_points(linework: tuple[tuple[Point3D, ...], ...]) -> tuple[Point3D, ...]:
    sample_points: list[Point3D] = []
    for polyline in linework:
        sample_points.extend(polyline)
        sample_points.extend(midpoint((start, end)) for start, end in zip(polyline, polyline[1:]))
    return tuple(sample_points)


def _linework_projection_distance_on_polyline(
    polyline: tuple[Point3D, ...],
    linework: tuple[tuple[Point3D, ...], ...],
) -> float | None:
    best_distance = math.inf
    best_along: float | None = None
    for point in _linework_sample_points(linework):
        projection = _project_point_to_polyline_distance_2d(point, polyline)
        if projection is None:
            continue
        distance, along = projection
        if distance < best_distance:
            best_distance = distance
            best_along = along
    return best_along


def _point_to_linework_distance_2d(point: Point3D, linework: tuple[tuple[Point3D, ...], ...]) -> float | None:
    distances: list[float] = []
    for polyline in linework:
        if len(polyline) == 1:
            distances.append(distance_2d(point, polyline[0]))
            continue
        distances.extend(
            _point_to_segment_distance_2d(point, start, end)
            for start, end in zip(polyline, polyline[1:])
        )
    if not distances:
        return None
    return min(distances)


def _traffic_light_ref_linework(
    lanelet_map: LaneletMap,
    regulatory_element: RegulatoryElement,
) -> tuple[tuple[Point3D, ...], ...]:
    linework: list[tuple[Point3D, ...]] = []
    for way_id in regulatory_element.members_by_role.get("ref_line", ()):
        way = lanelet_map.ways.get(way_id)
        if way is None:
            continue
        points = tuple(lanelet_map.nodes[node_id] for node_id in way.node_ids if node_id in lanelet_map.nodes)
        if points:
            linework.append(points)
    return tuple(linework)


def _traffic_light_stopline_endpoint(
    lanelet_map: LaneletMap,
    lanelet: Lanelet,
    regulatory_element: RegulatoryElement,
) -> str | None:
    placement = _traffic_light_stopline_placement(lanelet_map, lanelet, regulatory_element)
    if placement is None:
        return None
    return placement.endpoint


def _traffic_light_stopline_placement(
    lanelet_map: LaneletMap,
    lanelet: Lanelet,
    regulatory_element: RegulatoryElement,
) -> StopLinePlacement | None:
    linework = _traffic_light_ref_linework(lanelet_map, regulatory_element)
    if not linework:
        return None
    projection_distance = _linework_projection_distance_on_polyline(lanelet.centerline, linework)
    if projection_distance is not None:
        lanelet_length = polyline_length(lanelet.centerline)
        distance_to_end = max(lanelet_length - projection_distance, 0.0)
        if projection_distance <= STOPLINE_ENDPOINT_TOLERANCE_M:
            return StopLinePlacement(endpoint="start")
        if distance_to_end <= STOPLINE_ENDPOINT_TOLERANCE_M:
            return StopLinePlacement(endpoint="end")
        return StopLinePlacement(endpoint="end", stop_offset_m=distance_to_end)

    start_distance = _point_to_linework_distance_2d(lanelet.start, linework)
    end_distance = _point_to_linework_distance_2d(lanelet.end, linework)
    if start_distance is None or end_distance is None:
        return None
    return StopLinePlacement(endpoint="start" if start_distance <= end_distance else "end")


def _traffic_light_stopline_endpoints(lanelet_map: LaneletMap, lanelet: Lanelet) -> set[str]:
    endpoints: set[str] = set()
    for regulatory_element_id in lanelet.regulatory_ids:
        regulatory_element = lanelet_map.regulatory_elements.get(regulatory_element_id)
        if regulatory_element is None or regulatory_element.subtype != "traffic_light":
            continue
        if not _is_vehicle_traffic_light_regulatory_element(lanelet_map, regulatory_element):
            continue
        endpoint = _traffic_light_stopline_endpoint(lanelet_map, lanelet, regulatory_element)
        endpoints.add(endpoint or "end")
    return endpoints


def _signalized_intersection_area_ids(
    lanelet_map: LaneletMap,
    road_lanelets: dict[str, Lanelet],
    successors: dict[str, list[str]],
) -> set[str]:
    traffic_light_regs = _traffic_light_regulatory_elements(lanelet_map)
    lanelets_by_regulatory_element = _lanelets_by_regulatory_element(lanelet_map)
    intersection_area_ids: set[str] = set()

    for regulatory_element in traffic_light_regs.values():
        if not _is_vehicle_traffic_light_regulatory_element(lanelet_map, regulatory_element):
            continue
        for lanelet_id in lanelets_by_regulatory_element.get(regulatory_element.id, ()):
            if lanelet_id not in road_lanelets:
                continue
            intersection_area_id = _intersection_area_id(road_lanelets[lanelet_id])
            if intersection_area_id is not None:
                intersection_area_ids.add(intersection_area_id)
                continue
            intersection_area_ids.update(
                _reachable_intersection_area_signature((lanelet_id,), road_lanelets, successors)
            )

    return intersection_area_ids


def _parallel_compatible_group_clusters(bucket_groups: list[LaneGroup]) -> list[list[LaneGroup]]:
    remaining = sorted(bucket_groups, key=lambda group: _sort_key(group.group_id))
    clusters: list[list[LaneGroup]] = []
    while remaining:
        cluster = [remaining.pop(0)]
        next_remaining: list[LaneGroup] = []
        for candidate in remaining:
            if all(
                parallel_polylines_compatible(
                    candidate.centerline,
                    member.centerline,
                    max_length_ratio=MAX_PARALLEL_GROUP_LENGTH_RATIO,
                    max_mean_gap_m=MAX_PARALLEL_GROUP_MEAN_GAP_M,
                    max_sample_gap_m=MAX_PARALLEL_GROUP_SAMPLE_GAP_M,
                    max_segment_heading_diff_deg=MAX_PARALLEL_GROUP_SEGMENT_HEADING_DIFF_DEG,
                )
                for member in cluster
            ):
                cluster.append(candidate)
                continue
            next_remaining.append(candidate)
        clusters.append(cluster)
        remaining = next_remaining
    return clusters


def _merge_parallel_lane_groups(
    lane_groups: list[LaneGroup],
    road_lanelets: dict[str, Lanelet],
    lanelet_to_group: dict[str, str],
    successors: dict[str, list[str]],
) -> tuple[list[LaneGroup], int]:
    provisional_node_ids, _ = _assign_node_ids(lane_groups, lanelet_to_group, successors, {})
    predecessors = _build_predecessors(successors)
    buckets: dict[tuple[str, str], list[LaneGroup]] = defaultdict(list)
    for group in lane_groups:
        if _group_contains_intersection_area(group, road_lanelets):
            continue
        start_lanelet_ids = tuple(lanelet_path[0] for lanelet_path in group.lanelet_paths)
        end_lanelet_ids = tuple(lanelet_path[-1] for lanelet_path in group.lanelet_paths)
        predecessor_signature = tuple(
            sorted(
                {
                    lanelet_to_group[source_lanelet_id]
                    for lanelet_id in start_lanelet_ids
                    for source_lanelet_id in predecessors.get(lanelet_id, [])
                    if lanelet_to_group[source_lanelet_id] != group.group_id
                },
                key=_sort_key,
            )
        )
        downstream_intersection_signature = _reachable_intersection_area_signature(end_lanelet_ids, road_lanelets, successors)
        bucket_key = (
            predecessor_signature if predecessor_signature else (provisional_node_ids[f"{group.group_id}:start"],),
            downstream_intersection_signature if downstream_intersection_signature else (provisional_node_ids[f"{group.group_id}:end"],),
        )
        buckets[bucket_key].append(group)

    merged_group_ids: set[str] = set()
    merged_groups: list[LaneGroup] = []
    merged_count = 0

    for bucket_groups in buckets.values():
        if len(bucket_groups) < 2:
            continue
        for compatible_groups in _parallel_compatible_group_clusters(bucket_groups):
            if len(compatible_groups) < 2:
                continue
            merged_group_ids.update(group.group_id for group in compatible_groups)
            merged_count += len(compatible_groups) - 1
            merged_groups.append(
                _group_from_lanelet_paths(
                    group_id=compatible_groups[0].group_id,
                    edge_id=compatible_groups[0].edge_id,
                    lanelet_paths=tuple(
                        lanelet_path
                        for group in compatible_groups
                        for lanelet_path in group.lanelet_paths
                    ),
                    road_lanelets=road_lanelets,
                )
            )

    next_groups = [group for group in lane_groups if group.group_id not in merged_group_ids]
    next_groups.extend(merged_groups)
    next_groups.sort(key=lambda group: _sort_key(group.group_id))
    return next_groups, merged_count


def _merge_serial_lane_groups(
    lane_groups: list[LaneGroup],
    road_lanelets: dict[str, Lanelet],
    lanelet_map: LaneletMap,
    successors: dict[str, list[str]],
    lane_change_analysis,
) -> tuple[list[LaneGroup], int]:
    merged_count = 0
    current_groups = list(lane_groups)
    predecessors = _build_predecessors(successors)

    while True:
        lanelet_to_group = _rebuild_lanelet_to_group(current_groups)
        group_by_id = {group.group_id: group for group in current_groups}
        candidate_successors: dict[str, str] = {}
        for group in current_groups:
            group_successor_ids = {
                lanelet_to_group[target_lanelet_id]
                for lanelet_path in group.lanelet_paths
                for target_lanelet_id in successors.get(lanelet_path[-1], [])
                if lanelet_to_group[target_lanelet_id] != group.group_id
            }
            if len(group_successor_ids) != 1:
                continue

            target_group_id = next(iter(group_successor_ids))
            target_group = group_by_id[target_group_id]

            if not _groups_are_mergeable(
                group,
                target_group,
                road_lanelets,
                lanelet_map,
                lanelet_to_group,
                successors,
                predecessors,
                lane_change_analysis,
            ):
                continue
            candidate_successors[group.group_id] = target_group_id

        if not candidate_successors:
            return current_groups, merged_count

        candidate_targets = set(candidate_successors.values())
        visited_group_ids: set[str] = set()
        next_groups: list[LaneGroup] = []

        for group in current_groups:
            if group.group_id in visited_group_ids:
                continue
            if group.group_id in candidate_targets:
                continue

            merged_group = group
            visited_group_ids.add(group.group_id)
            current_group_id = group.group_id
            while current_group_id in candidate_successors:
                target_group_id = candidate_successors[current_group_id]
                if target_group_id in visited_group_ids:
                    break
                target_group = group_by_id[target_group_id]
                if _would_create_foldback_edge(merged_group, target_group, road_lanelets):
                    break
                merged_group = _merge_two_groups(merged_group, target_group, road_lanelets)
                visited_group_ids.add(target_group_id)
                merged_count += 1
                current_group_id = target_group_id
            next_groups.append(merged_group)

        for group in current_groups:
            if group.group_id in visited_group_ids:
                continue
            next_groups.append(group)
            visited_group_ids.add(group.group_id)

        current_groups = next_groups


def _assign_node_ids(
    lane_groups: list[LaneGroup],
    lanelet_to_group: dict[str, str],
    successors: dict[str, list[str]],
    intersection_clusters: dict[str, IntersectionCluster],
) -> tuple[dict[str, str], dict[str, Point3D]]:
    endpoints = DisjointSet()
    endpoint_points: dict[str, Point3D] = {}
    group_by_id = {group.group_id: group for group in lane_groups}
    for group in lane_groups:
        start_token = f"{group.group_id}:start"
        end_token = f"{group.group_id}:end"
        endpoints.add(start_token)
        endpoints.add(end_token)
        endpoint_points[start_token] = group.start
        endpoint_points[end_token] = group.end

    collapsed_group_area_ids = {
        group_id: cluster.intersection_area_id
        for cluster in intersection_clusters.values()
        for group_id in cluster.group_ids
    }
    for cluster in intersection_clusters.values():
        cluster_token = f"cluster:{cluster.intersection_area_id}"
        endpoints.add(cluster_token)
        endpoint_points[cluster_token] = cluster.centroid

    def group_start_token(group_id: str) -> str | None:
        if group_id in group_by_id:
            return f"{group_id}:start"
        area_id = collapsed_group_area_ids.get(group_id)
        if area_id is not None:
            return f"cluster:{area_id}"
        return None

    def group_end_token(group_id: str) -> str | None:
        if group_id in group_by_id:
            return f"{group_id}:end"
        area_id = collapsed_group_area_ids.get(group_id)
        if area_id is not None:
            return f"cluster:{area_id}"
        return None

    for source_lanelet_id, target_lanelet_ids in successors.items():
        source_group = lanelet_to_group[source_lanelet_id]
        source_token = group_end_token(source_group)
        if source_token is None:
            continue
        for target_lanelet_id in target_lanelet_ids:
            target_group = lanelet_to_group[target_lanelet_id]
            if source_group == target_group:
                continue
            target_token = group_start_token(target_group)
            if target_token is None:
                continue
            endpoints.union(source_token, target_token)

    for cluster in intersection_clusters.values():
        cluster_token = f"cluster:{cluster.intersection_area_id}"
        for incoming_lanelet_id in cluster.incoming_lanelet_ids:
            incoming_group = lanelet_to_group[incoming_lanelet_id]
            incoming_token = group_end_token(incoming_group)
            if incoming_token is None:
                continue
            endpoints.union(incoming_token, cluster_token)
        for outgoing_lanelet_id in cluster.outgoing_lanelet_ids:
            outgoing_group = lanelet_to_group[outgoing_lanelet_id]
            outgoing_token = group_start_token(outgoing_group)
            if outgoing_token is None:
                continue
            endpoints.union(outgoing_token, cluster_token)

    node_ids: dict[str, str] = {}
    node_points: dict[str, Point3D] = {}
    for node_index, (_, members) in enumerate(sorted(endpoints.components().items(), key=lambda item: [_sort_key(member) for member in item[1]])):
        node_id = f"node_{node_index}"
        averaged = _point_average([endpoint_points[member] for member in members])
        node_points[node_id] = averaged
        for member in members:
            node_ids[member] = node_id

    return node_ids, node_points


def _collect_reachable_outgoing_lanelets(
    start_lanelet_id: str,
    internal_lanelet_ids: set[str],
    successors: dict[str, list[str]],
) -> set[str]:
    reachable_outgoing: set[str] = set()
    stack = [start_lanelet_id]
    visited: set[str] = set()

    while stack:
        current_lanelet_id = stack.pop()
        if current_lanelet_id in visited:
            continue
        visited.add(current_lanelet_id)
        for target_lanelet_id in successors.get(current_lanelet_id, []):
            if target_lanelet_id in internal_lanelet_ids:
                stack.append(target_lanelet_id)
                continue
            reachable_outgoing.add(target_lanelet_id)

    return reachable_outgoing


def _cluster_centroid(lanelet_ids: list[str], road_lanelets: dict[str, Lanelet]) -> Point3D:
    points = [midpoint(road_lanelets[lanelet_id].centerline) for lanelet_id in lanelet_ids]
    return _point_average(points)


def _shape_xy_string(points: tuple[Point3D, ...]) -> str:
    return " ".join(f"{point.x:.3f},{point.y:.3f}" for point in points)


def _intersection_area_shape(lanelet_map: LaneletMap, intersection_area_id: str) -> tuple[Point3D, ...]:
    area_way = lanelet_map.ways.get(intersection_area_id)
    if area_way is None or area_way.tags.get("type") != "intersection_area":
        return tuple()
    points = tuple(lanelet_map.nodes[node_id] for node_id in area_way.node_ids if node_id in lanelet_map.nodes)
    if len(points) < 3:
        return tuple()
    return points


def _build_intersection_area_node_joins(
    lane_groups: list[LaneGroup],
    road_lanelets: dict[str, Lanelet],
    lanelet_map: LaneletMap,
    node_ids: dict[str, str],
    node_points: dict[str, Point3D],
    signalized_intersection_area_ids: set[str],
) -> tuple[list[IntersectionAreaNodeJoin], dict[str, object]]:
    node_ids_by_area: dict[str, set[str]] = defaultdict(set)
    mixed_group_count = 0

    for group in lane_groups:
        group_area_ids = {
            _intersection_area_id(road_lanelets[lanelet_id])
            for lanelet_path in group.lanelet_paths
            for lanelet_id in lanelet_path
        }
        if len(group_area_ids) != 1 or None in group_area_ids:
            if any(area_id in signalized_intersection_area_ids for area_id in group_area_ids if area_id is not None):
                mixed_group_count += 1
            continue
        intersection_area_id = next(iter(group_area_ids))
        if intersection_area_id not in signalized_intersection_area_ids:
            continue
        start_node_id = node_ids.get(f"{group.group_id}:start")
        end_node_id = node_ids.get(f"{group.group_id}:end")
        if start_node_id is not None:
            node_ids_by_area[intersection_area_id].add(start_node_id)
        if end_node_id is not None:
            node_ids_by_area[intersection_area_id].add(end_node_id)

    joins: list[IntersectionAreaNodeJoin] = []
    skipped_missing_shape_area_ids: list[str] = []
    skipped_too_few_node_area_ids: list[str] = []
    for intersection_area_id, area_node_ids in sorted(node_ids_by_area.items(), key=lambda item: _sort_key(item[0])):
        if len(area_node_ids) < 2:
            skipped_too_few_node_area_ids.append(intersection_area_id)
            continue
        shape = _intersection_area_shape(lanelet_map, intersection_area_id)
        if not shape:
            skipped_missing_shape_area_ids.append(intersection_area_id)
            continue
        ordered_node_ids = tuple(sorted(area_node_ids, key=_sort_key))
        point = _point_average([node_points[node_id] for node_id in ordered_node_ids if node_id in node_points])
        joins.append(
            IntersectionAreaNodeJoin(
                intersection_area_id=intersection_area_id,
                join_id=f"ia_{intersection_area_id}",
                node_ids=ordered_node_ids,
                point=point,
                shape=shape,
            )
        )

    summary = {
        "join_count": len(joins),
        "joined_intersection_area_ids": [join.intersection_area_id for join in joins],
        "mixed_signalized_group_count": mixed_group_count,
        "skipped_missing_shape_count": len(skipped_missing_shape_area_ids),
        "skipped_missing_shape_area_ids": skipped_missing_shape_area_ids,
        "skipped_too_few_node_count": len(skipped_too_few_node_area_ids),
        "skipped_too_few_node_area_ids": skipped_too_few_node_area_ids,
    }
    return joins, summary


def _build_intersection_clusters(
    road_lanelets: dict[str, Lanelet],
    successors: dict[str, list[str]],
    lanelet_to_group: dict[str, str],
    collapsed_group_area_ids: dict[str, str],
) -> dict[str, IntersectionCluster]:
    predecessors = _build_predecessors(successors)
    lanelet_ids_by_area: dict[str, list[str]] = defaultdict(list)
    for lanelet_id, lanelet in road_lanelets.items():
        intersection_area_id = collapsed_group_area_ids.get(lanelet_to_group[lanelet_id])
        if not intersection_area_id:
            continue
        lanelet_ids_by_area[intersection_area_id].append(lanelet.id)

    clusters: dict[str, IntersectionCluster] = {}
    for intersection_area_id, lanelet_ids in sorted(lanelet_ids_by_area.items(), key=lambda item: _sort_key(item[0])):
        internal_lanelet_ids = set(lanelet_ids)
        incoming_lanelet_ids: set[str] = set()
        outgoing_lanelet_ids: set[str] = set()
        movement_lanelet_pairs: set[tuple[str, str]] = set()

        for lanelet_id in lanelet_ids:
            for predecessor_lanelet_id in predecessors.get(lanelet_id, []):
                if predecessor_lanelet_id in internal_lanelet_ids:
                    continue
                incoming_lanelet_ids.add(predecessor_lanelet_id)
                for outgoing_lanelet_id in _collect_reachable_outgoing_lanelets(
                    lanelet_id,
                    internal_lanelet_ids,
                    successors,
                ):
                    outgoing_lanelet_ids.add(outgoing_lanelet_id)
                    movement_lanelet_pairs.add((predecessor_lanelet_id, outgoing_lanelet_id))

            for successor_lanelet_id in successors.get(lanelet_id, []):
                if successor_lanelet_id in internal_lanelet_ids:
                    continue
                outgoing_lanelet_ids.add(successor_lanelet_id)

        group_ids = sorted({lanelet_to_group[lanelet_id] for lanelet_id in lanelet_ids}, key=_sort_key)
        clusters[intersection_area_id] = IntersectionCluster(
            intersection_area_id=intersection_area_id,
            lanelet_ids=tuple(sorted(lanelet_ids, key=_sort_key)),
            group_ids=tuple(group_ids),
            incoming_lanelet_ids=tuple(sorted(incoming_lanelet_ids, key=_sort_key)),
            outgoing_lanelet_ids=tuple(sorted(outgoing_lanelet_ids, key=_sort_key)),
            movement_lanelet_pairs=tuple(sorted(movement_lanelet_pairs, key=lambda pair: (_sort_key(pair[0]), _sort_key(pair[1])))),
            centroid=_cluster_centroid(lanelet_ids, road_lanelets),
        )

    return clusters


def _write_nodes_xml(
    path: Path,
    node_points: dict[str, Point3D],
    tls_ids_by_node_id: dict[str, str] | None = None,
    intersection_area_node_joins: list[IntersectionAreaNodeJoin] | None = None,
) -> None:
    tls_ids_by_node_id = tls_ids_by_node_id or {}
    intersection_area_node_joins = intersection_area_node_joins or []
    root = ET.Element("nodes")
    for node_id, point in sorted(node_points.items(), key=lambda item: _sort_key(item[0])):
        node_attributes = {
            "id": node_id,
            "x": f"{point.x:.3f}",
            "y": f"{point.y:.3f}",
            "z": f"{point.z:.3f}",
        }
        if node_id in tls_ids_by_node_id:
            node_attributes["type"] = "traffic_light"
            node_attributes["tl"] = tls_ids_by_node_id[node_id]
            node_attributes["tlType"] = "static"
        ET.SubElement(root, "node", node_attributes)
    for node_join in sorted(intersection_area_node_joins, key=lambda join: _sort_key(join.join_id)):
        ET.SubElement(
            root,
            "join",
            {
                "id": node_join.join_id,
                "nodes": " ".join(node_join.node_ids),
                "x": f"{node_join.point.x:.3f}",
                "y": f"{node_join.point.y:.3f}",
                "z": f"{node_join.point.z:.3f}",
                "shape": _shape_xy_string(node_join.shape),
            },
        )
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _lane_change_entry(
    lanelet_path: tuple[str, ...],
    direction: str,
    neighbor_lanelet_path: tuple[str, ...] | None,
    decisions: dict[tuple[str, str], object],
    lanelet_map: LaneletMap,
    road_lanelets: dict[str, Lanelet],
) -> dict[str, object]:
    representative_lanelet = road_lanelets[lanelet_path[0]]
    if neighbor_lanelet_path is None:
        return {"status": "none"}
    decision_sequence = [decisions.get((lanelet_id, neighbor_lanelet_id)) for lanelet_id, neighbor_lanelet_id in zip(lanelet_path, neighbor_lanelet_path)]
    if any(decision is None for decision in decision_sequence):
        return {
            "status": "blocked",
            "target_lanelet_id": neighbor_lanelet_path[0],
            "reason": "no_shared_boundary",
            "source": "grouping",
            "direction": direction,
        }
    decision = next((candidate for candidate in decision_sequence if candidate is not None and not candidate.allowed), decision_sequence[0])
    boundary_tags = lanelet_map.ways[decision.boundary_id].tags
    return {
        "status": "allowed" if all(candidate.allowed for candidate in decision_sequence if candidate is not None) else "blocked",
        "target_lanelet_id": neighbor_lanelet_path[0],
        "boundary_id": decision.boundary_id,
        "reason": decision.reason,
        "source": decision.source,
        "boundary_tags": {
            key: value
            for key, value in boundary_tags.items()
            if key in {"type", "subtype", "lane_change", "lane_change:left", "lane_change:right"}
        },
        "lanelet_tags": {key: value for key, value in representative_lanelet.tags.items() if key.startswith("lane_change")},
        "direction": direction,
    }


def _lane_change_attributes(
    change_left: dict[str, object],
    change_right: dict[str, object],
    lane_change_mode: str,
) -> dict[str, str]:
    if lane_change_mode == "unrestricted":
        return {}

    lane_attributes: dict[str, str] = {}
    if change_left["status"] == "blocked":
        lane_attributes["changeLeft"] = blocked_change_permissions()
    if change_right["status"] == "blocked":
        lane_attributes["changeRight"] = blocked_change_permissions()
    return lane_attributes


def _lanelet_path_signal_stop_offset_m(
    lanelet_path: tuple[str, ...],
    lanelet_map: LaneletMap,
    road_lanelets: dict[str, Lanelet],
) -> float | None:
    lanelet_lengths = [polyline_length(road_lanelets[lanelet_id].centerline) for lanelet_id in lanelet_path]
    remaining_lengths_after: list[float] = []
    remaining_length = 0.0
    for length in reversed(lanelet_lengths):
        remaining_lengths_after.append(remaining_length)
        remaining_length += length
    remaining_lengths_after.reverse()

    offsets: list[float] = []
    for lanelet_id, remaining_after in zip(lanelet_path, remaining_lengths_after):
        lanelet = road_lanelets[lanelet_id]
        for regulatory_element_id in lanelet.regulatory_ids:
            regulatory_element = lanelet_map.regulatory_elements.get(regulatory_element_id)
            if regulatory_element is None or regulatory_element.subtype != "traffic_light":
                continue
            if not _is_vehicle_traffic_light_regulatory_element(lanelet_map, regulatory_element):
                continue
            placement = _traffic_light_stopline_placement(lanelet_map, lanelet, regulatory_element)
            if placement is None or placement.endpoint != "end" or placement.stop_offset_m is None:
                continue
            offsets.append(remaining_after + placement.stop_offset_m)

    if not offsets:
        return None
    path_length = sum(lanelet_lengths)
    stop_offset = max(offsets)
    if stop_offset <= 0.0:
        return None
    return min(stop_offset, max(path_length - 0.1, 0.0))


def _write_edges_xml(
    path: Path,
    lane_groups: list[LaneGroup],
    road_lanelets: dict[str, Lanelet],
    lanelet_map: LaneletMap,
    node_ids: dict[str, str],
    lanelet_to_group: dict[str, str],
    lane_change_analysis,
    lane_change_mode: str,
    sidecar: dict[str, object],
) -> dict[str, int]:
    root = ET.Element("edges")
    lanelet_to_lane_index: dict[str, int] = {}
    edge_id_by_group: dict[str, str] = {}

    sidecar_lineage: dict[str, object] = {}

    for group in lane_groups:
        group_lanelets = [road_lanelets[lanelet_path[0]] for lanelet_path in group.lanelet_paths]
        from_node = node_ids[f"{group.group_id}:start"]
        to_node = node_ids[f"{group.group_id}:end"]
        edge_id_by_group[group.group_id] = group.edge_id
        edge_element = ET.SubElement(
            root,
            "edge",
            {
                "id": group.edge_id,
                "from": from_node,
                "to": to_node,
                "numLanes": str(len(group_lanelets)),
                "speed": f"{sum(_speed_mps(lanelet) for lanelet in group_lanelets) / len(group_lanelets):.3f}",
                "priority": _priority_for_group(group_lanelets),
                "spreadType": "center",
                "shape": _shape_string(group.centerline),
            },
        )
        for lane_index, lanelet_path in enumerate(group.lanelet_paths):
            representative_lanelet = road_lanelets[lanelet_path[0]]
            lane_shape = _concat_polylines([road_lanelets[lanelet_id].centerline for lanelet_id in lanelet_path])
            for lanelet_id in lanelet_path:
                lanelet_to_lane_index[lanelet_id] = lane_index

            left_neighbor_path = group.lanelet_paths[lane_index + 1] if lane_index + 1 < len(group.lanelet_paths) else None
            right_neighbor_path = group.lanelet_paths[lane_index - 1] if lane_index > 0 else None

            change_left = _lane_change_entry(
                lanelet_path,
                "left",
                left_neighbor_path,
                lane_change_analysis.decisions,
                lanelet_map,
                road_lanelets,
            )
            change_right = _lane_change_entry(
                lanelet_path,
                "right",
                right_neighbor_path,
                lane_change_analysis.decisions,
                lanelet_map,
                road_lanelets,
            )

            lane_attributes = {
                "index": str(lane_index),
                "speed": f"{sum(_speed_mps(road_lanelets[lanelet_id]) for lanelet_id in lanelet_path) / len(lanelet_path):.3f}",
                "length": f"{polyline_length(lane_shape):.3f}",
                "shape": _shape_string(lane_shape),
            }
            lane_attributes.update(_lane_change_attributes(change_left, change_right, lane_change_mode))
            lane_element = ET.SubElement(edge_element, "lane", lane_attributes)
            traffic_light_stop_offset_m = _lanelet_path_signal_stop_offset_m(
                lanelet_path,
                lanelet_map,
                road_lanelets,
            )
            if traffic_light_stop_offset_m is not None:
                ET.SubElement(lane_element, "stopOffset", {"value": f"{traffic_light_stop_offset_m:.3f}"})

            lane_id = f"{group.edge_id}_{lane_index}"
            sidecar_lineage[lane_id] = {
                "edge_id": group.edge_id,
                "lane_index": lane_index,
                "lanelet_id": representative_lanelet.id,
                "lanelet_ids": list(lanelet_path),
                "changeLeft": change_left,
                "changeRight": change_right,
            }
            if traffic_light_stop_offset_m is not None:
                sidecar_lineage[lane_id]["trafficLightStopOffset"] = round(traffic_light_stop_offset_m, 3)

    sidecar["lane_change_lineage"] = sidecar_lineage
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return lanelet_to_lane_index


def _collapsed_internal_path(
    source_lanelet_id: str,
    target_lanelet_id: str,
    successors: dict[str, list[str]],
    collapsed_internal_lanelet_ids: set[str],
) -> tuple[str, ...]:
    queue: list[tuple[str, tuple[str, ...]]] = [
        (lanelet_id, tuple())
        for lanelet_id in sorted(successors.get(source_lanelet_id, []), key=_sort_key)
    ]
    visited: set[str] = set()
    while queue:
        lanelet_id, path = queue.pop(0)
        if lanelet_id == target_lanelet_id:
            return path
        if lanelet_id in visited:
            continue
        visited.add(lanelet_id)
        if lanelet_id not in collapsed_internal_lanelet_ids:
            continue
        next_path = path + (lanelet_id,)
        for successor_lanelet_id in sorted(successors.get(lanelet_id, []), key=_sort_key):
            queue.append((successor_lanelet_id, next_path))
    return tuple()


def _connection_shape_from_intermediate_lanelets(
    source_lanelet_id: str,
    target_lanelet_id: str,
    intermediate_lanelet_ids: tuple[str, ...],
    road_lanelets: dict[str, Lanelet],
) -> ConnectionShape | None:
    if not intermediate_lanelet_ids:
        return None
    if source_lanelet_id not in road_lanelets or target_lanelet_id not in road_lanelets:
        return None
    points: list[Point3D] = [road_lanelets[source_lanelet_id].end]
    for lanelet_id in intermediate_lanelet_ids:
        lanelet = road_lanelets.get(lanelet_id)
        if lanelet is None:
            return None
        points.extend(lanelet.centerline)
    points.append(road_lanelets[target_lanelet_id].start)
    usable_shape = _usable_connection_shape(points)
    if usable_shape is None:
        return None
    return ConnectionShape(usable_shape, "intersection_area")


def _connection_shape_from_endpoint_bridge(
    source_lanelet_id: str,
    target_lanelet_id: str,
    road_lanelets: dict[str, Lanelet],
) -> ConnectionShape | None:
    source_lanelet = road_lanelets.get(source_lanelet_id)
    target_lanelet = road_lanelets.get(target_lanelet_id)
    if source_lanelet is None or target_lanelet is None:
        return None
    if distance_2d(source_lanelet.end, target_lanelet.start) > MAX_ENDPOINT_BRIDGE_CONNECTION_LENGTH_M:
        return None
    usable_shape = _usable_connection_shape((source_lanelet.end, target_lanelet.start))
    if usable_shape is None:
        return None
    return ConnectionShape(usable_shape, "endpoint_bridge")


def _connection_shape_from_tangents(
    source_lanelet_id: str,
    target_lanelet_id: str,
    road_lanelets: dict[str, Lanelet],
) -> ConnectionShape | None:
    source_lanelet = road_lanelets.get(source_lanelet_id)
    target_lanelet = road_lanelets.get(target_lanelet_id)
    if source_lanelet is None or target_lanelet is None:
        return None
    start = source_lanelet.end

    incoming_segment = _last_nonzero_segment(source_lanelet.centerline)
    outgoing_segment = _first_nonzero_segment(target_lanelet.centerline)
    source_length = polyline_length(source_lanelet.centerline)
    target_length = polyline_length(target_lanelet.centerline)
    reference_length = min(length for length in (source_length, target_length) if length > 0.0) if source_length > 0.0 or target_length > 0.0 else TANGENT_CONNECTOR_MIN_LENGTH_M
    connector_length = max(
        TANGENT_CONNECTOR_MIN_LENGTH_M,
        min(TANGENT_CONNECTOR_MAX_LENGTH_M, TANGENT_CONNECTOR_LENGTH_RATIO * reference_length),
    )

    outgoing_point = (
        _point_along_direction(start, outgoing_segment[0], outgoing_segment[1], connector_length)
        if outgoing_segment is not None
        else None
    )
    incoming_point = (
        _point_along_direction(start, incoming_segment[0], incoming_segment[1], connector_length * 0.5)
        if incoming_segment is not None
        else None
    )
    if outgoing_point is None and incoming_point is None:
        return None

    if incoming_segment is not None and outgoing_segment is not None and outgoing_point is not None:
        incoming_heading = heading_deg(incoming_segment[0], incoming_segment[1])
        outgoing_heading = heading_deg(outgoing_segment[0], outgoing_segment[1])
        if angle_diff_deg(incoming_heading, outgoing_heading) <= TANGENT_CONNECTOR_STRAIGHT_THRESHOLD_DEG:
            usable_shape = _usable_connection_shape((start, outgoing_point))
            return ConnectionShape(usable_shape, "tangent_fallback") if usable_shape is not None else None
        if incoming_point is not None:
            usable_shape = _usable_connection_shape((start, incoming_point, outgoing_point))
            return ConnectionShape(usable_shape, "tangent_fallback") if usable_shape is not None else None

    fallback_end = outgoing_point or incoming_point
    usable_shape = _usable_connection_shape((start, fallback_end))
    return ConnectionShape(usable_shape, "tangent_fallback") if usable_shape is not None else None


def _connection_shape_for_lanelets(
    source_lanelet_id: str,
    target_lanelet_id: str,
    intermediate_lanelet_ids: tuple[str, ...],
    road_lanelets: dict[str, Lanelet],
) -> ConnectionShape | None:
    return (
        _connection_shape_from_intermediate_lanelets(
            source_lanelet_id,
            target_lanelet_id,
            intermediate_lanelet_ids,
            road_lanelets,
        )
        or _connection_shape_from_endpoint_bridge(source_lanelet_id, target_lanelet_id, road_lanelets)
        or _connection_shape_from_tangents(source_lanelet_id, target_lanelet_id, road_lanelets)
    )


def _write_connections_xml(
    path: Path,
    successors: dict[str, list[str]],
    lanelet_to_group: dict[str, str],
    lane_groups: list[LaneGroup],
    lanelet_to_lane_index: dict[str, int],
    intersection_clusters: dict[str, IntersectionCluster],
    road_lanelets: dict[str, Lanelet] | None = None,
    edge_node_ids_by_edge_id: dict[str, tuple[str, str]] | None = None,
) -> dict[str, object]:
    road_lanelets = road_lanelets or {}
    edge_id_by_group = {group.group_id: group.edge_id for group in lane_groups}
    group_id_by_edge_id = {group.edge_id: group.group_id for group in lane_groups}
    lanelet_path_by_group_lane: dict[tuple[str, int], tuple[str, ...]] = {
        (group.group_id, lane_index): lanelet_path
        for group in lane_groups
        for lane_index, lanelet_path in enumerate(group.lanelet_paths)
    }
    predecessors = _build_predecessors(successors)
    collapsed_internal_lanelet_ids = {
        lanelet_id
        for cluster in intersection_clusters.values()
        for lanelet_id in cluster.lanelet_ids
    }
    seen_connections: set[tuple[str, str, int, int]] = set()
    root = ET.Element("connections")
    summary: dict[str, object] = {
        "connection_count": 0,
        "shaped_connection_count": 0,
        "merged_non_junction_seam_count": 0,
        "intersection_area_shape_count": 0,
        "endpoint_bridge_shape_count": 0,
        "tangent_fallback_shape_count": 0,
        "unshaped_connection_count": 0,
        "examples": [],
    }

    def reachable_exported_targets(source_lanelet_id: str) -> set[str]:
        source_group = lanelet_to_group[source_lanelet_id]
        reachable_targets: set[str] = set()
        stack = list(successors.get(source_lanelet_id, []))
        visited: set[str] = set()

        while stack:
            lanelet_id = stack.pop()
            if lanelet_id in visited:
                continue
            visited.add(lanelet_id)

            target_group = lanelet_to_group[lanelet_id]
            if target_group == source_group:
                stack.extend(successors.get(lanelet_id, []))
                continue

            if target_group in edge_id_by_group and lanelet_id in lanelet_to_lane_index:
                reachable_targets.add(lanelet_id)
                continue

            if lanelet_id in collapsed_internal_lanelet_ids:
                stack.extend(successors.get(lanelet_id, []))
                continue
            stack.extend(successors.get(lanelet_id, []))

        return reachable_targets

    def resolve_exported_predecessors(lanelet_id: str) -> set[str]:
        resolved: set[str] = set()
        stack = [lanelet_id]
        visited: set[str] = set()

        while stack:
            current_lanelet_id = stack.pop()
            if current_lanelet_id in visited:
                continue
            visited.add(current_lanelet_id)

            current_group = lanelet_to_group[current_lanelet_id]
            if current_group in edge_id_by_group and current_lanelet_id in lanelet_to_lane_index:
                resolved.add(current_lanelet_id)
                continue

            if current_lanelet_id in collapsed_internal_lanelet_ids:
                stack.extend(predecessors.get(current_lanelet_id, []))
                continue
            stack.extend(predecessors.get(current_lanelet_id, []))

        return resolved

    def resolve_exported_successors(lanelet_id: str) -> set[str]:
        resolved: set[str] = set()
        stack = [lanelet_id]
        visited: set[str] = set()

        while stack:
            current_lanelet_id = stack.pop()
            if current_lanelet_id in visited:
                continue
            visited.add(current_lanelet_id)

            current_group = lanelet_to_group[current_lanelet_id]
            if current_group in edge_id_by_group and current_lanelet_id in lanelet_to_lane_index:
                resolved.add(current_lanelet_id)
                continue

            if current_lanelet_id in collapsed_internal_lanelet_ids:
                stack.extend(successors.get(current_lanelet_id, []))
                continue
            stack.extend(successors.get(current_lanelet_id, []))

        return resolved

    def record_example(example: dict[str, object]) -> None:
        examples = summary["examples"]
        assert isinstance(examples, list)
        if len(examples) < MAX_CONNECTION_SHAPE_EXAMPLES:
            examples.append(example)

    def exported_connection_lanelet_ids(
        from_edge: str,
        to_edge: str,
        from_lane: int,
        to_lane: int,
        fallback_source_lanelet_id: str,
        fallback_target_lanelet_id: str,
    ) -> tuple[str, str]:
        source_group = group_id_by_edge_id.get(from_edge)
        target_group = group_id_by_edge_id.get(to_edge)
        source_path = (
            lanelet_path_by_group_lane.get((source_group, from_lane))
            if source_group is not None
            else None
        )
        target_path = (
            lanelet_path_by_group_lane.get((target_group, to_lane))
            if target_group is not None
            else None
        )
        source_lanelet_id = source_path[-1] if source_path else fallback_source_lanelet_id
        target_lanelet_id = target_path[0] if target_path else fallback_target_lanelet_id
        return source_lanelet_id, target_lanelet_id

    def append_connection(
        from_edge: str,
        to_edge: str,
        from_lane: int,
        to_lane: int,
        source_lanelet_id: str,
        target_lanelet_id: str,
        intermediate_lanelet_ids: tuple[str, ...] = tuple(),
    ) -> None:
        if edge_node_ids_by_edge_id is not None:
            from_edge_nodes = edge_node_ids_by_edge_id.get(from_edge)
            to_edge_nodes = edge_node_ids_by_edge_id.get(to_edge)
            if from_edge_nodes is None or to_edge_nodes is None or from_edge_nodes[1] != to_edge_nodes[0]:
                return
        connection = (from_edge, to_edge, from_lane, to_lane)
        if connection in seen_connections:
            return
        seen_connections.add(connection)
        attributes = {
            "from": from_edge,
            "to": to_edge,
            "fromLane": str(from_lane),
            "toLane": str(to_lane),
        }
        shape_source_lanelet_id, shape_target_lanelet_id = exported_connection_lanelet_ids(
            from_edge,
            to_edge,
            from_lane,
            to_lane,
            source_lanelet_id,
            target_lanelet_id,
        )
        shape_intermediate_lanelet_ids = _collapsed_internal_path(
            shape_source_lanelet_id,
            shape_target_lanelet_id,
            successors,
            collapsed_internal_lanelet_ids,
        ) or intermediate_lanelet_ids
        connection_shape = _connection_shape_for_lanelets(
            shape_source_lanelet_id,
            shape_target_lanelet_id,
            shape_intermediate_lanelet_ids,
            road_lanelets,
        )
        if connection_shape is None:
            summary["unshaped_connection_count"] = int(summary["unshaped_connection_count"]) + 1
            record_example(
                {
                    "from": from_edge,
                    "to": to_edge,
                    "fromLane": from_lane,
                    "toLane": to_lane,
                    "source_lanelet_id": shape_source_lanelet_id,
                    "target_lanelet_id": shape_target_lanelet_id,
                    "original_source_lanelet_id": source_lanelet_id,
                    "original_target_lanelet_id": target_lanelet_id,
                    "source": "unshaped",
                }
            )
        else:
            attributes["shape"] = _shape_string(connection_shape.points)
            attributes["length"] = f"{polyline_length(connection_shape.points):.3f}"
            summary["shaped_connection_count"] = int(summary["shaped_connection_count"]) + 1
            summary[f"{connection_shape.source}_shape_count"] = int(summary[f"{connection_shape.source}_shape_count"]) + 1
            record_example(
                {
                    "from": from_edge,
                    "to": to_edge,
                    "fromLane": from_lane,
                    "toLane": to_lane,
                    "source_lanelet_id": shape_source_lanelet_id,
                    "target_lanelet_id": shape_target_lanelet_id,
                    "original_source_lanelet_id": source_lanelet_id,
                    "original_target_lanelet_id": target_lanelet_id,
                    "source": connection_shape.source,
                    "length_m": round(polyline_length(connection_shape.points), 6),
                }
            )
        ET.SubElement(root, "connection", attributes)

    for source_lanelet_id, _ in sorted(successors.items(), key=lambda item: _sort_key(item[0])):
        source_group = lanelet_to_group[source_lanelet_id]
        if source_group not in edge_id_by_group or source_lanelet_id not in lanelet_to_lane_index:
            continue
        source_edge = edge_id_by_group[source_group]
        source_lane = lanelet_to_lane_index[source_lanelet_id]
        for target_lanelet_id in sorted(reachable_exported_targets(source_lanelet_id), key=_sort_key):
            target_group = lanelet_to_group[target_lanelet_id]
            if source_group == target_group:
                continue
            if target_group not in edge_id_by_group or target_lanelet_id not in lanelet_to_lane_index:
                continue
            intermediate_lanelet_ids = _collapsed_internal_path(
                source_lanelet_id,
                target_lanelet_id,
                successors,
                collapsed_internal_lanelet_ids,
            )
            append_connection(
                source_edge,
                edge_id_by_group[target_group],
                source_lane,
                lanelet_to_lane_index[target_lanelet_id],
                source_lanelet_id,
                target_lanelet_id,
                intermediate_lanelet_ids,
            )

    for cluster in intersection_clusters.values():
        for source_lanelet_id, target_lanelet_id in cluster.movement_lanelet_pairs:
            for resolved_source_lanelet_id in sorted(resolve_exported_predecessors(source_lanelet_id), key=_sort_key):
                source_group = lanelet_to_group[resolved_source_lanelet_id]
                source_edge = edge_id_by_group[source_group]
                source_lane = lanelet_to_lane_index[resolved_source_lanelet_id]
                for resolved_target_lanelet_id in sorted(resolve_exported_successors(target_lanelet_id), key=_sort_key):
                    target_group = lanelet_to_group[resolved_target_lanelet_id]
                    if source_group == target_group:
                        continue
                    intermediate_lanelet_ids = _collapsed_internal_path(
                        resolved_source_lanelet_id,
                        resolved_target_lanelet_id,
                        successors,
                        collapsed_internal_lanelet_ids,
                    ) or _collapsed_internal_path(
                        source_lanelet_id,
                        target_lanelet_id,
                        successors,
                        collapsed_internal_lanelet_ids,
                    )
                    append_connection(
                        source_edge,
                        edge_id_by_group[target_group],
                        source_lane,
                        lanelet_to_lane_index[resolved_target_lanelet_id],
                        resolved_source_lanelet_id,
                        resolved_target_lanelet_id,
                        intermediate_lanelet_ids,
                    )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    summary["connection_count"] = len(seen_connections)
    return summary


def _plan_vehicle_signals(
    lanelet_map: LaneletMap,
    road_lanelets: dict[str, Lanelet],
    successors: dict[str, list[str]],
    intersection_clusters: dict[str, IntersectionCluster],
    node_ids: dict[str, str],
    lanelet_to_group: dict[str, str],
) -> tuple[dict[str, str], dict[str, int], list[dict[str, object]]]:
    traffic_light_regs = _traffic_light_regulatory_elements(lanelet_map)
    lanelets_by_regulatory_element = _lanelets_by_regulatory_element(lanelet_map)

    control_buckets: dict[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], list[str]] = defaultdict(list)
    head_keys: set[tuple[str, ...]] = set()
    for regulatory_element in traffic_light_regs.values():
        if not _is_vehicle_traffic_light_regulatory_element(lanelet_map, regulatory_element):
            continue
        refers_way_ids = tuple(sorted(regulatory_element.members_by_role.get("refers", ()), key=_sort_key))
        head_keys.add(refers_way_ids)
        ref_line_way_ids = tuple(sorted(regulatory_element.members_by_role.get("ref_line", ()), key=_sort_key))
        attached_lanelet_ids = tuple(sorted(lanelets_by_regulatory_element.get(regulatory_element.id, ()), key=_sort_key))
        lanelet_key = tuple() if ref_line_way_ids else attached_lanelet_ids
        control_buckets[(refers_way_ids, ref_line_way_ids, lanelet_key)].append(regulatory_element.id)

    unmapped_signals: list[dict[str, object]] = []
    inferred_no_refline_count = 0
    mapped_relation_ids: set[str] = set()
    tls_ids_by_node_id: dict[str, str] = {}

    for control_key, relation_ids in sorted(
        control_buckets.items(),
        key=lambda item: (
            [_sort_key(value) for value in item[0][0]],
            [_sort_key(value) for value in item[0][1]],
            [_sort_key(value) for value in item[0][2]],
        ),
    ):
        _, ref_line_way_ids, _ = control_key
        attached_lanelet_ids = tuple(
            sorted(
                {
                    lanelet_id
                    for relation_id in relation_ids
                    for lanelet_id in lanelets_by_regulatory_element.get(relation_id, ())
                },
                key=_sort_key,
            )
        )
        attached_road_lanelet_ids = tuple(lanelet_id for lanelet_id in attached_lanelet_ids if lanelet_id in road_lanelets)
        if not attached_road_lanelet_ids:
            unmapped_signals.append(
                {
                    "relation_ids": list(relation_ids),
                    "reason": "no_attached_road_lanelets",
                }
            )
            continue

        reachable_area_ids: set[str] = set()
        for lanelet_id in attached_road_lanelet_ids:
            intersection_area_id = _intersection_area_id(road_lanelets[lanelet_id])
            if intersection_area_id is not None:
                reachable_area_ids.add(intersection_area_id)
                continue
            reachable_area_ids.update(_reachable_intersection_area_signature((lanelet_id,), road_lanelets, successors))

        if len(reachable_area_ids) != 1:
            unmapped_signals.append(
                {
                    "relation_ids": list(relation_ids),
                    "reason": "multiple_reachable_intersection_areas" if reachable_area_ids else "no_reachable_intersection_area",
                }
            )
            continue

        intersection_area_id = next(iter(reachable_area_ids))
        candidate_node_ids: set[str] = set()
        for relation_id in relation_ids:
            regulatory_element = traffic_light_regs[relation_id]
            for lanelet_id in attached_road_lanelet_ids:
                if lanelet_id not in lanelets_by_regulatory_element.get(relation_id, ()):
                    continue
                group_id = lanelet_to_group.get(lanelet_id)
                if group_id is None:
                    continue
                endpoint = _traffic_light_stopline_endpoint(
                    lanelet_map,
                    road_lanelets[lanelet_id],
                    regulatory_element,
                )
                stopline_node_key = f"{group_id}:{endpoint or 'end'}"
                if stopline_node_key in node_ids:
                    candidate_node_ids.add(node_ids[stopline_node_key])

        cluster_node_key = f"cluster:{intersection_area_id}"
        if not candidate_node_ids and intersection_area_id in intersection_clusters and cluster_node_key in node_ids:
            candidate_node_ids.add(node_ids[cluster_node_key])

        if not candidate_node_ids:
            unmapped_signals.append(
                {
                    "relation_ids": list(relation_ids),
                    "reason": "stopline_or_cluster_node_not_exported",
                    "intersection_area_id": intersection_area_id,
                }
            )
            continue

        tls_id = f"tls_{intersection_area_id}"
        for node_id in sorted(candidate_node_ids, key=_sort_key):
            tls_ids_by_node_id[node_id] = tls_id
        mapped_relation_ids.update(relation_ids)
        if not ref_line_way_ids:
            inferred_no_refline_count += 1

    vehicle_relation_ids = {
        regulatory_element.id
        for regulatory_element in traffic_light_regs.values()
        if _is_vehicle_traffic_light_regulatory_element(lanelet_map, regulatory_element)
        and tuple(sorted(regulatory_element.members_by_role.get("refers", ()), key=_sort_key)) in head_keys
    }
    signal_summary = {
        "raw_relation_count": len(traffic_light_regs),
        "normalized_head_count": len(head_keys),
        "normalized_control_count": len(control_buckets),
        "signalized_node_count": len(tls_ids_by_node_id),
        "tls_cluster_count": 0,
        "vehicle_tls_link_count": 0,
        "inferred_no_refline_count": inferred_no_refline_count,
        "unmapped_relation_count": len(vehicle_relation_ids - mapped_relation_ids),
    }
    return tls_ids_by_node_id, signal_summary, unmapped_signals


def _run_netconvert(
    nodes_path: Path,
    edges_path: Path,
    connections_path: Path,
    output_path: Path,
    binary: str,
    build_tls_from_nodes: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        binary,
        "--lefthand",
        "--offset.disable-normalization",
        "--no-turnarounds",
        "--junctions.minimal-shape",
        "--precision",
        "3",
        "--precision.geo",
        "8",
        "--node-files",
        str(nodes_path),
        "--edge-files",
        str(edges_path),
        "--connection-files",
        str(connections_path),
        "--output-file",
        str(output_path),
    ]
    if build_tls_from_nodes:
        command.extend(
            [
                "--junctions.join",
                "--tls.discard-simple",
                "--tls.join",
                "--tls.default-type",
                "static",
            ]
        )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"netconvert failed with exit code {result.returncode}:\n{result.stderr}")
    return result


def convert_map(
    input_path: str | Path,
    out_dir: str | Path,
    lane_change_mode: str = "lanelet-infer",
    signal_mode: str = "jp-static",
    run_netconvert: bool = True,
    netconvert_binary: str = "netconvert",
) -> dict[str, object]:
    if lane_change_mode not in {"lanelet-infer", "unrestricted"}:
        raise ValueError(f"Unsupported lane change mode: {lane_change_mode}")
    if signal_mode not in {"none", "jp-static"}:
        raise ValueError(f"Unsupported signal mode: {signal_mode}")

    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lanelet_map = parse_lanelet_map(input_path)
    road_lanelets = {lanelet_id: lanelet for lanelet_id, lanelet in lanelet_map.lanelets.items() if lanelet.subtype == "road"}
    ignored_subtypes = Counter(
        lanelet.subtype for lanelet in lanelet_map.lanelets.values() if lanelet.subtype != "road"
    )

    lane_change_analysis = analyze_lane_changes(lanelet_map)
    successors = _successor_candidates(road_lanelets)
    lane_groups, lanelet_to_group = _build_lane_groups(road_lanelets, lane_change_analysis)
    merged_serial_group_count = 0
    merged_parallel_group_count = 0
    while True:
        lane_groups, serial_count = _merge_serial_lane_groups(
            lane_groups,
            road_lanelets,
            lanelet_map,
            successors,
            lane_change_analysis,
        )
        merged_serial_group_count += serial_count
        lanelet_to_group = _rebuild_lanelet_to_group(lane_groups)
        lane_groups, parallel_count = _merge_parallel_lane_groups(
            lane_groups,
            road_lanelets,
            lanelet_to_group,
            successors,
        )
        merged_parallel_group_count += parallel_count
        if serial_count == 0 and parallel_count == 0:
            break
    lanelet_to_group = _rebuild_lanelet_to_group(lane_groups)
    signalized_intersection_area_ids = (
        _signalized_intersection_area_ids(lanelet_map, road_lanelets, successors)
        if signal_mode == "jp-static"
        else set()
    )
    collapsed_intersection_group_area_ids = _collapsed_intersection_group_area_ids(
        lane_groups,
        road_lanelets,
        excluded_area_ids=signalized_intersection_area_ids,
    )
    intersection_clusters = _build_intersection_clusters(
        road_lanelets,
        successors,
        lanelet_to_group,
        collapsed_intersection_group_area_ids,
    )
    collapsed_intersection_group_ids = set(collapsed_intersection_group_area_ids)
    pre_export_lane_groups = [group for group in lane_groups if group.group_id not in collapsed_intersection_group_ids]
    node_ids, node_points = _assign_node_ids(pre_export_lane_groups, lanelet_to_group, successors, intersection_clusters)
    dropped_self_loop_groups = {
        group.group_id
        for group in pre_export_lane_groups
        if node_ids[f"{group.group_id}:start"] == node_ids[f"{group.group_id}:end"]
    }
    dropped_tiny_groups = {
        group.group_id
        for group in pre_export_lane_groups
        if group.group_id not in dropped_self_loop_groups
        and polyline_length(group.centerline) < MIN_EXPORT_EDGE_LENGTH_M
    }
    dropped_export_group_ids = dropped_self_loop_groups | dropped_tiny_groups
    exported_lane_groups = [group for group in pre_export_lane_groups if group.group_id not in dropped_export_group_ids]
    dropped_lanelet_ids = sorted(
        [lanelet_id for lanelet_id, group_id in lanelet_to_group.items() if group_id in dropped_self_loop_groups],
        key=_sort_key,
    )
    dropped_tiny_lanelet_ids = sorted(
        [lanelet_id for lanelet_id, group_id in lanelet_to_group.items() if group_id in dropped_tiny_groups],
        key=_sort_key,
    )
    collapsed_intersection_lanelet_ids = sorted(
        [lanelet_id for lanelet_id, group_id in lanelet_to_group.items() if group_id in collapsed_intersection_group_ids],
        key=_sort_key,
    )
    edge_node_ids_by_edge_id = {
        group.edge_id: (
            node_ids[f"{group.group_id}:start"],
            node_ids[f"{group.group_id}:end"],
        )
        for group in exported_lane_groups
    }

    nodes_path = out_dir / "network.nod.xml"
    edges_path = out_dir / "network.edg.xml"
    connections_path = out_dir / "network.con.xml"
    net_path = out_dir / "network.net.xml"
    sidecar_path = out_dir / "retention.sidecar.json"
    report_path = out_dir / "conversion.report.json"

    sidecar: dict[str, object] = {
        "input": {
            "path": str(input_path),
            "lane_change_mode": lane_change_mode,
            "signal_mode": signal_mode,
        }
    }
    geo_location_patched = False

    lanelet_to_lane_index = _write_edges_xml(
        edges_path,
        exported_lane_groups,
        road_lanelets,
        lanelet_map,
        node_ids,
        lanelet_to_group,
        lane_change_analysis,
        lane_change_mode,
        sidecar,
    )
    signal_summary = {
        "raw_relation_count": 0,
        "normalized_head_count": 0,
        "normalized_control_count": 0,
        "signalized_node_count": 0,
        "tls_cluster_count": 0,
        "vehicle_tls_link_count": 0,
        "inferred_no_refline_count": 0,
        "unmapped_relation_count": 0,
    }
    unmapped_signals: list[dict[str, object]] = []
    tls_ids_by_node_id: dict[str, str] = {}
    if signal_mode == "jp-static":
        tls_ids_by_node_id, signal_summary, unmapped_signals = _plan_vehicle_signals(
            lanelet_map,
            road_lanelets,
            successors,
            intersection_clusters,
            node_ids,
            lanelet_to_group,
        )
    intersection_area_node_joins, intersection_area_node_join_summary = _build_intersection_area_node_joins(
        exported_lane_groups,
        road_lanelets,
        lanelet_map,
        node_ids,
        node_points,
        signalized_intersection_area_ids,
    )
    _write_nodes_xml(
        nodes_path,
        node_points,
        tls_ids_by_node_id=tls_ids_by_node_id,
        intersection_area_node_joins=intersection_area_node_joins,
    )

    connection_shape_summary = _write_connections_xml(
        connections_path,
        successors,
        lanelet_to_group,
        exported_lane_groups,
        lanelet_to_lane_index,
        intersection_clusters,
        road_lanelets,
        edge_node_ids_by_edge_id=edge_node_ids_by_edge_id,
    )
    connection_shape_summary["merged_non_junction_seam_count"] = merged_serial_group_count
    connection_count = int(connection_shape_summary["connection_count"])

    netconvert_result: subprocess.CompletedProcess[str] | None = None
    lane_length_patch_summary: dict[str, object] | None = None
    tls_phase_patch_summary: dict[str, object] | None = None
    internal_connection_shape_sync_summary: dict[str, object] | None = None
    internal_connection_shape_align_summary: dict[str, object] | None = None
    internal_shape_audit_summary: dict[str, object] | None = None
    internal_shape_repair_summary: dict[str, object] | None = None
    connectivity_summary: dict[str, object] | None = None
    if run_netconvert:
        build_tls_from_nodes = signal_mode == "jp-static" and bool(tls_ids_by_node_id)
        netconvert_result = _run_netconvert(
            nodes_path,
            edges_path,
            connections_path,
            net_path,
            netconvert_binary,
            build_tls_from_nodes=build_tls_from_nodes,
        )
        if build_tls_from_nodes:
            signal_summary.update(net_postprocess._summarize_net_tls(net_path))
            joined_intersection_area_tls_ids = (
                net_postprocess._joined_intersection_area_tls_ids(net_path)
                if intersection_area_node_joins
                else set()
            )
            tls_phase_patch_summary = net_postprocess._patch_net_japanese_tls_phases(
                net_path,
                excluded_tls_ids=joined_intersection_area_tls_ids,
            )
            signal_summary["japanese_phase_patch"] = tls_phase_patch_summary
            signal_summary.update(net_postprocess._summarize_net_tls(net_path))
        internal_connection_shape_sync_summary = net_postprocess._sync_internal_lane_shapes_from_connection_shapes(net_path)
        internal_connection_shape_align_summary = net_postprocess._align_internal_connection_shapes_to_net_lanes(net_path)
        internal_shape_audit_summary = net_postprocess._audit_degenerate_internal_lane_shapes(net_path)
        if int(internal_shape_audit_summary["degenerate_internal_lane_count"]) > 0:
            internal_shape_repair_summary = net_postprocess._repair_degenerate_internal_lane_shapes(net_path)
        else:
            internal_shape_repair_summary = {
                "scanned_internal_lane_count": internal_shape_audit_summary["scanned_internal_lane_count"],
                "degenerate_internal_lane_count": 0,
                "repaired_internal_lane_count": 0,
                "unrepaired_internal_lane_count": 0,
                "examples": [],
            }
        lane_length_patch_summary = net_postprocess._patch_net_lane_lengths_to_shape(net_path)
        geo_location_patched = patch_net_location(net_path, lanelet_map.geo_reference)
        connectivity_summary = net_postprocess._summarize_net_connectivity_and_write_safe_weights(net_path, out_dir)

    sidecar["ignored_lanelets_by_subtype"] = dict(sorted(ignored_subtypes.items()))
    sidecar["geo_reference"] = _geo_reference_dict(lanelet_map.geo_reference, geo_location_patched)
    sidecar["intersection_area_junction_clusters"] = {
        intersection_area_id: {
            "centroid": {
                "x": round(cluster.centroid.x, 3),
                "y": round(cluster.centroid.y, 3),
                "z": round(cluster.centroid.z, 3),
            },
            "incoming_lanelet_ids": list(cluster.incoming_lanelet_ids),
            "outgoing_lanelet_ids": list(cluster.outgoing_lanelet_ids),
            "internal_group_ids": list(cluster.group_ids),
            "internal_lanelet_ids": list(cluster.lanelet_ids),
            "movement_lanelet_pairs": [list(pair) for pair in cluster.movement_lanelet_pairs],
        }
        for intersection_area_id, cluster in intersection_clusters.items()
    }
    sidecar["collapsed_intersection_area_lanelets"] = collapsed_intersection_lanelet_ids
    sidecar["dropped_self_loop_lanelets"] = dropped_lanelet_ids
    sidecar["dropped_tiny_lanelets"] = dropped_tiny_lanelet_ids
    edge_id_by_group = {group.group_id: group.edge_id for group in exported_lane_groups}
    sidecar["road_lanelet_to_edge"] = {
        lanelet_id: (
            {
                "group_id": lanelet_to_group[lanelet_id],
                "intersection_area_id": road_lanelets[lanelet_id].tags.get("intersection_area"),
                "export_status": "collapsed_intersection_area",
            }
            if lanelet_id in collapsed_intersection_lanelet_ids
            else (
            {
                "group_id": lanelet_to_group[lanelet_id],
                "export_status": "dropped_self_loop",
            }
            if lanelet_id in dropped_lanelet_ids
            else (
            {
                "group_id": lanelet_to_group[lanelet_id],
                "export_status": "dropped_tiny",
            }
            if lanelet_id in dropped_tiny_lanelet_ids
            else {
                "group_id": lanelet_to_group[lanelet_id],
                "edge_id": edge_id_by_group[lanelet_to_group[lanelet_id]],
                "lane_index": lanelet_to_lane_index[lanelet_id],
                "export_status": "exported",
            }
            )
            )
        )
        for lanelet_id in sorted(road_lanelets, key=_sort_key)
    }
    if signal_mode == "jp-static":
        sidecar["vehicle_signals"] = {
            "tls_ids_by_node_id": dict(sorted(tls_ids_by_node_id.items(), key=lambda item: _sort_key(item[0]))),
            "unmapped_signals": unmapped_signals,
        }
    sidecar["intersection_area_node_joins"] = {
        node_join.intersection_area_id: {
            "join_id": node_join.join_id,
            "node_ids": list(node_join.node_ids),
            "centroid": {
                "x": round(node_join.point.x, 3),
                "y": round(node_join.point.y, 3),
                "z": round(node_join.point.z, 3),
            },
            "shape_point_count": len(node_join.shape),
        }
        for node_join in intersection_area_node_joins
    }
    sidecar["intersection_area_node_join_summary"] = intersection_area_node_join_summary
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
    report = {
        "input_path": str(input_path),
        "output_dir": str(out_dir),
        "lanelet_count": len(lanelet_map.lanelets),
        "road_lanelet_count": len(road_lanelets),
        "ignored_lanelets_by_subtype": dict(sorted(ignored_subtypes.items())),
        "lane_group_count": len(exported_lane_groups),
        "merged_serial_group_count": merged_serial_group_count,
        "merged_parallel_group_count": merged_parallel_group_count,
        "intersection_area_cluster_count": len(intersection_clusters),
        "collapsed_intersection_group_count": len(collapsed_intersection_group_ids),
        "collapsed_intersection_lanelet_count": len(collapsed_intersection_lanelet_ids),
        "dropped_self_loop_group_count": len(dropped_self_loop_groups),
        "dropped_self_loop_lanelet_count": len(dropped_lanelet_ids),
        "dropped_tiny_group_count": len(dropped_tiny_groups),
        "dropped_tiny_lanelet_count": len(dropped_tiny_lanelet_ids),
        "node_count": len(node_points),
        "edge_count": len(exported_lane_groups),
        "connection_count": connection_count,
        "connection_shape_summary": connection_shape_summary,
        "lane_change_summary": lane_change_analysis.summary,
        "signal_summary": signal_summary,
        "intersection_area_node_join_summary": intersection_area_node_join_summary,
        "geo_reference": _geo_reference_dict(lanelet_map.geo_reference, geo_location_patched),
    }
    if netconvert_result is not None:
        report["netconvert"] = {
            "binary": netconvert_binary,
            "stdout": netconvert_result.stdout.strip(),
            "stderr": netconvert_result.stderr.strip(),
        }
    if lane_length_patch_summary is not None:
        report["lane_length_shape_patch"] = lane_length_patch_summary
    if internal_connection_shape_sync_summary is not None:
        report["internal_connection_shape_sync"] = internal_connection_shape_sync_summary
    if internal_connection_shape_align_summary is not None:
        report["internal_connection_shape_align"] = internal_connection_shape_align_summary
    if internal_shape_audit_summary is not None:
        report["internal_shape_audit"] = internal_shape_audit_summary
    if internal_shape_repair_summary is not None:
        report["internal_shape_repair"] = internal_shape_repair_summary
    if connectivity_summary is not None:
        report["connectivity_summary"] = connectivity_summary
    if tls_phase_patch_summary is not None:
        report["tls_phase_patch"] = tls_phase_patch_summary
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
        "connections_path": str(connections_path),
        "net_path": str(net_path) if run_netconvert else None,
        "sidecar_path": str(sidecar_path),
        "report_path": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a lanelet2 OSM map into a SUMO network.")
    parser.add_argument("--input", required=True, help="Path to the lanelet2 OSM map.")
    parser.add_argument("--out-dir", required=True, help="Directory for generated SUMO artifacts.")
    parser.add_argument(
        "--lane-change-mode",
        default="lanelet-infer",
        choices=["lanelet-infer", "unrestricted"],
        help="Lane change export mode. 'unrestricted' omits SUMO lane-change restrictions.",
    )
    parser.add_argument(
        "--signal-mode",
        default="jp-static",
        choices=["none", "jp-static"],
        help="Traffic signal export mode.",
    )
    parser.add_argument(
        "--skip-netconvert",
        action="store_true",
        help="Write plain XML artifacts but skip netconvert.",
    )
    parser.add_argument(
        "--netconvert-binary",
        default="netconvert",
        help="Path to the netconvert executable.",
    )
    args = parser.parse_args()

    result = convert_map(
        input_path=args.input,
        out_dir=args.out_dir,
        lane_change_mode=args.lane_change_mode,
        signal_mode=args.signal_mode,
        run_netconvert=not args.skip_netconvert,
        netconvert_binary=args.netconvert_binary,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
