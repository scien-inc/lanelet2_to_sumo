from __future__ import annotations

from collections import Counter, defaultdict

from ll2sumo.geometry import angle_diff_deg, distance_2d, max_z_gap, parallel_polylines_compatible
from ll2sumo.model import BoundaryNeighbor, LaneChangeAnalysis, LaneChangeDecision, Lanelet, LaneletMap

BLOCK_ALL_BUT_EMERGENCY = "emergency"
MAX_BUNDLE_LANE_LENGTH_RATIO = 1.35
MAX_BUNDLE_MEAN_GAP_M = 8.0
MAX_BUNDLE_SAMPLE_GAP_M = 12.0
MAX_BUNDLE_SEGMENT_HEADING_DIFF_DEG = 45.0


def _lane_change_flag(tags: dict[str, str], direction: str) -> bool | None:
    directional_value = tags.get(f"lane_change:{direction}")
    if directional_value in {"yes", "no"}:
        return directional_value == "yes"

    generic_value = tags.get("lane_change")
    if generic_value in {"yes", "no"}:
        return generic_value == "yes"
    if generic_value == "left":
        return direction == "left"
    if generic_value == "right":
        return direction == "right"
    return None


def _explicit_lanelet_decision(source_lanelet: Lanelet, direction: str) -> tuple[bool, str, str] | None:
    value = _lane_change_flag(source_lanelet.tags, direction)
    if value is None:
        return None
    if value:
        return True, "lanelet_explicit_yes", "lanelet"
    return False, "lanelet_explicit_no", "lanelet"


def _explicit_boundary_decision(boundary_tags: dict[str, str], direction: str) -> tuple[bool, str, str] | None:
    value = _lane_change_flag(boundary_tags, direction)
    if value is None:
        return None
    if value:
        return True, "boundary_explicit_yes", "boundary"
    return False, "boundary_explicit_no", "boundary"


def _fallback_boundary_decision(boundary_tags: dict[str, str]) -> tuple[bool, str, str]:
    boundary_type = boundary_tags.get("type")
    boundary_subtype = boundary_tags.get("subtype")
    if boundary_type in {"road_border", "fence", "guard_rail"}:
        return False, "physical_boundary", "boundary"
    if boundary_type == "virtual":
        return True, "inferred_virtual", "boundary"
    if boundary_type == "line_thin" and boundary_subtype == "dashed":
        return True, "inferred_dashed", "boundary"
    if boundary_type == "line_thin" and boundary_subtype == "solid":
        return False, "solid_boundary", "boundary"
    return False, "unknown_boundary", "boundary"


def _evaluate_direction(
    source_lanelet: Lanelet,
    target_lanelet: Lanelet,
    boundary_id: str,
    boundary_tags: dict[str, str],
    direction: str,
    heading_diff_deg: float,
    z_gap_m: float,
) -> LaneChangeDecision:
    if heading_diff_deg > 30.0:
        return LaneChangeDecision(
            source_lanelet_id=source_lanelet.id,
            target_lanelet_id=target_lanelet.id,
            direction=direction,
            boundary_id=boundary_id,
            allowed=False,
            reason="heading_mismatch",
            source="geometry",
        )
    if z_gap_m > 0.5:
        return LaneChangeDecision(
            source_lanelet_id=source_lanelet.id,
            target_lanelet_id=target_lanelet.id,
            direction=direction,
            boundary_id=boundary_id,
            allowed=False,
            reason="z_gap",
            source="geometry",
        )

    explicit_lanelet = _explicit_lanelet_decision(source_lanelet, direction)
    if explicit_lanelet is not None:
        allowed, reason, source = explicit_lanelet
        return LaneChangeDecision(source_lanelet.id, target_lanelet.id, direction, boundary_id, allowed, reason, source)

    explicit_boundary = _explicit_boundary_decision(boundary_tags, direction)
    if explicit_boundary is not None:
        allowed, reason, source = explicit_boundary
        return LaneChangeDecision(source_lanelet.id, target_lanelet.id, direction, boundary_id, allowed, reason, source)

    allowed, reason, source = _fallback_boundary_decision(boundary_tags)
    return LaneChangeDecision(source_lanelet.id, target_lanelet.id, direction, boundary_id, allowed, reason, source)


def analyze_lane_changes(lanelet_map: LaneletMap) -> LaneChangeAnalysis:
    road_lanelets = {lanelet_id: lanelet for lanelet_id, lanelet in lanelet_map.lanelets.items() if lanelet.subtype == "road"}
    left_boundary_index: dict[str, list[str]] = defaultdict(list)
    right_boundary_index: dict[str, list[str]] = defaultdict(list)

    for lanelet in road_lanelets.values():
        left_boundary_index[lanelet.left_way_id].append(lanelet.id)
        right_boundary_index[lanelet.right_way_id].append(lanelet.id)

    analysis = LaneChangeAnalysis(summary=Counter())
    decisions: dict[tuple[str, str], LaneChangeDecision] = {}

    for boundary_id in sorted(set(left_boundary_index) & set(right_boundary_index)):
        boundary_way = lanelet_map.ways[boundary_id]
        boundary_tags = boundary_way.tags
        for right_lanelet_id in left_boundary_index[boundary_id]:
            for left_lanelet_id in right_boundary_index[boundary_id]:
                if right_lanelet_id == left_lanelet_id:
                    continue
                right_lanelet = road_lanelets[right_lanelet_id]
                left_lanelet = road_lanelets[left_lanelet_id]
                neighbor = BoundaryNeighbor(
                    boundary_id=boundary_id,
                    left_lanelet_id=left_lanelet_id,
                    right_lanelet_id=right_lanelet_id,
                    start_gap_m=distance_2d(left_lanelet.start, right_lanelet.start),
                    end_gap_m=distance_2d(left_lanelet.end, right_lanelet.end),
                    heading_diff_deg=angle_diff_deg(left_lanelet.avg_heading_deg, right_lanelet.avg_heading_deg),
                    z_gap_m=max_z_gap(left_lanelet.centerline, right_lanelet.centerline),
                    bundle_eligible=False,
                )
                bundle_eligible = (
                    neighbor.start_gap_m <= 8.0
                    and neighbor.end_gap_m <= 8.0
                    and neighbor.heading_diff_deg <= 30.0
                    and neighbor.z_gap_m <= 0.5
                    and parallel_polylines_compatible(
                        left_lanelet.centerline,
                        right_lanelet.centerline,
                        max_length_ratio=MAX_BUNDLE_LANE_LENGTH_RATIO,
                        max_mean_gap_m=MAX_BUNDLE_MEAN_GAP_M,
                        max_sample_gap_m=MAX_BUNDLE_SAMPLE_GAP_M,
                        max_segment_heading_diff_deg=MAX_BUNDLE_SEGMENT_HEADING_DIFF_DEG,
                    )
                )
                analysis.neighbors.append(
                    BoundaryNeighbor(
                        boundary_id=neighbor.boundary_id,
                        left_lanelet_id=neighbor.left_lanelet_id,
                        right_lanelet_id=neighbor.right_lanelet_id,
                        start_gap_m=neighbor.start_gap_m,
                        end_gap_m=neighbor.end_gap_m,
                        heading_diff_deg=neighbor.heading_diff_deg,
                        z_gap_m=neighbor.z_gap_m,
                        bundle_eligible=bundle_eligible,
                    )
                )

                right_to_left = _evaluate_direction(
                    source_lanelet=right_lanelet,
                    target_lanelet=left_lanelet,
                    boundary_id=boundary_id,
                    boundary_tags=boundary_tags,
                    direction="left",
                    heading_diff_deg=neighbor.heading_diff_deg,
                    z_gap_m=neighbor.z_gap_m,
                )
                decisions[(right_lanelet_id, left_lanelet_id)] = right_to_left

                left_to_right = _evaluate_direction(
                    source_lanelet=left_lanelet,
                    target_lanelet=right_lanelet,
                    boundary_id=boundary_id,
                    boundary_tags=boundary_tags,
                    direction="right",
                    heading_diff_deg=neighbor.heading_diff_deg,
                    z_gap_m=neighbor.z_gap_m,
                )
                decisions[(left_lanelet_id, right_lanelet_id)] = left_to_right

    summary = Counter()
    for decision in decisions.values():
        if decision.allowed:
            summary["allowed_pairs"] += 1
            continue
        summary["blocked_pairs"] += 1
        if decision.reason in {"lanelet_explicit_no", "boundary_explicit_no"}:
            summary["blocked_by_explicit_no"] += 1
        if decision.reason == "solid_boundary":
            summary["blocked_by_solid_boundary"] += 1
        if decision.reason == "z_gap":
            summary["blocked_by_z_gap"] += 1
        if decision.reason == "heading_mismatch":
            summary["blocked_by_heading_mismatch"] += 1
        if decision.reason == "physical_boundary":
            summary["blocked_by_physical_boundary"] += 1
        if decision.reason == "unknown_boundary":
            summary["blocked_by_unknown_boundary"] += 1

    analysis.decisions = decisions
    analysis.summary = dict(summary)
    return analysis


def blocked_change_permissions() -> str:
    return BLOCK_ALL_BUT_EMERGENCY
