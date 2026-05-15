from __future__ import annotations

import unittest

from ll2sumo.convert import _lane_change_attributes
from ll2sumo.geometry import heading_deg, polyline_length
from ll2sumo.lane_change import analyze_lane_changes
from ll2sumo.model import Lanelet, LaneletMap, Point3D, Way


def make_lanelet(
    lanelet_id: str,
    *,
    left_way_id: str,
    right_way_id: str,
    y_start: float,
    y_end: float | None = None,
    z: float = 0.0,
    subtype: str = "road",
    tags: dict[str, str] | None = None,
    centerline: tuple[Point3D, ...] | None = None,
) -> Lanelet:
    y_end = y_start if y_end is None else y_end
    centerline = centerline or (
        Point3D(0.0, y_start, z),
        Point3D(10.0, y_end, z),
    )
    return Lanelet(
        id=lanelet_id,
        subtype=subtype,
        tags=tags or {"type": "lanelet", "subtype": subtype},
        left_way_id=left_way_id,
        right_way_id=right_way_id,
        regulatory_ids=tuple(),
        left_node_ids=(f"{lanelet_id}_left_start", f"{lanelet_id}_left_end"),
        right_node_ids=(f"{lanelet_id}_right_start", f"{lanelet_id}_right_end"),
        left_boundary=centerline,
        right_boundary=centerline,
        centerline=centerline,
        start=centerline[0],
        end=centerline[-1],
        avg_heading_deg=heading_deg(centerline[0], centerline[-1]),
        length_m=polyline_length(centerline),
    )


def make_map(shared_boundary_tags: dict[str, str], left_lanelet: Lanelet, right_lanelet: Lanelet) -> LaneletMap:
    ways = {
        left_lanelet.left_way_id: Way(id=left_lanelet.left_way_id, node_ids=tuple(), tags={"type": "road_border"}),
        right_lanelet.right_way_id: Way(id=right_lanelet.right_way_id, node_ids=tuple(), tags={"type": "road_border"}),
        "shared": Way(id="shared", node_ids=tuple(), tags=shared_boundary_tags),
    }
    return LaneletMap(nodes={}, ways=ways, lanelets={left_lanelet.id: left_lanelet, right_lanelet.id: right_lanelet})


class LaneChangeAnalysisTest(unittest.TestCase):
    def test_virtual_boundary_is_changeable(self) -> None:
        left_lanelet = make_lanelet("left", left_way_id="left_outer", right_way_id="shared", y_start=4.0)
        right_lanelet = make_lanelet("right", left_way_id="shared", right_way_id="right_outer", y_start=0.0)
        lanelet_map = make_map({"type": "virtual"}, left_lanelet, right_lanelet)

        analysis = analyze_lane_changes(lanelet_map)

        self.assertTrue(analysis.decisions[("right", "left")].allowed)
        self.assertTrue(analysis.decisions[("left", "right")].allowed)

    def test_dashed_boundary_without_explicit_tag_is_changeable(self) -> None:
        left_lanelet = make_lanelet("left", left_way_id="left_outer", right_way_id="shared", y_start=4.0)
        right_lanelet = make_lanelet("right", left_way_id="shared", right_way_id="right_outer", y_start=0.0)
        lanelet_map = make_map({"type": "line_thin", "subtype": "dashed"}, left_lanelet, right_lanelet)

        analysis = analyze_lane_changes(lanelet_map)

        self.assertEqual(analysis.decisions[("right", "left")].reason, "inferred_dashed")
        self.assertTrue(analysis.decisions[("left", "right")].allowed)

    def test_explicit_no_overrides_dashed_boundary(self) -> None:
        left_lanelet = make_lanelet("left", left_way_id="left_outer", right_way_id="shared", y_start=4.0)
        right_lanelet = make_lanelet("right", left_way_id="shared", right_way_id="right_outer", y_start=0.0)
        lanelet_map = make_map(
            {"type": "line_thin", "subtype": "dashed", "lane_change": "no"},
            left_lanelet,
            right_lanelet,
        )

        analysis = analyze_lane_changes(lanelet_map)

        self.assertFalse(analysis.decisions[("right", "left")].allowed)
        self.assertFalse(analysis.decisions[("left", "right")].allowed)
        self.assertEqual(analysis.summary["blocked_by_explicit_no"], 2)

    def test_solid_boundary_blocks_change(self) -> None:
        left_lanelet = make_lanelet("left", left_way_id="left_outer", right_way_id="shared", y_start=4.0)
        right_lanelet = make_lanelet("right", left_way_id="shared", right_way_id="right_outer", y_start=0.0)
        lanelet_map = make_map({"type": "line_thin", "subtype": "solid"}, left_lanelet, right_lanelet)

        analysis = analyze_lane_changes(lanelet_map)

        self.assertFalse(analysis.decisions[("right", "left")].allowed)
        self.assertEqual(analysis.summary["blocked_by_solid_boundary"], 2)

    def test_directional_boundary_tags_are_asymmetric(self) -> None:
        left_lanelet = make_lanelet("left", left_way_id="left_outer", right_way_id="shared", y_start=4.0)
        right_lanelet = make_lanelet("right", left_way_id="shared", right_way_id="right_outer", y_start=0.0)
        lanelet_map = make_map(
            {"type": "line_thin", "subtype": "dashed", "lane_change:left": "no", "lane_change:right": "yes"},
            left_lanelet,
            right_lanelet,
        )

        analysis = analyze_lane_changes(lanelet_map)

        self.assertFalse(analysis.decisions[("right", "left")].allowed)
        self.assertTrue(analysis.decisions[("left", "right")].allowed)

    def test_z_gap_blocks_change(self) -> None:
        left_lanelet = make_lanelet("left", left_way_id="left_outer", right_way_id="shared", y_start=4.0, z=1.0)
        right_lanelet = make_lanelet("right", left_way_id="shared", right_way_id="right_outer", y_start=0.0, z=0.0)
        lanelet_map = make_map({"type": "virtual"}, left_lanelet, right_lanelet)

        analysis = analyze_lane_changes(lanelet_map)

        self.assertFalse(analysis.decisions[("right", "left")].allowed)
        self.assertEqual(analysis.summary["blocked_by_z_gap"], 2)

    def test_non_road_lanelets_are_not_considered(self) -> None:
        left_lanelet = make_lanelet("left", left_way_id="left_outer", right_way_id="shared", y_start=4.0, subtype="crosswalk")
        right_lanelet = make_lanelet("right", left_way_id="shared", right_way_id="right_outer", y_start=0.0)
        lanelet_map = make_map({"type": "virtual"}, left_lanelet, right_lanelet)

        analysis = analyze_lane_changes(lanelet_map)

        self.assertEqual(analysis.decisions, {})

    def test_parallel_lanelets_are_bundle_eligible(self) -> None:
        left_lanelet = make_lanelet("left", left_way_id="left_outer", right_way_id="shared", y_start=4.0)
        right_lanelet = make_lanelet("right", left_way_id="shared", right_way_id="right_outer", y_start=0.0)
        lanelet_map = make_map({"type": "virtual"}, left_lanelet, right_lanelet)

        analysis = analyze_lane_changes(lanelet_map)

        self.assertTrue(analysis.neighbors[0].bundle_eligible)

    def test_diverging_mid_geometry_is_not_bundle_eligible(self) -> None:
        left_lanelet = make_lanelet(
            "left",
            left_way_id="left_outer",
            right_way_id="shared",
            y_start=4.0,
            centerline=(
                Point3D(0.0, 4.0, 0.0),
                Point3D(5.0, 30.0, 0.0),
                Point3D(10.0, 4.0, 0.0),
            ),
        )
        right_lanelet = make_lanelet("right", left_way_id="shared", right_way_id="right_outer", y_start=0.0)
        lanelet_map = make_map({"type": "virtual"}, left_lanelet, right_lanelet)

        analysis = analyze_lane_changes(lanelet_map)

        self.assertFalse(analysis.neighbors[0].bundle_eligible)

    def test_unrestricted_mode_omits_blocked_lane_change_attributes(self) -> None:
        attributes = _lane_change_attributes(
            {"status": "blocked"},
            {"status": "blocked"},
            "unrestricted",
        )

        self.assertEqual(attributes, {})

    def test_lanelet_infer_mode_writes_blocked_lane_change_attributes(self) -> None:
        attributes = _lane_change_attributes(
            {"status": "blocked"},
            {"status": "allowed"},
            "lanelet-infer",
        )

        self.assertIn("changeLeft", attributes)
        self.assertNotIn("changeRight", attributes)


if __name__ == "__main__":
    unittest.main()
