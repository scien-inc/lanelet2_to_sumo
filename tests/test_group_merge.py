from __future__ import annotations

import unittest

from ll2sumo.convert import LaneGroup, _merge_parallel_lane_groups, _merge_serial_lane_groups, _rebuild_lanelet_to_group
from ll2sumo.geometry import heading_deg, polyline_length
from ll2sumo.model import LaneChangeAnalysis, LaneChangeDecision, Lanelet, LaneletMap, Point3D, RegulatoryElement, Way


def make_lanelet(
    lanelet_id: str,
    *,
    tags: dict[str, str] | None = None,
    regulatory_ids: tuple[str, ...] = tuple(),
    centerline: tuple[Point3D, ...] | None = None,
) -> Lanelet:
    lanelet_tags = {"type": "lanelet", "subtype": "road", "one_way": "yes"}
    if tags:
        lanelet_tags.update(tags)
    centerline = centerline or (Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0))
    return Lanelet(
        id=lanelet_id,
        subtype="road",
        tags=lanelet_tags,
        left_way_id=f"{lanelet_id}_left",
        right_way_id=f"{lanelet_id}_right",
        regulatory_ids=regulatory_ids,
        left_node_ids=(f"{lanelet_id}_l0", f"{lanelet_id}_l1"),
        right_node_ids=(f"{lanelet_id}_r0", f"{lanelet_id}_r1"),
        left_boundary=centerline,
        right_boundary=centerline,
        centerline=centerline,
        start=centerline[0],
        end=centerline[-1],
        avg_heading_deg=heading_deg(centerline[0], centerline[-1]),
        length_m=polyline_length(centerline),
    )


class SerialGroupMergeTest(unittest.TestCase):
    def test_single_lane_serial_groups_are_merged(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a"),
            "b": make_lanelet("b"),
        }
        first = LaneGroup(
            group_id="group_a",
            edge_id="edge_a",
            lanelet_paths=(("a",),),
            start=Point3D(0.0, 0.0, 0.0),
            end=Point3D(10.0, 0.0, 0.0),
            centerline=(Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0)),
        )
        second = LaneGroup(
            group_id="group_b",
            edge_id="edge_b",
            lanelet_paths=(("b",),),
            start=Point3D(10.0, 0.0, 0.0),
            end=Point3D(20.0, 0.0, 0.0),
            centerline=(Point3D(10.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0)),
        )

        merged_groups, merged_count = _merge_serial_lane_groups(
            [first, second],
            road_lanelets,
            LaneletMap(nodes={}, ways={}, lanelets=road_lanelets),
            {"a": ["b"]},
            LaneChangeAnalysis(),
        )

        self.assertEqual(merged_count, 1)
        self.assertEqual(len(merged_groups), 1)
        self.assertEqual(merged_groups[0].lanelet_paths, (("a", "b"),))

    def test_serial_chain_is_merged_without_duplicate_intermediate_group(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a"),
            "b": make_lanelet("b"),
            "c": make_lanelet("c"),
        }
        first = LaneGroup(
            group_id="group_a",
            edge_id="edge_a",
            lanelet_paths=(("a",),),
            start=Point3D(0.0, 0.0, 0.0),
            end=Point3D(10.0, 0.0, 0.0),
            centerline=(Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0)),
        )
        second = LaneGroup(
            group_id="group_b",
            edge_id="edge_b",
            lanelet_paths=(("b",),),
            start=Point3D(10.0, 0.0, 0.0),
            end=Point3D(20.0, 0.0, 0.0),
            centerline=(Point3D(10.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0)),
        )
        third = LaneGroup(
            group_id="group_c",
            edge_id="edge_c",
            lanelet_paths=(("c",),),
            start=Point3D(20.0, 0.0, 0.0),
            end=Point3D(30.0, 0.0, 0.0),
            centerline=(Point3D(20.0, 0.0, 0.0), Point3D(30.0, 0.0, 0.0)),
        )

        merged_groups, merged_count = _merge_serial_lane_groups(
            [second, first, third],
            road_lanelets,
            LaneletMap(nodes={}, ways={}, lanelets=road_lanelets),
            {"a": ["b"], "b": ["c"]},
            LaneChangeAnalysis(),
        )

        self.assertEqual(merged_count, 2)
        self.assertEqual(len(merged_groups), 1)
        self.assertEqual(merged_groups[0].lanelet_paths, (("a", "b", "c"),))

    def test_turn_direction_does_not_block_merge_without_intersection_area(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a"),
            "b": make_lanelet("b", tags={"turn_direction": "right"}),
        }
        first = LaneGroup(
            group_id="group_a",
            edge_id="edge_a",
            lanelet_paths=(("a",),),
            start=Point3D(0.0, 0.0, 0.0),
            end=Point3D(10.0, 0.0, 0.0),
            centerline=(Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0)),
        )
        second = LaneGroup(
            group_id="group_b",
            edge_id="edge_b",
            lanelet_paths=(("b",),),
            start=Point3D(10.0, 0.0, 0.0),
            end=Point3D(20.0, 0.0, 0.0),
            centerline=(Point3D(10.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0)),
        )

        merged_groups, merged_count = _merge_serial_lane_groups(
            [first, second],
            road_lanelets,
            LaneletMap(nodes={}, ways={}, lanelets=road_lanelets),
            {"a": ["b"]},
            LaneChangeAnalysis(),
        )

        self.assertEqual(merged_count, 1)
        self.assertEqual(len(merged_groups), 1)

    def test_regulatory_elements_do_not_block_merge_without_intersection_area(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a", regulatory_ids=("row",)),
            "b": make_lanelet("b"),
        }
        lanelet_map = LaneletMap(
            nodes={},
            ways={},
            lanelets=road_lanelets,
        )
        first = LaneGroup(
            group_id="group_a",
            edge_id="edge_a",
            lanelet_paths=(("a",),),
            start=Point3D(0.0, 0.0, 0.0),
            end=Point3D(10.0, 0.0, 0.0),
            centerline=(Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0)),
        )
        second = LaneGroup(
            group_id="group_b",
            edge_id="edge_b",
            lanelet_paths=(("b",),),
            start=Point3D(10.0, 0.0, 0.0),
            end=Point3D(20.0, 0.0, 0.0),
            centerline=(Point3D(10.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0)),
        )

        merged_groups, merged_count = _merge_serial_lane_groups(
            [first, second],
            road_lanelets,
            lanelet_map,
            {"a": ["b"]},
            LaneChangeAnalysis(),
        )

        self.assertEqual(merged_count, 1)
        self.assertEqual(len(merged_groups), 1)

    def test_intersection_area_blocks_merge_at_boundary(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a"),
            "b": make_lanelet("b", tags={"intersection_area": "ia-1", "turn_direction": "straight"}),
        }
        first = LaneGroup(
            group_id="group_a",
            edge_id="edge_a",
            lanelet_paths=(("a",),),
            start=Point3D(0.0, 0.0, 0.0),
            end=Point3D(10.0, 0.0, 0.0),
            centerline=(Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0)),
        )
        second = LaneGroup(
            group_id="group_b",
            edge_id="edge_b",
            lanelet_paths=(("b",),),
            start=Point3D(10.0, 0.0, 0.0),
            end=Point3D(20.0, 0.0, 0.0),
            centerline=(Point3D(10.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0)),
        )

        merged_groups, merged_count = _merge_serial_lane_groups(
            [first, second],
            road_lanelets,
            LaneletMap(nodes={}, ways={}, lanelets=road_lanelets),
            {"a": ["b"]},
            LaneChangeAnalysis(),
        )

        self.assertEqual(merged_count, 0)
        self.assertEqual(len(merged_groups), 2)

    def test_traffic_light_stopline_blocks_same_point_serial_merge(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a", regulatory_ids=("tl",), centerline=(Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0))),
            "b": make_lanelet("b", centerline=(Point3D(10.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0))),
        }
        lanelet_map = LaneletMap(
            nodes={"s0": Point3D(10.0, -1.0, 0.0), "s1": Point3D(10.0, 1.0, 0.0)},
            ways={
                "head": Way("head", ("h0", "h1"), {"subtype": "red_yellow_green"}),
                "stop": Way("stop", ("s0", "s1"), {}),
            },
            lanelets=road_lanelets,
            regulatory_elements={
                "tl": RegulatoryElement(
                    "tl",
                    "traffic_light",
                    {"subtype": "traffic_light"},
                    {"refers": ("head",), "ref_line": ("stop",)},
                )
            },
        )
        first = LaneGroup(
            group_id="group_a",
            edge_id="edge_a",
            lanelet_paths=(("a",),),
            start=Point3D(0.0, 0.0, 0.0),
            end=Point3D(10.0, 0.0, 0.0),
            centerline=(Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0)),
        )
        second = LaneGroup(
            group_id="group_b",
            edge_id="edge_b",
            lanelet_paths=(("b",),),
            start=Point3D(10.0, 0.0, 0.0),
            end=Point3D(20.0, 0.0, 0.0),
            centerline=(Point3D(10.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0)),
        )

        merged_groups, merged_count = _merge_serial_lane_groups(
            [first, second],
            road_lanelets,
            lanelet_map,
            {"a": ["b"]},
            LaneChangeAnalysis(),
        )

        self.assertEqual(merged_count, 0)
        self.assertEqual(len(merged_groups), 2)

    def test_serial_merge_absorbs_long_group_when_exported_state_matches(self) -> None:
        road_lanelets = {
            "a0": make_lanelet("a0"),
            "a1": make_lanelet("a1"),
            "b0": make_lanelet("b0"),
            "b1": make_lanelet("b1"),
        }
        first = LaneGroup(
            group_id="group_a",
            edge_id="edge_a",
            lanelet_paths=(("a0",), ("a1",)),
            start=Point3D(0.0, 0.0, 0.0),
            end=Point3D(10.0, 0.0, 0.0),
            centerline=(Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0)),
        )
        second = LaneGroup(
            group_id="group_b",
            edge_id="edge_b",
            lanelet_paths=(("b0",), ("b1",)),
            start=Point3D(10.0, 0.0, 0.0),
            end=Point3D(20.0, 0.0, 0.0),
            centerline=(Point3D(10.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0)),
        )
        lane_change_analysis = LaneChangeAnalysis(
            decisions={
                ("a0", "a1"): LaneChangeDecision("a0", "a1", "left", "boundary_a", False, "boundary_explicit_no", "boundary"),
                ("a1", "a0"): LaneChangeDecision("a1", "a0", "right", "boundary_a", True, "boundary_explicit_yes", "boundary"),
                ("b0", "b1"): LaneChangeDecision("b0", "b1", "left", "boundary_b", False, "solid_boundary", "boundary"),
                ("b1", "b0"): LaneChangeDecision("b1", "b0", "right", "boundary_b", True, "boundary_explicit_yes", "boundary"),
            }
        )

        merged_groups, merged_count = _merge_serial_lane_groups(
            [first, second],
            road_lanelets,
            LaneletMap(nodes={}, ways={}, lanelets=road_lanelets),
            {"a0": ["b0"], "a1": ["b1"]},
            lane_change_analysis,
        )

        self.assertEqual(merged_count, 1)
        self.assertEqual(len(merged_groups), 1)
        self.assertEqual(merged_groups[0].lanelet_paths, (("a0", "b0"), ("a1", "b1")))

    def test_serial_merge_keeps_long_group_when_exported_state_changes(self) -> None:
        road_lanelets = {
            "a0": make_lanelet("a0"),
            "a1": make_lanelet("a1"),
            "a2": make_lanelet("a2"),
            "b0": make_lanelet("b0"),
            "b1": make_lanelet("b1"),
            "b2": make_lanelet("b2"),
        }
        first = LaneGroup(
            group_id="group_a",
            edge_id="edge_a",
            lanelet_paths=(("a0",), ("a1",), ("a2",)),
            start=Point3D(0.0, 0.0, 0.0),
            end=Point3D(10.0, 0.0, 0.0),
            centerline=(Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0)),
        )
        second = LaneGroup(
            group_id="group_b",
            edge_id="edge_b",
            lanelet_paths=(("b0",), ("b1",), ("b2",)),
            start=Point3D(10.0, 0.0, 0.0),
            end=Point3D(20.0, 0.0, 0.0),
            centerline=(Point3D(10.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0)),
        )
        lane_change_analysis = LaneChangeAnalysis(
            decisions={
                ("a0", "a1"): LaneChangeDecision("a0", "a1", "left", "boundary_a0", False, "boundary_explicit_no", "boundary"),
                ("a1", "a0"): LaneChangeDecision("a1", "a0", "right", "boundary_a0", True, "boundary_explicit_yes", "boundary"),
                ("a1", "a2"): LaneChangeDecision("a1", "a2", "left", "boundary_a1", True, "boundary_explicit_yes", "boundary"),
                ("a2", "a1"): LaneChangeDecision("a2", "a1", "right", "boundary_a1", False, "boundary_explicit_no", "boundary"),
                ("b0", "b1"): LaneChangeDecision("b0", "b1", "left", "boundary_b0", False, "boundary_explicit_no", "boundary"),
                ("b1", "b0"): LaneChangeDecision("b1", "b0", "right", "boundary_b0", True, "boundary_explicit_yes", "boundary"),
                ("b1", "b2"): LaneChangeDecision("b1", "b2", "left", "boundary_b1", False, "boundary_explicit_no", "boundary"),
                ("b2", "b1"): LaneChangeDecision("b2", "b1", "right", "boundary_b1", False, "boundary_explicit_no", "boundary"),
            }
        )

        merged_groups, merged_count = _merge_serial_lane_groups(
            [first, second],
            road_lanelets,
            LaneletMap(nodes={}, ways={}, lanelets=road_lanelets),
            {"a0": ["b0"], "a1": ["b1"], "a2": ["b2"]},
            lane_change_analysis,
        )

        self.assertEqual(merged_count, 0)
        self.assertEqual(len(merged_groups), 2)

    def test_serial_merge_absorbs_short_restrictive_tail(self) -> None:
        road_lanelets = {
            "a0": make_lanelet("a0"),
            "a1": make_lanelet("a1"),
            "a2": make_lanelet("a2"),
            "b0": make_lanelet("b0"),
            "b1": make_lanelet("b1"),
            "b2": make_lanelet("b2"),
        }
        first = LaneGroup(
            group_id="group_a",
            edge_id="edge_a",
            lanelet_paths=(("a0",), ("a1",), ("a2",)),
            start=Point3D(0.0, 0.0, 0.0),
            end=Point3D(10.0, 0.0, 0.0),
            centerline=(Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0)),
        )
        second = LaneGroup(
            group_id="group_b",
            edge_id="edge_b",
            lanelet_paths=(("b0",), ("b1",), ("b2",)),
            start=Point3D(10.0, 0.0, 0.0),
            end=Point3D(12.0, 0.0, 0.0),
            centerline=(Point3D(10.0, 0.0, 0.0), Point3D(12.0, 0.0, 0.0)),
        )
        lane_change_analysis = LaneChangeAnalysis(
            decisions={
                ("a0", "a1"): LaneChangeDecision("a0", "a1", "left", "boundary_a0", False, "boundary_explicit_no", "boundary"),
                ("a1", "a0"): LaneChangeDecision("a1", "a0", "right", "boundary_a0", True, "boundary_explicit_yes", "boundary"),
                ("a1", "a2"): LaneChangeDecision("a1", "a2", "left", "boundary_a1", True, "boundary_explicit_yes", "boundary"),
                ("a2", "a1"): LaneChangeDecision("a2", "a1", "right", "boundary_a1", False, "boundary_explicit_no", "boundary"),
                ("b0", "b1"): LaneChangeDecision("b0", "b1", "left", "boundary_b0", False, "boundary_explicit_no", "boundary"),
                ("b1", "b0"): LaneChangeDecision("b1", "b0", "right", "boundary_b0", True, "boundary_explicit_yes", "boundary"),
                ("b1", "b2"): LaneChangeDecision("b1", "b2", "left", "boundary_b1", False, "boundary_explicit_no", "boundary"),
                ("b2", "b1"): LaneChangeDecision("b2", "b1", "right", "boundary_b1", False, "boundary_explicit_no", "boundary"),
            }
        )

        merged_groups, merged_count = _merge_serial_lane_groups(
            [first, second],
            road_lanelets,
            LaneletMap(nodes={}, ways={}, lanelets=road_lanelets),
            {"a0": ["b0"], "a1": ["b1"], "a2": ["b2"]},
            lane_change_analysis,
        )

        self.assertEqual(merged_count, 1)
        self.assertEqual(len(merged_groups), 1)
        self.assertEqual(merged_groups[0].lanelet_paths, (("a0", "b0"), ("a1", "b1"), ("a2", "b2")))

    def test_serial_merge_does_not_create_foldback_edge(self) -> None:
        first_centerline = (
            Point3D(0.0, 0.0, 0.0),
            Point3D(20.0, 0.0, 0.0),
            Point3D(20.0, 20.0, 0.0),
            Point3D(0.0, 20.0, 0.0),
        )
        second_centerline = (
            Point3D(0.0, 20.0, 0.0),
            Point3D(0.0, 0.0, 0.0),
        )
        road_lanelets = {
            "a": make_lanelet("a", centerline=first_centerline),
            "b": make_lanelet("b", centerline=second_centerline),
        }
        first = LaneGroup(
            group_id="group_a",
            edge_id="edge_a",
            lanelet_paths=(("a",),),
            start=first_centerline[0],
            end=first_centerline[-1],
            centerline=first_centerline,
        )
        second = LaneGroup(
            group_id="group_b",
            edge_id="edge_b",
            lanelet_paths=(("b",),),
            start=second_centerline[0],
            end=second_centerline[-1],
            centerline=second_centerline,
        )

        merged_groups, merged_count = _merge_serial_lane_groups(
            [first, second],
            road_lanelets,
            LaneletMap(nodes={}, ways={}, lanelets=road_lanelets),
            {"a": ["b"]},
            LaneChangeAnalysis(),
        )

        self.assertEqual(merged_count, 0)
        self.assertEqual(len(merged_groups), 2)

    def test_serial_chain_stops_before_foldback_edge(self) -> None:
        centerlines = {
            "a": (Point3D(0.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0)),
            "b": (Point3D(20.0, 0.0, 0.0), Point3D(20.0, 20.0, 0.0)),
            "c": (Point3D(20.0, 20.0, 0.0), Point3D(0.0, 20.0, 0.0)),
            "d": (Point3D(0.0, 20.0, 0.0), Point3D(0.0, 0.0, 0.0)),
        }
        road_lanelets = {
            lanelet_id: make_lanelet(lanelet_id, centerline=centerline)
            for lanelet_id, centerline in centerlines.items()
        }
        groups = [
            LaneGroup(
                group_id=f"group_{lanelet_id}",
                edge_id=f"edge_{lanelet_id}",
                lanelet_paths=((lanelet_id,),),
                start=centerline[0],
                end=centerline[-1],
                centerline=centerline,
            )
            for lanelet_id, centerline in centerlines.items()
        ]

        merged_groups, _ = _merge_serial_lane_groups(
            groups,
            road_lanelets,
            LaneletMap(nodes={}, ways={}, lanelets=road_lanelets),
            {"a": ["b"], "b": ["c"], "c": ["d"]},
            LaneChangeAnalysis(),
        )

        self.assertFalse(any(group.lanelet_paths == (("a", "b", "c", "d"),) for group in merged_groups))

    def test_parallel_groups_with_same_nodes_are_merged(self) -> None:
        road_lanelets = {
            "p": make_lanelet("p"),
            "a": make_lanelet("a"),
            "b": make_lanelet("b"),
            "t": make_lanelet("t"),
        }
        predecessor = LaneGroup(
            group_id="group_p",
            edge_id="edge_p",
            lanelet_paths=(("p",),),
            start=Point3D(-10.0, 0.0, 0.0),
            end=Point3D(0.0, 0.0, 0.0),
            centerline=(Point3D(-10.0, 0.0, 0.0), Point3D(0.0, 0.0, 0.0)),
        )
        first = LaneGroup(
            group_id="group_a",
            edge_id="edge_a",
            lanelet_paths=(("a",),),
            start=Point3D(0.0, 1.0, 0.0),
            end=Point3D(10.0, 1.0, 0.0),
            centerline=(Point3D(0.0, 1.0, 0.0), Point3D(10.0, 1.0, 0.0)),
        )
        second = LaneGroup(
            group_id="group_b",
            edge_id="edge_b",
            lanelet_paths=(("b",),),
            start=Point3D(0.0, -1.0, 0.0),
            end=Point3D(10.0, -1.0, 0.0),
            centerline=(Point3D(0.0, -1.0, 0.0), Point3D(10.0, -1.0, 0.0)),
        )
        target = LaneGroup(
            group_id="group_t",
            edge_id="edge_t",
            lanelet_paths=(("t",),),
            start=Point3D(10.0, 0.0, 0.0),
            end=Point3D(20.0, 0.0, 0.0),
            centerline=(Point3D(10.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0)),
        )

        merged_groups, merged_count = _merge_parallel_lane_groups(
            [predecessor, first, second, target],
            road_lanelets,
            _rebuild_lanelet_to_group([predecessor, first, second, target]),
            {"p": ["a", "b"], "a": ["t"], "b": ["t"]},
        )

        self.assertEqual(merged_count, 1)
        merged = [group for group in merged_groups if group.group_id == "group_a"][0]
        self.assertEqual({path[0] for path in merged.lanelet_paths}, {"a", "b"})

    def test_parallel_groups_with_divergent_geometry_are_not_merged(self) -> None:
        road_lanelets = {
            "p": make_lanelet("p"),
            "a": make_lanelet("a", centerline=(Point3D(0.0, 1.0, 0.0), Point3D(10.0, 1.0, 0.0))),
            "b": make_lanelet(
                "b",
                centerline=(
                    Point3D(0.0, -1.0, 0.0),
                    Point3D(50.0, -1.0, 0.0),
                    Point3D(10.0, -1.0, 0.0),
                ),
            ),
            "t": make_lanelet("t"),
        }
        predecessor = LaneGroup(
            group_id="group_p",
            edge_id="edge_p",
            lanelet_paths=(("p",),),
            start=Point3D(-10.0, 0.0, 0.0),
            end=Point3D(0.0, 0.0, 0.0),
            centerline=(Point3D(-10.0, 0.0, 0.0), Point3D(0.0, 0.0, 0.0)),
        )
        first = LaneGroup(
            group_id="group_a",
            edge_id="edge_a",
            lanelet_paths=(("a",),),
            start=Point3D(0.0, 1.0, 0.0),
            end=Point3D(10.0, 1.0, 0.0),
            centerline=road_lanelets["a"].centerline,
        )
        second = LaneGroup(
            group_id="group_b",
            edge_id="edge_b",
            lanelet_paths=(("b",),),
            start=Point3D(0.0, -1.0, 0.0),
            end=Point3D(10.0, -1.0, 0.0),
            centerline=road_lanelets["b"].centerline,
        )
        target = LaneGroup(
            group_id="group_t",
            edge_id="edge_t",
            lanelet_paths=(("t",),),
            start=Point3D(10.0, 0.0, 0.0),
            end=Point3D(20.0, 0.0, 0.0),
            centerline=(Point3D(10.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0)),
        )

        merged_groups, merged_count = _merge_parallel_lane_groups(
            [predecessor, first, second, target],
            road_lanelets,
            _rebuild_lanelet_to_group([predecessor, first, second, target]),
            {"p": ["a", "b"], "a": ["t"], "b": ["t"]},
        )

        self.assertEqual(merged_count, 0)
        self.assertEqual({group.group_id for group in merged_groups}, {"group_p", "group_a", "group_b", "group_t"})


if __name__ == "__main__":
    unittest.main()
