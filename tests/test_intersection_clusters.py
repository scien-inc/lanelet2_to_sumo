from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from ll2sumo.convert import (
    IntersectionCluster,
    IntersectionAreaNodeJoin,
    LaneGroup,
    _assign_node_ids,
    _collapsed_intersection_group_area_ids,
    _build_intersection_clusters,
    _write_connections_xml,
)
from ll2sumo.geometry import polyline_length
from ll2sumo.model import Lanelet, Point3D


def make_lanelet(
    lanelet_id: str,
    *,
    left_node_ids: tuple[str, str],
    right_node_ids: tuple[str, str],
    start_x: float,
    end_x: float,
    tags: dict[str, str] | None = None,
) -> Lanelet:
    lanelet_tags = {"type": "lanelet", "subtype": "road", "speed_limit": "60", "location": "urban", "one_way": "yes"}
    if tags:
        lanelet_tags.update(tags)
    centerline = (Point3D(start_x, 0.0, 0.0), Point3D(end_x, 0.0, 0.0))
    return Lanelet(
        id=lanelet_id,
        subtype="road",
        tags=lanelet_tags,
        left_way_id=f"{lanelet_id}_left",
        right_way_id=f"{lanelet_id}_right",
        regulatory_ids=tuple(),
        left_node_ids=left_node_ids,
        right_node_ids=right_node_ids,
        left_boundary=centerline,
        right_boundary=centerline,
        centerline=centerline,
        start=centerline[0],
        end=centerline[-1],
        avg_heading_deg=0.0,
        length_m=end_x - start_x,
    )


class IntersectionClusterTest(unittest.TestCase):
    def test_cluster_tracks_internal_lanelets_and_movements(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a", left_node_ids=("l0", "l1"), right_node_ids=("r0", "r1"), start_x=0.0, end_x=10.0),
            "b": make_lanelet(
                "b",
                left_node_ids=("l1", "l2"),
                right_node_ids=("r1", "r2"),
                start_x=10.0,
                end_x=20.0,
                tags={"intersection_area": "ia-1", "turn_direction": "straight"},
            ),
            "c": make_lanelet("c", left_node_ids=("l2", "l3"), right_node_ids=("r2", "r3"), start_x=20.0, end_x=30.0),
        }
        successors = {"a": ["b"], "b": ["c"]}
        lanelet_to_group = {"a": "group_a", "b": "group_b", "c": "group_c"}
        collapsed_group_area_ids = {"group_b": "ia-1"}

        clusters = _build_intersection_clusters(road_lanelets, successors, lanelet_to_group, collapsed_group_area_ids)

        self.assertEqual(tuple(clusters), ("ia-1",))
        cluster = clusters["ia-1"]
        self.assertEqual(cluster.group_ids, ("group_b",))
        self.assertEqual(cluster.incoming_lanelet_ids, ("a",))
        self.assertEqual(cluster.outgoing_lanelet_ids, ("c",))
        self.assertEqual(cluster.movement_lanelet_pairs, (("a", "c"),))

    def test_cluster_collapses_internal_edge_into_single_node_and_connection(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a", left_node_ids=("l0", "l1"), right_node_ids=("r0", "r1"), start_x=0.0, end_x=10.0),
            "b": make_lanelet(
                "b",
                left_node_ids=("l1", "l2"),
                right_node_ids=("r1", "r2"),
                start_x=10.0,
                end_x=20.0,
                tags={"intersection_area": "ia-1", "turn_direction": "straight"},
            ),
            "c": make_lanelet("c", left_node_ids=("l2", "l3"), right_node_ids=("r2", "r3"), start_x=20.0, end_x=30.0),
        }
        successors = {"a": ["b"], "b": ["c"]}
        lanelet_to_group = {"a": "group_a", "b": "group_b", "c": "group_c"}
        collapsed_group_area_ids = {"group_b": "ia-1"}
        exported_groups = [
            LaneGroup(
                group_id="group_a",
                edge_id="edge_a",
                lanelet_paths=(("a",),),
                start=road_lanelets["a"].start,
                end=road_lanelets["a"].end,
                centerline=road_lanelets["a"].centerline,
            ),
            LaneGroup(
                group_id="group_c",
                edge_id="edge_c",
                lanelet_paths=(("c",),),
                start=road_lanelets["c"].start,
                end=road_lanelets["c"].end,
                centerline=road_lanelets["c"].centerline,
            ),
        ]
        clusters = _build_intersection_clusters(road_lanelets, successors, lanelet_to_group, collapsed_group_area_ids)

        node_ids, _ = _assign_node_ids(exported_groups, lanelet_to_group, successors, clusters)
        self.assertEqual(node_ids["group_a:end"], node_ids["group_c:start"])

        with tempfile.TemporaryDirectory() as temp_dir:
            connections_path = Path(temp_dir) / "network.con.xml"
            connection_summary = _write_connections_xml(
                connections_path,
                successors,
                lanelet_to_group,
                exported_groups,
                {"a": 0, "c": 0},
                clusters,
                road_lanelets,
            )

            self.assertEqual(connection_summary["connection_count"], 1)
            root = ET.parse(connections_path).getroot()
            connections = [
                (
                    element.attrib["from"],
                    element.attrib["to"],
                    element.attrib["fromLane"],
                    element.attrib["toLane"],
                    element.attrib.get("shape"),
                    element.attrib.get("length"),
                )
                for element in root.findall("connection")
            ]
            self.assertEqual(connections, [("edge_a", "edge_c", "0", "0", "10.000,0.000,0.000 20.000,0.000,0.000", "10.000")])
            self.assertEqual(connection_summary["intersection_area_shape_count"], 1)

    def test_joined_intersection_connection_shape_uses_lanelet_centerline(self) -> None:
        def custom_lanelet(lanelet_id: str, points: tuple[Point3D, ...], tags: dict[str, str] | None = None) -> Lanelet:
            lanelet_tags = {
                "type": "lanelet",
                "subtype": "road",
                "speed_limit": "60",
                "location": "urban",
                "one_way": "yes",
            }
            if tags:
                lanelet_tags.update(tags)
            return Lanelet(
                id=lanelet_id,
                subtype="road",
                tags=lanelet_tags,
                left_way_id=f"{lanelet_id}_left",
                right_way_id=f"{lanelet_id}_right",
                regulatory_ids=tuple(),
                left_node_ids=(f"{lanelet_id}_l0", f"{lanelet_id}_l1"),
                right_node_ids=(f"{lanelet_id}_r0", f"{lanelet_id}_r1"),
                left_boundary=points,
                right_boundary=points,
                centerline=points,
                start=points[0],
                end=points[-1],
                avg_heading_deg=0.0,
                length_m=polyline_length(points),
            )

        road_lanelets = {
            "a": custom_lanelet("a", (Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0))),
            "b": custom_lanelet(
                "b",
                (Point3D(10.0, 0.0, 0.0), Point3D(15.0, 5.0, 0.0), Point3D(20.0, 0.0, 0.0)),
                {"intersection_area": "ia-1", "turn_direction": "left"},
            ),
            "c": custom_lanelet("c", (Point3D(20.0, 0.0, 0.0), Point3D(30.0, 0.0, 0.0))),
        }
        successors = {"a": ["b"], "b": ["c"]}
        lanelet_to_group = {"a": "group_a", "b": "group_b", "c": "group_c"}
        lane_groups = [
            LaneGroup("group_a", "edge_a", (("a",),), road_lanelets["a"].start, road_lanelets["a"].end, road_lanelets["a"].centerline),
            LaneGroup("group_b", "edge_b", (("b",),), road_lanelets["b"].start, road_lanelets["b"].end, road_lanelets["b"].centerline),
            LaneGroup("group_c", "edge_c", (("c",),), road_lanelets["c"].start, road_lanelets["c"].end, road_lanelets["c"].centerline),
        ]
        node_join = IntersectionAreaNodeJoin(
            intersection_area_id="ia-1",
            join_id="ia_ia-1",
            node_ids=("node_a_end", "node_b_start", "node_b_end", "node_c_start"),
            point=Point3D(15.0, 0.0, 0.0),
            shape=(Point3D(9.0, -1.0, 0.0), Point3D(21.0, -1.0, 0.0), Point3D(21.0, 1.0, 0.0), Point3D(9.0, 1.0, 0.0)),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            connections_path = Path(temp_dir) / "network.con.xml"
            connection_summary = _write_connections_xml(
                connections_path,
                successors,
                lanelet_to_group,
                lane_groups,
                {"a": 0, "b": 0, "c": 0},
                {},
                road_lanelets,
                edge_node_ids_by_edge_id={
                    "edge_a": ("node_a_start", "node_a_end"),
                    "edge_b": ("node_b_start", "node_b_end"),
                    "edge_c": ("node_c_start", "node_c_end"),
                },
                intersection_area_node_joins=[node_join],
            )

            self.assertEqual(connection_summary["joined_intersection_area_shape_count"], 1)
            root = ET.parse(connections_path).getroot()
            direct_connection = root.find("connection[@from='edge_a'][@to='edge_c']")
            self.assertIsNotNone(direct_connection)
            assert direct_connection is not None
            self.assertEqual(
                direct_connection.attrib["shape"],
                "10.000,0.000,0.000 15.000,5.000,0.000 20.000,0.000,0.000",
            )

    def test_mixed_group_is_not_collapsed(self) -> None:
        road_lanelets = {
            "a0": make_lanelet("a0", left_node_ids=("l0", "l1"), right_node_ids=("r0", "r1"), start_x=0.0, end_x=10.0),
            "a1": make_lanelet("a1", left_node_ids=("l0b", "l1b"), right_node_ids=("r0b", "r1b"), start_x=0.0, end_x=10.0),
            "b0": make_lanelet(
                "b0",
                left_node_ids=("l1", "l2"),
                right_node_ids=("r1", "r2"),
                start_x=10.0,
                end_x=20.0,
                tags={"intersection_area": "ia-1", "turn_direction": "straight"},
            ),
            "b1": make_lanelet("b1", left_node_ids=("l1b", "l2b"), right_node_ids=("r1b", "r2b"), start_x=10.0, end_x=20.0),
            "c0": make_lanelet("c0", left_node_ids=("l2", "l3"), right_node_ids=("r2", "r3"), start_x=20.0, end_x=30.0),
            "c1": make_lanelet("c1", left_node_ids=("l2b", "l3b"), right_node_ids=("r2b", "r3b"), start_x=20.0, end_x=30.0),
        }
        successors = {"a0": ["b0"], "a1": ["b1"], "b0": ["c0"], "b1": ["c1"]}
        lanelet_to_group = {
            "a0": "group_a",
            "a1": "group_a",
            "b0": "group_b",
            "b1": "group_b",
            "c0": "group_c",
            "c1": "group_c",
        }
        lane_groups = [
            LaneGroup(
                group_id="group_a",
                edge_id="edge_a",
                lanelet_paths=(("a0",), ("a1",)),
                start=road_lanelets["a0"].start,
                end=road_lanelets["a0"].end,
                centerline=road_lanelets["a0"].centerline,
            ),
            LaneGroup(
                group_id="group_b",
                edge_id="edge_b",
                lanelet_paths=(("b0",), ("b1",)),
                start=road_lanelets["b0"].start,
                end=road_lanelets["b0"].end,
                centerline=road_lanelets["b0"].centerline,
            ),
            LaneGroup(
                group_id="group_c",
                edge_id="edge_c",
                lanelet_paths=(("c0",), ("c1",)),
                start=road_lanelets["c0"].start,
                end=road_lanelets["c0"].end,
                centerline=road_lanelets["c0"].centerline,
            ),
        ]

        collapsed_group_area_ids = _collapsed_intersection_group_area_ids(lane_groups, road_lanelets)
        self.assertEqual(collapsed_group_area_ids, {})

        clusters = _build_intersection_clusters(road_lanelets, successors, lanelet_to_group, collapsed_group_area_ids)
        self.assertEqual(clusters, {})

        node_ids, _ = _assign_node_ids(lane_groups, lanelet_to_group, successors, clusters)
        self.assertEqual(node_ids["group_a:end"], node_ids["group_b:start"])
        self.assertEqual(node_ids["group_b:end"], node_ids["group_c:start"])

        with tempfile.TemporaryDirectory() as temp_dir:
            connections_path = Path(temp_dir) / "network.con.xml"
            connection_summary = _write_connections_xml(
                connections_path,
                successors,
                lanelet_to_group,
                lane_groups,
                {"a0": 0, "a1": 1, "b0": 0, "b1": 1, "c0": 0, "c1": 1},
                clusters,
                road_lanelets,
            )

            self.assertEqual(connection_summary["connection_count"], 4)
            root = ET.parse(connections_path).getroot()
            connections = sorted(
                (
                    element.attrib["from"],
                    element.attrib["to"],
                    element.attrib["fromLane"],
                    element.attrib["toLane"],
                )
                for element in root.findall("connection")
            )
            self.assertEqual(
                connections,
                [
                    ("edge_a", "edge_b", "0", "0"),
                    ("edge_a", "edge_b", "1", "1"),
                    ("edge_b", "edge_c", "0", "0"),
                    ("edge_b", "edge_c", "1", "1"),
                ],
            )

    def test_connections_bridge_over_non_exported_lanelets(self) -> None:
        successors = {
            "a": ["b"],
            "b": ["c"],
            "c": ["d"],
        }
        road_lanelets = {
            "a": make_lanelet("a", left_node_ids=("l0", "l1"), right_node_ids=("r0", "r1"), start_x=0.0, end_x=1.0),
            "b": make_lanelet("b", left_node_ids=("l1", "l2"), right_node_ids=("r1", "r2"), start_x=1.0, end_x=1.5),
            "c": make_lanelet("c", left_node_ids=("l2", "l3"), right_node_ids=("r2", "r3"), start_x=1.5, end_x=2.0),
            "d": make_lanelet("d", left_node_ids=("l3", "l4"), right_node_ids=("r3", "r4"), start_x=2.0, end_x=3.0),
        }
        lanelet_to_group = {
            "a": "group_a",
            "b": "group_b",
            "c": "group_c",
            "d": "group_d",
        }
        exported_groups = [
            LaneGroup(
                group_id="group_a",
                edge_id="edge_a",
                lanelet_paths=(("a",),),
                start=Point3D(0.0, 0.0, 0.0),
                end=Point3D(1.0, 0.0, 0.0),
                centerline=(Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0)),
            ),
            LaneGroup(
                group_id="group_d",
                edge_id="edge_d",
                lanelet_paths=(("d",),),
                start=Point3D(2.0, 0.0, 0.0),
                end=Point3D(3.0, 0.0, 0.0),
                centerline=(Point3D(2.0, 0.0, 0.0), Point3D(3.0, 0.0, 0.0)),
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            connections_path = Path(temp_dir) / "network.con.xml"
            connection_summary = _write_connections_xml(
                connections_path,
                successors,
                lanelet_to_group,
                exported_groups,
                {"a": 0, "d": 0},
                {},
                road_lanelets,
            )

            self.assertEqual(connection_summary["connection_count"], 1)
            root = ET.parse(connections_path).getroot()
            connections = [
                (
                    element.attrib["from"],
                    element.attrib["to"],
                    element.attrib["fromLane"],
                    element.attrib["toLane"],
                )
                for element in root.findall("connection")
            ]
            self.assertEqual(connections, [("edge_a", "edge_d", "0", "0")])
            connection = root.find("connection")
            assert connection is not None
            self.assertEqual(connection.attrib["shape"], "1.000,0.000,0.000 2.000,0.000,0.000")
            self.assertEqual(connection_summary["endpoint_bridge_shape_count"], 1)

    def test_same_point_branch_connection_gets_tangent_shape(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a", left_node_ids=("l0", "l1"), right_node_ids=("r0", "r1"), start_x=0.0, end_x=10.0),
            "b": make_lanelet("b", left_node_ids=("l1", "l2"), right_node_ids=("r1", "r2"), start_x=10.0, end_x=20.0),
            "c": make_lanelet(
                "c",
                left_node_ids=("l1", "l3"),
                right_node_ids=("r1", "r3"),
                start_x=10.0,
                end_x=20.0,
                tags={"turn_direction": "right"},
            ),
        }
        successors = {"a": ["b", "c"]}
        lanelet_to_group = {"a": "group_a", "b": "group_b", "c": "group_c"}
        lane_groups = [
            LaneGroup("group_a", "edge_a", (("a",),), road_lanelets["a"].start, road_lanelets["a"].end, road_lanelets["a"].centerline),
            LaneGroup("group_b", "edge_b", (("b",),), road_lanelets["b"].start, road_lanelets["b"].end, road_lanelets["b"].centerline),
            LaneGroup("group_c", "edge_c", (("c",),), road_lanelets["c"].start, road_lanelets["c"].end, road_lanelets["c"].centerline),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            connections_path = Path(temp_dir) / "network.con.xml"
            connection_summary = _write_connections_xml(
                connections_path,
                successors,
                lanelet_to_group,
                lane_groups,
                {"a": 0, "b": 0, "c": 0},
                {},
                road_lanelets,
            )

            self.assertEqual(connection_summary["connection_count"], 2)
            self.assertEqual(connection_summary["tangent_fallback_shape_count"], 2)
            root = ET.parse(connections_path).getroot()
            for connection in root.findall("connection"):
                self.assertIn("shape", connection.attrib)
                self.assertGreater(float(connection.attrib["length"]), 0.0)

    def test_connection_shape_uses_exported_edge_boundary_after_serial_merge(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a", left_node_ids=("l0", "l1"), right_node_ids=("r0", "r1"), start_x=0.0, end_x=10.0),
            "b": make_lanelet("b", left_node_ids=("l1", "l2"), right_node_ids=("r1", "r2"), start_x=10.0, end_x=20.0),
            "c": make_lanelet("c", left_node_ids=("l2", "l3"), right_node_ids=("r2", "r3"), start_x=20.0, end_x=30.0),
        }
        successors = {"a": ["b", "c"], "b": ["c"]}
        lanelet_to_group = {"a": "group_ab", "b": "group_ab", "c": "group_c"}
        lane_groups = [
            LaneGroup(
                "group_ab",
                "edge_ab",
                (("a", "b"),),
                road_lanelets["a"].start,
                road_lanelets["b"].end,
                (Point3D(0.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0)),
            ),
            LaneGroup(
                "group_c",
                "edge_c",
                (("c",),),
                road_lanelets["c"].start,
                road_lanelets["c"].end,
                road_lanelets["c"].centerline,
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            connections_path = Path(temp_dir) / "network.con.xml"
            connection_summary = _write_connections_xml(
                connections_path,
                successors,
                lanelet_to_group,
                lane_groups,
                {"a": 0, "b": 0, "c": 0},
                {},
                road_lanelets,
            )

            self.assertEqual(connection_summary["connection_count"], 1)
            connection = ET.parse(connections_path).getroot().find("connection")
            assert connection is not None
            self.assertTrue(connection.attrib["shape"].startswith("20.000,0.000,0.000 "))

    def test_far_endpoint_connection_uses_bounded_tangent_shape(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a", left_node_ids=("l0", "l1"), right_node_ids=("r0", "r1"), start_x=0.0, end_x=10.0),
            "b": make_lanelet("b", left_node_ids=("l9", "l10"), right_node_ids=("r9", "r10"), start_x=1000.0, end_x=1010.0),
        }
        successors = {"a": ["b"]}
        lanelet_to_group = {"a": "group_a", "b": "group_b"}
        lane_groups = [
            LaneGroup("group_a", "edge_a", (("a",),), road_lanelets["a"].start, road_lanelets["a"].end, road_lanelets["a"].centerline),
            LaneGroup("group_b", "edge_b", (("b",),), road_lanelets["b"].start, road_lanelets["b"].end, road_lanelets["b"].centerline),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            connections_path = Path(temp_dir) / "network.con.xml"
            connection_summary = _write_connections_xml(
                connections_path,
                successors,
                lanelet_to_group,
                lane_groups,
                {"a": 0, "b": 0},
                {},
                road_lanelets,
            )

            self.assertEqual(connection_summary["endpoint_bridge_shape_count"], 0)
            self.assertEqual(connection_summary["tangent_fallback_shape_count"], 1)
            connection = ET.parse(connections_path).getroot().find("connection")
            assert connection is not None
            self.assertLessEqual(float(connection.attrib["length"]), 3.0)

    def test_unshapeable_connection_is_reported(self) -> None:
        successors = {"a": ["b"]}
        lanelet_to_group = {"a": "group_a", "b": "group_b"}
        lane_groups = [
            LaneGroup("group_a", "edge_a", (("a",),), Point3D(0.0, 0.0, 0.0), Point3D(0.0, 0.0, 0.0), (Point3D(0.0, 0.0, 0.0),)),
            LaneGroup("group_b", "edge_b", (("b",),), Point3D(0.0, 0.0, 0.0), Point3D(0.0, 0.0, 0.0), (Point3D(0.0, 0.0, 0.0),)),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            connections_path = Path(temp_dir) / "network.con.xml"
            connection_summary = _write_connections_xml(
                connections_path,
                successors,
                lanelet_to_group,
                lane_groups,
                {"a": 0, "b": 0},
                {},
            )

            self.assertEqual(connection_summary["connection_count"], 1)
            self.assertEqual(connection_summary["unshaped_connection_count"], 1)
            connection = ET.parse(connections_path).getroot().find("connection")
            assert connection is not None
            self.assertNotIn("shape", connection.attrib)

    def test_adjacent_collapsed_intersection_clusters_share_export_node(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a", left_node_ids=("l0", "l1"), right_node_ids=("r0", "r1"), start_x=0.0, end_x=10.0),
            "b": make_lanelet(
                "b",
                left_node_ids=("l1", "l2"),
                right_node_ids=("r1", "r2"),
                start_x=10.0,
                end_x=20.0,
                tags={"intersection_area": "ia-1"},
            ),
            "c": make_lanelet(
                "c",
                left_node_ids=("l2", "l3"),
                right_node_ids=("r2", "r3"),
                start_x=20.0,
                end_x=30.0,
                tags={"intersection_area": "ia-2"},
            ),
            "d": make_lanelet("d", left_node_ids=("l3", "l4"), right_node_ids=("r3", "r4"), start_x=30.0, end_x=40.0),
        }
        successors = {"a": ["b"], "b": ["c"], "c": ["d"]}
        lanelet_to_group = {"a": "group_a", "b": "group_b", "c": "group_c", "d": "group_d"}
        collapsed_group_area_ids = {"group_b": "ia-1", "group_c": "ia-2"}
        exported_groups = [
            LaneGroup(
                group_id="group_a",
                edge_id="edge_a",
                lanelet_paths=(("a",),),
                start=road_lanelets["a"].start,
                end=road_lanelets["a"].end,
                centerline=road_lanelets["a"].centerline,
            ),
            LaneGroup(
                group_id="group_d",
                edge_id="edge_d",
                lanelet_paths=(("d",),),
                start=road_lanelets["d"].start,
                end=road_lanelets["d"].end,
                centerline=road_lanelets["d"].centerline,
            ),
        ]
        clusters = _build_intersection_clusters(road_lanelets, successors, lanelet_to_group, collapsed_group_area_ids)

        node_ids, _ = _assign_node_ids(exported_groups, lanelet_to_group, successors, clusters)
        self.assertEqual(node_ids["group_a:end"], node_ids["group_d:start"])
        self.assertEqual(node_ids["cluster:ia-1"], node_ids["cluster:ia-2"])

        with tempfile.TemporaryDirectory() as temp_dir:
            connections_path = Path(temp_dir) / "network.con.xml"
            connection_summary = _write_connections_xml(
                connections_path,
                successors,
                lanelet_to_group,
                exported_groups,
                {"a": 0, "d": 0},
                clusters,
                road_lanelets,
                edge_node_ids_by_edge_id={
                    "edge_a": (node_ids["group_a:start"], node_ids["group_a:end"]),
                    "edge_d": (node_ids["group_d:start"], node_ids["group_d:end"]),
                },
            )

            self.assertEqual(connection_summary["connection_count"], 1)
            root = ET.parse(connections_path).getroot()
            connections = [
                (
                    element.attrib["from"],
                    element.attrib["to"],
                    element.attrib["fromLane"],
                    element.attrib["toLane"],
                )
                for element in root.findall("connection")
            ]
            self.assertEqual(connections, [("edge_a", "edge_d", "0", "0")])


if __name__ == "__main__":
    unittest.main()
