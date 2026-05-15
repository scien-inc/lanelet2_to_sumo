from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch
import unittest

from ll2sumo.convert import (
    IntersectionCluster,
    _lanelet_path_signal_stop_offset_m,
    _plan_vehicle_signals,
    _run_netconvert,
    _signalized_intersection_area_ids,
)
from ll2sumo.model import Lanelet, LaneletMap, Point3D, RegulatoryElement, Way


def make_lanelet(
    lanelet_id: str,
    *,
    tags: dict[str, str] | None = None,
    regulatory_ids: tuple[str, ...] = (),
    heading_deg: float = 0.0,
) -> Lanelet:
    lanelet_tags = {"type": "lanelet", "subtype": "road", "one_way": "yes", "speed_limit": "50"}
    if tags:
        lanelet_tags.update(tags)
    centerline = (Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0))
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
        avg_heading_deg=heading_deg,
        length_m=10.0,
    )


class VehicleSignalPlanningTest(unittest.TestCase):
    def test_attached_lanelet_marks_cluster_as_signalized(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a", regulatory_ids=("r1",)),
            "b": make_lanelet("b"),
            "c": make_lanelet("c", tags={"intersection_area": "ia"}),
            "d": make_lanelet("d"),
        }
        lanelet_map = LaneletMap(
            nodes={},
            ways={
                "refers": Way(id="refers", node_ids=("n0", "n1"), tags={"subtype": "red_yellow_green"}),
                "ref_line": Way(id="ref_line", node_ids=("n2", "n3"), tags={}),
            },
            lanelets=road_lanelets,
            regulatory_elements={
                "r1": RegulatoryElement(
                    id="r1",
                    subtype="traffic_light",
                    tags={"subtype": "traffic_light"},
                    members_by_role={"refers": ("refers",), "ref_line": ("ref_line",)},
                )
            },
        )
        tls_ids_by_node_id, signal_summary, unmapped = _plan_vehicle_signals(
            lanelet_map,
            road_lanelets,
            {"a": ["b"], "b": ["c"], "c": ["d"]},
            {
                "ia": IntersectionCluster(
                    intersection_area_id="ia",
                    lanelet_ids=("c",),
                    group_ids=("group_c",),
                    incoming_lanelet_ids=("b",),
                    outgoing_lanelet_ids=("d",),
                    movement_lanelet_pairs=(("b", "d"),),
                    centroid=Point3D(0.0, 0.0, 0.0),
                )
            },
            {"cluster:ia": "node_tls"},
            {"a": "group_a", "b": "group_b", "c": "group_c", "d": "group_d"},
        )

        self.assertEqual(tls_ids_by_node_id, {"node_tls": "tls_ia"})
        self.assertEqual(unmapped, [])
        self.assertEqual(signal_summary["normalized_head_count"], 1)
        self.assertEqual(signal_summary["normalized_control_count"], 1)
        self.assertEqual(signal_summary["tls_cluster_count"], 0)
        self.assertEqual(signal_summary["inferred_no_refline_count"], 0)

    def test_attached_lanelet_marks_stopline_node_when_signalized_area_is_exported(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a", regulatory_ids=("r1",)),
            "c": make_lanelet("c", tags={"intersection_area": "ia"}),
            "d": make_lanelet("d"),
        }
        lanelet_map = LaneletMap(
            nodes={},
            ways={
                "refers": Way(id="refers", node_ids=("n0", "n1"), tags={"subtype": "red_yellow_green"}),
                "ref_line": Way(id="ref_line", node_ids=("n2", "n3"), tags={"subtype": "stop_line"}),
            },
            lanelets=road_lanelets,
            regulatory_elements={
                "r1": RegulatoryElement(
                    id="r1",
                    subtype="traffic_light",
                    tags={"subtype": "traffic_light"},
                    members_by_role={"refers": ("refers",), "ref_line": ("ref_line",)},
                )
            },
        )

        tls_ids_by_node_id, signal_summary, unmapped = _plan_vehicle_signals(
            lanelet_map,
            road_lanelets,
            {"a": ["c"], "c": ["d"]},
            {},
            {"group_a:end": "node_stopline", "group_c:start": "node_stopline"},
            {"a": "group_a", "c": "group_c", "d": "group_d"},
        )

        self.assertEqual(tls_ids_by_node_id, {"node_stopline": "tls_ia"})
        self.assertEqual(unmapped, [])
        self.assertEqual(signal_summary["signalized_node_count"], 1)

    def test_refline_near_lanelet_start_marks_start_node(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a", regulatory_ids=("r1",)),
            "c": make_lanelet("c", tags={"intersection_area": "ia"}),
        }
        lanelet_map = LaneletMap(
            nodes={
                "n2": Point3D(0.0, -1.0, 0.0),
                "n3": Point3D(0.0, 1.0, 0.0),
            },
            ways={
                "refers": Way(id="refers", node_ids=("n0", "n1"), tags={"subtype": "red_yellow_green"}),
                "ref_line": Way(id="ref_line", node_ids=("n2", "n3"), tags={"subtype": "stop_line"}),
            },
            lanelets=road_lanelets,
            regulatory_elements={
                "r1": RegulatoryElement(
                    id="r1",
                    subtype="traffic_light",
                    tags={"subtype": "traffic_light"},
                    members_by_role={"refers": ("refers",), "ref_line": ("ref_line",)},
                )
            },
        )

        tls_ids_by_node_id, _, unmapped = _plan_vehicle_signals(
            lanelet_map,
            road_lanelets,
            {"a": ["c"]},
            {},
            {"group_a:start": "node_stopline", "group_a:end": "node_wrong"},
            {"a": "group_a", "c": "group_c"},
        )

        self.assertEqual(tls_ids_by_node_id, {"node_stopline": "tls_ia"})
        self.assertEqual(unmapped, [])

    def test_refline_inside_lanelet_uses_downstream_node_with_stop_offset(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a", regulatory_ids=("r1",)),
            "c": make_lanelet("c", tags={"intersection_area": "ia"}),
        }
        lanelet_map = LaneletMap(
            nodes={
                "n2": Point3D(3.0, -1.0, 0.0),
                "n3": Point3D(3.0, 1.0, 0.0),
            },
            ways={
                "refers": Way(id="refers", node_ids=("n0", "n1"), tags={"subtype": "red_yellow_green"}),
                "ref_line": Way(id="ref_line", node_ids=("n2", "n3"), tags={"subtype": "stop_line"}),
            },
            lanelets=road_lanelets,
            regulatory_elements={
                "r1": RegulatoryElement(
                    id="r1",
                    subtype="traffic_light",
                    tags={"subtype": "traffic_light"},
                    members_by_role={"refers": ("refers",), "ref_line": ("ref_line",)},
                )
            },
        )

        tls_ids_by_node_id, _, unmapped = _plan_vehicle_signals(
            lanelet_map,
            road_lanelets,
            {"a": ["c"]},
            {},
            {"group_a:start": "node_wrong", "group_a:end": "node_tls"},
            {"a": "group_a", "c": "group_c"},
        )

        self.assertEqual(tls_ids_by_node_id, {"node_tls": "tls_ia"})
        self.assertAlmostEqual(_lanelet_path_signal_stop_offset_m(("a",), lanelet_map, road_lanelets), 7.0)
        self.assertEqual(unmapped, [])

    def test_signalized_intersection_areas_are_not_collapsed(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a", regulatory_ids=("r1",)),
            "c": make_lanelet("c", tags={"intersection_area": "ia"}),
            "d": make_lanelet("d"),
        }
        lanelet_map = LaneletMap(
            nodes={},
            ways={"refers": Way(id="refers", node_ids=("n0", "n1"), tags={"subtype": "red_yellow_green"})},
            lanelets=road_lanelets,
            regulatory_elements={
                "r1": RegulatoryElement(
                    id="r1",
                    subtype="traffic_light",
                    tags={"subtype": "traffic_light"},
                    members_by_role={"refers": ("refers",), "ref_line": ("ref_line",)},
                )
            },
        )

        self.assertEqual(
            _signalized_intersection_area_ids(lanelet_map, road_lanelets, {"a": ["c"], "c": ["d"]}),
            {"ia"},
        )


class NetconvertTlsBuildingTest(unittest.TestCase):
    def test_uses_lanelet_stopline_tls_building_options(self) -> None:
        with patch("ll2sumo.convert.subprocess.run") as run:
            run.return_value = CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            _run_netconvert(
                Path("network.nod.xml"),
                Path("network.edg.xml"),
                Path("network.con.xml"),
                Path("network.net.xml"),
                "netconvert",
                build_tls_from_nodes=True,
            )

        command = run.call_args.args[0]
        self.assertIn("--junctions.join", command)
        self.assertIn("--junctions.minimal-shape", command)
        self.assertIn("--tls.discard-simple", command)
        self.assertIn("--tls.join", command)
        self.assertIn("--tls.default-type", command)
        self.assertNotIn("--tls.guess-signals", command)
        self.assertNotIn("--tllogic-files", command)


if __name__ == "__main__":
    unittest.main()
