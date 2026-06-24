from __future__ import annotations

import json
import tempfile
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch
import unittest
import xml.etree.ElementTree as ET

from ll2sumo.convert import (
    IntersectionCluster,
    LaneGroup,
    _build_sumo_link_signal_mapping_records,
    _build_intersection_area_node_joins,
    _lanelet_path_signal_stop_offset_m,
    _plan_vehicle_signals,
    _resolve_signal_mapping_records,
    _run_netconvert,
    _run_netconvert_connection_patch,
    _signalized_intersection_area_ids,
    _write_signal_id_mapping_json,
    _write_nodes_xml,
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
        tls_ids_by_node_id, signal_summary, unmapped, mapping_records = _plan_vehicle_signals(
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
        self.assertEqual(len(mapping_records), 1)
        self.assertEqual(mapping_records[0]["lanelet_regulatory_element_ids"], ["r1"])
        self.assertEqual(mapping_records[0]["lanelet_traffic_light_way_ids"], ["refers"])
        self.assertEqual(mapping_records[0]["lanelet_ref_line_way_ids"], ["ref_line"])
        self.assertEqual(mapping_records[0]["planned_sumo_tls_id"], "tls_ia")
        self.assertEqual(mapping_records[0]["planned_sumo_node_ids"], ["node_tls"])
        self.assertEqual(mapping_records[0]["resolution_status"], "planned_only")

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

        tls_ids_by_node_id, signal_summary, unmapped, _ = _plan_vehicle_signals(
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

        tls_ids_by_node_id, _, unmapped, _ = _plan_vehicle_signals(
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

        tls_ids_by_node_id, _, unmapped, _ = _plan_vehicle_signals(
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

    def test_vehicle_signal_without_reachable_intersection_uses_standalone_tls(self) -> None:
        road_lanelets = {
            "a": make_lanelet("a", regulatory_ids=("r1",)),
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

        tls_ids_by_node_id, signal_summary, unmapped, mapping_records = _plan_vehicle_signals(
            lanelet_map,
            road_lanelets,
            {},
            {},
            {"group_a:start": "node_tls"},
            {"a": "group_a"},
        )

        self.assertEqual(tls_ids_by_node_id, {"node_tls": "tls_signal_r1"})
        self.assertEqual(unmapped, [])
        self.assertEqual(signal_summary["unmapped_relation_count"], 0)
        self.assertEqual(mapping_records[0]["planned_sumo_tls_id"], "tls_signal_r1")
        self.assertEqual(mapping_records[0]["planned_sumo_node_ids"], ["node_tls"])
        self.assertEqual(mapping_records[0]["resolution_status"], "planned_only")
        self.assertEqual(mapping_records[0]["reason"], "standalone_signal_no_reachable_intersection_area")

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

    def test_signalized_intersection_area_node_join_preserves_tls_nodes(self) -> None:
        road_lanelets = {
            "c": make_lanelet("c", tags={"intersection_area": "100"}),
        }
        exported_groups = [
            LaneGroup(
                group_id="group_c",
                edge_id="edge_c",
                lanelet_paths=(("c",),),
                start=road_lanelets["c"].start,
                end=road_lanelets["c"].end,
                centerline=road_lanelets["c"].centerline,
            )
        ]
        lanelet_map = LaneletMap(
            nodes={
                "p0": Point3D(0.0, -2.0, 0.0),
                "p1": Point3D(10.0, -2.0, 0.0),
                "p2": Point3D(10.0, 2.0, 0.0),
                "p3": Point3D(0.0, 2.0, 0.0),
                "p4": Point3D(0.0, -2.0, 0.0),
            },
            ways={"100": Way(id="100", node_ids=("p0", "p1", "p2", "p3", "p4"), tags={"type": "intersection_area"})},
            lanelets=road_lanelets,
        )

        joins, summary = _build_intersection_area_node_joins(
            exported_groups,
            road_lanelets,
            lanelet_map,
            {"group_c:start": "node_1", "group_c:end": "node_2"},
            {
                "node_1": Point3D(0.0, 0.0, 0.0),
                "node_2": Point3D(10.0, 0.0, 0.0),
            },
            {"100"},
        )

        self.assertEqual(summary["join_count"], 1)
        self.assertEqual(joins[0].join_id, "ia_100")
        self.assertEqual(joins[0].node_ids, ("node_1", "node_2"))

        with tempfile.TemporaryDirectory() as temp_dir:
            nodes_path = Path(temp_dir) / "network.nod.xml"
            _write_nodes_xml(
                nodes_path,
                {
                    "node_1": Point3D(0.0, 0.0, 0.0),
                    "node_2": Point3D(10.0, 0.0, 0.0),
                },
                tls_ids_by_node_id={"node_1": "tls_100"},
                intersection_area_node_joins=joins,
            )

            root = ET.parse(nodes_path).getroot()
            join = root.find("join")
            self.assertIsNotNone(join)
            assert join is not None
            self.assertEqual(join.attrib["id"], "ia_100")
            self.assertEqual(join.attrib["nodes"], "node_1 node_2")
            self.assertEqual(join.attrib["shape"], "0.000,-2.000 10.000,-2.000 10.000,2.000 0.000,2.000 0.000,-2.000")
            self.assertNotIn("type", join.attrib)
            self.assertNotIn("tl", join.attrib)
            self.assertEqual(root.find("node[@id='node_1']").attrib["tl"], "tls_100")

    def test_signal_mapping_resolves_joined_cluster_tls_from_intersection_area_via(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <junction id="ia_300007" type="traffic_light" x="0" y="0" incLanes="edge_in_0" intLanes=":ia_300007_0_0"/>
  <tlLogic id="cluster_node_a_node_b" type="static" programID="0" offset="0">
    <phase duration="35" state="G"/>
  </tlLogic>
  <connection from="edge_in" to="edge_out" fromLane="0" toLane="0" via=":ia_300007_0_0" tl="cluster_node_a_node_b" linkIndex="0"/>
</net>
""",
                encoding="utf-8",
            )
            records = [
                {
                    "lanelet_regulatory_element_ids": ["reg_1"],
                    "lanelet_traffic_light_way_ids": ["signal_way_1"],
                    "lanelet_ref_line_way_ids": ["stop_line_1"],
                    "attached_lanelet_ids": ["lanelet_1"],
                    "intersection_area_id": "300007",
                    "planned_sumo_tls_id": "tls_300007",
                    "planned_sumo_node_ids": ["node_a", "node_b"],
                    "actual_sumo_tls_ids": [],
                    "actual_sumo_junction_ids": [],
                    "actual_sumo_connection_count": 0,
                    "resolution_status": "planned_only",
                }
            ]

            resolved = _resolve_signal_mapping_records(records, net_path)

            self.assertEqual(resolved[0]["resolution_status"], "mapped")
            self.assertEqual(resolved[0]["actual_sumo_tls_ids"], ["cluster_node_a_node_b"])
            self.assertEqual(resolved[0]["actual_sumo_junction_ids"], ["ia_300007"])
            self.assertEqual(resolved[0]["actual_sumo_connection_count"], 1)

    def test_signal_mapping_resolves_tls_from_stopline_node_via_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <junction id="node_stopline" type="traffic_light" x="0" y="0" incLanes="edge_in_0" intLanes=":node_stopline_0_0"/>
  <tlLogic id="cluster_node_stopline_node_other" type="static" programID="0" offset="0">
    <phase duration="35" state="G"/>
  </tlLogic>
  <connection from="edge_in" to="edge_out" fromLane="0" toLane="0" via=":node_stopline_0_0" tl="cluster_node_stopline_node_other" linkIndex="0"/>
</net>
""",
                encoding="utf-8",
            )
            records = [
                {
                    "lanelet_regulatory_element_ids": ["reg_1"],
                    "lanelet_traffic_light_way_ids": ["signal_way_1"],
                    "lanelet_ref_line_way_ids": ["stop_line_1"],
                    "attached_lanelet_ids": ["lanelet_1"],
                    "intersection_area_id": "300007",
                    "planned_sumo_tls_id": "tls_300007",
                    "planned_sumo_node_ids": ["node_stopline"],
                    "actual_sumo_tls_ids": [],
                    "actual_sumo_junction_ids": [],
                    "actual_sumo_connection_count": 0,
                    "resolution_status": "planned_only",
                }
            ]

            resolved = _resolve_signal_mapping_records(records, net_path)

            self.assertEqual(resolved[0]["resolution_status"], "mapped")
            self.assertEqual(resolved[0]["actual_sumo_tls_ids"], ["cluster_node_stopline_node_other"])
            self.assertEqual(resolved[0]["actual_sumo_junction_ids"], ["node_stopline"])

    def test_signal_mapping_unions_joined_intersection_and_stopline_tls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <junction id="ia_300007" type="traffic_light" x="0" y="0" incLanes="edge_a_0" intLanes=":ia_300007_0_0"/>
  <junction id="node_stopline" type="traffic_light" x="10" y="0" incLanes="edge_b_0" intLanes=":node_stopline_0_0"/>
  <tlLogic id="cluster_node_a_node_b" type="static" programID="0" offset="0">
    <phase duration="35" state="G"/>
  </tlLogic>
  <tlLogic id="tls_300007" type="static" programID="0" offset="0">
    <phase duration="35" state="G"/>
  </tlLogic>
  <connection from="edge_a" to="edge_out" fromLane="0" toLane="0" via=":ia_300007_0_0" tl="cluster_node_a_node_b" linkIndex="0"/>
  <connection from="edge_b" to="edge_out" fromLane="0" toLane="0" via=":node_stopline_0_0" tl="tls_300007" linkIndex="0"/>
</net>
""",
                encoding="utf-8",
            )
            records = [
                {
                    "lanelet_regulatory_element_ids": ["reg_1"],
                    "lanelet_traffic_light_way_ids": ["signal_way_1"],
                    "lanelet_ref_line_way_ids": ["stop_line_1"],
                    "attached_lanelet_ids": ["lanelet_1"],
                    "intersection_area_id": "300007",
                    "planned_sumo_tls_id": "tls_300007",
                    "planned_sumo_node_ids": ["node_stopline"],
                    "actual_sumo_tls_ids": [],
                    "actual_sumo_junction_ids": [],
                    "actual_sumo_connection_count": 0,
                    "resolution_status": "planned_only",
                }
            ]

            resolved = _resolve_signal_mapping_records(records, net_path)

            self.assertEqual(resolved[0]["resolution_status"], "mapped")
            self.assertEqual(resolved[0]["actual_sumo_tls_ids"], ["cluster_node_a_node_b", "tls_300007"])
            self.assertEqual(resolved[0]["actual_sumo_junction_ids"], ["ia_300007", "node_stopline"])
            self.assertEqual(resolved[0]["actual_sumo_connection_count"], 2)

    def test_signal_link_mapping_matches_source_lanelet_to_regulatory_element(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <tlLogic id="cluster_tls" type="static" programID="0" offset="0">
    <phase duration="35" state="GG"/>
  </tlLogic>
  <connection from="edge_a" to="edge_out" fromLane="0" toLane="0" via=":ia_1_0_0" tl="cluster_tls" linkIndex="0" dir="s"/>
  <connection from="edge_b" to="edge_out" fromLane="0" toLane="0" via=":ia_1_1_0" tl="cluster_tls" linkIndex="1" dir="r"/>
</net>
""",
                encoding="utf-8",
            )
            records = [
                {
                    "lanelet_regulatory_element_ids": ["reg_a"],
                    "lanelet_traffic_light_way_ids": ["signal_way_a"],
                    "lanelet_ref_line_way_ids": ["stop_line_a"],
                    "attached_lanelet_ids": ["lanelet_a"],
                    "intersection_area_id": "1",
                    "planned_sumo_tls_id": "tls_1",
                    "planned_sumo_node_ids": ["node_a"],
                    "actual_sumo_tls_ids": ["cluster_tls"],
                    "actual_sumo_junction_ids": ["ia_1"],
                    "actual_sumo_connection_count": 2,
                    "resolution_status": "mapped",
                },
                {
                    "lanelet_regulatory_element_ids": ["reg_b"],
                    "lanelet_traffic_light_way_ids": ["signal_way_b"],
                    "lanelet_ref_line_way_ids": ["stop_line_b"],
                    "attached_lanelet_ids": ["lanelet_b"],
                    "intersection_area_id": "1",
                    "planned_sumo_tls_id": "tls_1",
                    "planned_sumo_node_ids": ["node_b"],
                    "actual_sumo_tls_ids": ["cluster_tls"],
                    "actual_sumo_junction_ids": ["ia_1"],
                    "actual_sumo_connection_count": 2,
                    "resolution_status": "mapped",
                },
            ]

            link_records = _build_sumo_link_signal_mapping_records(
                records,
                net_path,
                {
                    ("edge_a", "0"): ("lanelet_a", "lanelet_a_next"),
                    ("edge_b", "0"): ("lanelet_b",),
                },
            )

            self.assertEqual(len(link_records), 2)
            self.assertEqual(link_records[0]["linkIndex"], 0)
            self.assertEqual(link_records[0]["lanelet_regulatory_element_ids"], ["reg_a"])
            self.assertEqual(link_records[0]["lanelet_traffic_light_way_ids"], ["signal_way_a"])
            self.assertEqual(link_records[0]["match_status"], "matched")
            self.assertEqual(link_records[1]["linkIndex"], 1)
            self.assertEqual(link_records[1]["lanelet_regulatory_element_ids"], ["reg_b"])
            self.assertEqual(link_records[1]["lanelet_traffic_light_way_ids"], ["signal_way_b"])
            self.assertEqual(link_records[1]["match_status"], "matched")

    def test_signal_link_mapping_falls_back_to_tls_group_when_source_lanelet_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            net_path = temp_path / "network.net.xml"
            mapping_path = temp_path / "signal_id_mapping.json"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <tlLogic id="cluster_tls" type="static" programID="0" offset="0">
    <phase duration="35" state="G"/>
  </tlLogic>
  <connection from="edge_unknown" to="edge_out" fromLane="0" toLane="0" via=":ia_1_0_0" tl="cluster_tls" linkIndex="0"/>
</net>
""",
                encoding="utf-8",
            )
            records = [
                {
                    "lanelet_regulatory_element_ids": ["reg_a"],
                    "lanelet_traffic_light_way_ids": ["signal_way_a"],
                    "lanelet_ref_line_way_ids": ["stop_line_a"],
                    "attached_lanelet_ids": ["lanelet_a"],
                    "intersection_area_id": "1",
                    "planned_sumo_tls_id": "tls_1",
                    "planned_sumo_node_ids": ["node_a"],
                    "actual_sumo_tls_ids": ["cluster_tls"],
                    "actual_sumo_junction_ids": ["ia_1"],
                    "actual_sumo_connection_count": 1,
                    "resolution_status": "mapped",
                }
            ]

            link_records = _build_sumo_link_signal_mapping_records(records, net_path, {})
            summary = _write_signal_id_mapping_json(mapping_path, "jp-static", records, link_records, net_path)
            document = json.loads(mapping_path.read_text(encoding="utf-8"))

            self.assertEqual(link_records[0]["match_status"], "tls_group_fallback")
            self.assertEqual(summary["fallback_sumo_link_count"], 1)
            self.assertEqual(document["lanelet_signal_to_sumo_links"]["signal_way_a"][0]["match_status"], "tls_group_fallback")

    def test_signal_link_mapping_indexes_by_refers_way_and_audits_phase_mixing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            net_path = temp_path / "network.net.xml"
            mapping_path = temp_path / "signal_id_mapping.json"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <tlLogic id="cluster_tls" type="static" programID="0" offset="0">
    <phase duration="35" state="Gr"/>
  </tlLogic>
  <connection from="edge_a" to="edge_out" fromLane="0" toLane="0" via=":ia_1_0_0" tl="cluster_tls" linkIndex="0"/>
  <connection from="edge_b" to="edge_out" fromLane="0" toLane="0" via=":ia_1_1_0" tl="cluster_tls" linkIndex="1"/>
</net>
""",
                encoding="utf-8",
            )
            records = [
                {
                    "lanelet_regulatory_element_ids": ["reg_a"],
                    "lanelet_traffic_light_way_ids": ["signal_way_a"],
                    "lanelet_ref_line_way_ids": ["stop_line_a"],
                    "attached_lanelet_ids": ["lanelet_a", "lanelet_b"],
                    "intersection_area_id": "1",
                    "planned_sumo_tls_id": "tls_1",
                    "planned_sumo_node_ids": ["node_a"],
                    "actual_sumo_tls_ids": ["cluster_tls"],
                    "actual_sumo_junction_ids": ["ia_1"],
                    "actual_sumo_connection_count": 2,
                    "resolution_status": "mapped",
                }
            ]

            link_records = _build_sumo_link_signal_mapping_records(
                records,
                net_path,
                {
                    ("edge_a", "0"): ("lanelet_a",),
                    ("edge_b", "0"): ("lanelet_b",),
                },
            )
            summary = _write_signal_id_mapping_json(mapping_path, "jp-static", records, link_records, net_path)
            document = json.loads(mapping_path.read_text(encoding="utf-8"))

            self.assertEqual(summary["matched_sumo_link_count"], 2)
            self.assertEqual(summary["mixed_lanelet_signal_phase_count"], 1)
            self.assertEqual(
                [entry["linkIndex"] for entry in document["lanelet_signal_to_sumo_links"]["signal_way_a"]],
                [0, 1],
            )
            self.assertEqual(
                document["summary"]["examples"][0]["type"],
                "mixed_lanelet_signal_phase",
            )

    def test_signal_mapping_json_groups_multiple_lanelet_signals_by_sumo_tls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "signal_id_mapping.json"
            records = [
                {
                    "lanelet_regulatory_element_ids": ["reg_1"],
                    "lanelet_traffic_light_way_ids": ["signal_way_1"],
                    "lanelet_ref_line_way_ids": ["stop_line_1"],
                    "attached_lanelet_ids": ["lanelet_1"],
                    "intersection_area_id": "300007",
                    "planned_sumo_tls_id": "tls_300007",
                    "planned_sumo_node_ids": ["node_a"],
                    "actual_sumo_tls_ids": ["cluster_tls"],
                    "actual_sumo_junction_ids": ["ia_300007"],
                    "actual_sumo_connection_count": 2,
                    "resolution_status": "mapped",
                },
                {
                    "lanelet_regulatory_element_ids": ["reg_2"],
                    "lanelet_traffic_light_way_ids": ["signal_way_2"],
                    "lanelet_ref_line_way_ids": ["stop_line_2"],
                    "attached_lanelet_ids": ["lanelet_2"],
                    "intersection_area_id": "300007",
                    "planned_sumo_tls_id": "tls_300007",
                    "planned_sumo_node_ids": ["node_b"],
                    "actual_sumo_tls_ids": ["cluster_tls"],
                    "actual_sumo_junction_ids": ["ia_300007"],
                    "actual_sumo_connection_count": 2,
                    "resolution_status": "mapped",
                },
            ]

            summary = _write_signal_id_mapping_json(mapping_path, "jp-static", records)
            document = json.loads(mapping_path.read_text(encoding="utf-8"))

            self.assertEqual(summary["mapped_record_count"], 2)
            self.assertEqual(summary["actual_sumo_tls_count"], 1)
            self.assertEqual(summary["lanelet_regulatory_element_count"], 2)
            self.assertEqual(document["schema_version"], 2)
            self.assertEqual(document["sumo_link_to_lanelet_signal"], [])
            self.assertEqual(document["lanelet_signal_to_sumo_links"], {})
            self.assertEqual(
                document["sumo_to_lanelet"]["cluster_tls"]["lanelet_regulatory_element_ids"],
                ["reg_1", "reg_2"],
            )
            self.assertEqual(
                document["sumo_to_lanelet"]["cluster_tls"]["lanelet_traffic_light_way_ids"],
                ["signal_way_1", "signal_way_2"],
            )
            self.assertEqual(document["sumo_to_lanelet"]["cluster_tls"]["actual_sumo_connection_count"], 2)

    def test_signal_mapping_keeps_unmapped_and_planned_only_statuses(self) -> None:
        records = [
            {
                "lanelet_regulatory_element_ids": ["reg_unmapped"],
                "lanelet_traffic_light_way_ids": ["signal_way_unmapped"],
                "lanelet_ref_line_way_ids": [],
                "attached_lanelet_ids": [],
                "intersection_area_id": None,
                "planned_sumo_tls_id": None,
                "planned_sumo_node_ids": [],
                "actual_sumo_tls_ids": [],
                "actual_sumo_junction_ids": [],
                "actual_sumo_connection_count": 0,
                "resolution_status": "unmapped",
                "reason": "no_attached_road_lanelets",
            },
            {
                "lanelet_regulatory_element_ids": ["reg_planned"],
                "lanelet_traffic_light_way_ids": ["signal_way_planned"],
                "lanelet_ref_line_way_ids": ["stop_line_planned"],
                "attached_lanelet_ids": ["lanelet_1"],
                "intersection_area_id": "300007",
                "planned_sumo_tls_id": "tls_300007",
                "planned_sumo_node_ids": ["node_a"],
                "actual_sumo_tls_ids": [],
                "actual_sumo_junction_ids": [],
                "actual_sumo_connection_count": 0,
                "resolution_status": "planned_only",
            },
        ]

        resolved = _resolve_signal_mapping_records(records, None)

        self.assertEqual(resolved[0]["resolution_status"], "unmapped")
        self.assertEqual(resolved[0]["reason"], "no_attached_road_lanelets")
        self.assertEqual(resolved[1]["resolution_status"], "planned_only")


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

    def test_connection_patch_uses_existing_net_and_delete_connections(self) -> None:
        with patch("ll2sumo.convert.subprocess.run") as run:
            run.return_value = CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            _run_netconvert_connection_patch(
                Path("network.net.xml"),
                Path("network.joined-delete.con.xml"),
                Path("network.filtered.net.xml"),
                "netconvert",
            )

        command = run.call_args.args[0]
        self.assertIn("--sumo-net-file", command)
        self.assertIn("network.net.xml", command)
        self.assertIn("--connection-files", command)
        self.assertIn("network.joined-delete.con.xml", command)
        self.assertIn("--output-file", command)
        self.assertIn("network.filtered.net.xml", command)
        self.assertNotIn("--node-files", command)
        self.assertNotIn("--edge-files", command)


if __name__ == "__main__":
    unittest.main()
