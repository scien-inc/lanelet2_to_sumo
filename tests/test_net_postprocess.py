from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from ll2sumo.net_postprocess import (
    _align_internal_connection_shapes_to_net_lanes,
    _patch_net_japanese_tls_phases,
    _patch_net_lane_lengths_to_shape,
    _repair_degenerate_internal_lane_shapes,
    _summarize_joined_unmapped_connections,
    _summarize_tls_phase_sync,
    _sync_internal_lane_shapes_from_connection_shapes,
    _summarize_net_connectivity_and_write_safe_weights,
    _write_joined_unmapped_connection_deletions,
)


class NetPostprocessTest(unittest.TestCase):
    def _shape_xy(self, net_path: Path, lane_id: str) -> list[tuple[float, float]]:
        lane = ET.parse(net_path).getroot().find(f".//lane[@id='{lane_id}']")
        assert lane is not None
        return [
            (float(parts[0]), float(parts[1]))
            for parts in (token.split(",") for token in lane.attrib["shape"].split())
        ]

    def test_lane_length_is_patched_to_shape_length(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="edge_a">
    <lane id="edge_a_0" index="0" speed="13.9" length="999.0" shape="0,0,0 3,4,0"/>
  </edge>
</net>
""",
                encoding="utf-8",
            )

            summary = _patch_net_lane_lengths_to_shape(net_path)

            self.assertEqual(summary["patched_lane_count"], 1)
            self.assertEqual(summary["patched_normal_lane_count"], 1)
            self.assertEqual(summary["patched_internal_lane_count"], 0)
            self.assertAlmostEqual(summary["max_abs_diff_before_m"], 994.0)
            lane = ET.parse(net_path).getroot().find(".//lane")
            assert lane is not None
            self.assertEqual(lane.attrib["length"], "5.000")

    def test_internal_lane_length_is_patched_to_shape_length(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id=":node_0_1" function="internal">
    <lane id=":node_0_1_0" index="0" speed="13.9" length="10.0" shape="0,0,0 0,3,0 4,3,0"/>
  </edge>
</net>
""",
                encoding="utf-8",
            )

            summary = _patch_net_lane_lengths_to_shape(net_path)

            self.assertEqual(summary["patched_lane_count"], 1)
            self.assertEqual(summary["patched_normal_lane_count"], 0)
            self.assertEqual(summary["patched_internal_lane_count"], 1)
            lane = ET.parse(net_path).getroot().find(".//lane")
            assert lane is not None
            self.assertEqual(lane.attrib["length"], "7.000")

    def test_degenerate_internal_lane_length_keeps_sumo_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id=":node_0_1" function="internal">
    <lane id=":node_0_1_0" index="0" speed="13.9" length="10.0" shape="0,0,0 0,0,0"/>
  </edge>
</net>
""",
                encoding="utf-8",
            )

            summary = _patch_net_lane_lengths_to_shape(net_path)

            self.assertEqual(summary["patched_lane_count"], 1)
            self.assertEqual(summary["patched_internal_lane_count"], 1)
            lane = ET.parse(net_path).getroot().find(".//lane")
            assert lane is not None
            self.assertEqual(lane.attrib["length"], "0.100")

    def test_degenerate_internal_lane_shape_is_repaired_from_neighbor_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="from_edge"><lane id="from_edge_0" index="0" speed="13.9" length="1.0" shape="0,0,0 1,0,0"/></edge>
  <edge id="to_edge"><lane id="to_edge_0" index="0" speed="13.9" length="1.0" shape="2,0,0 3,0,0"/></edge>
  <edge id=":node_0" function="internal">
    <lane id=":node_0_0" index="0" speed="13.9" length="0.1" shape="1,0,0 1,0,0"/>
  </edge>
  <connection from="from_edge" to="to_edge" fromLane="0" toLane="0" via=":node_0_0"/>
</net>
""",
                encoding="utf-8",
            )

            summary = _repair_degenerate_internal_lane_shapes(net_path)

            self.assertEqual(summary["degenerate_internal_lane_count"], 1)
            self.assertEqual(summary["repaired_internal_lane_count"], 1)
            self.assertEqual(summary["unrepaired_internal_lane_count"], 0)
            self.assertEqual(self._shape_xy(net_path, ":node_0_0"), [(1.0, 0.0), (2.0, 0.0)])

    def test_degenerate_internal_lane_shape_extends_along_downstream_when_endpoints_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="from_edge"><lane id="from_edge_0" index="0" speed="13.9" length="1.0" shape="0,0,0 1,0,0"/></edge>
  <edge id="to_edge"><lane id="to_edge_0" index="0" speed="13.9" length="1.0" shape="1,0,0 2,0,0"/></edge>
  <edge id=":node_0" function="internal">
    <lane id=":node_0_0" index="0" speed="13.9" length="0.1" shape="1,0,0 1,0,0"/>
  </edge>
  <connection from="from_edge" to="to_edge" fromLane="0" toLane="0" via=":node_0_0"/>
</net>
""",
                encoding="utf-8",
            )

            summary = _repair_degenerate_internal_lane_shapes(net_path)

            self.assertEqual(summary["repaired_internal_lane_count"], 1)
            self.assertEqual(self._shape_xy(net_path, ":node_0_0"), [(1.0, 0.0), (1.25, 0.0)])

    def test_degenerate_internal_lane_without_via_connection_is_reported_unrepaired(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id=":node_0" function="internal">
    <lane id=":node_0_0" index="0" speed="13.9" length="0.1" shape="1,0,0 1,0,0"/>
  </edge>
</net>
""",
                encoding="utf-8",
            )

            summary = _repair_degenerate_internal_lane_shapes(net_path)

            self.assertEqual(summary["degenerate_internal_lane_count"], 1)
            self.assertEqual(summary["repaired_internal_lane_count"], 0)
            self.assertEqual(summary["unrepaired_internal_lane_count"], 1)
            self.assertEqual(summary["examples"][0]["reason"], "missing_via_connection")

    def test_degenerate_internal_lane_shape_syncs_from_connection_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="from_edge"><lane id="from_edge_0" index="0" speed="13.9" length="1.0" shape="0,0,0 1,0,0"/></edge>
  <edge id="to_edge"><lane id="to_edge_0" index="0" speed="13.9" length="1.0" shape="1,0,0 2,0,0"/></edge>
  <edge id=":node_0" function="internal">
    <lane id=":node_0_0" index="0" speed="13.9" length="0.1" shape="1,0,0 1,0,0"/>
  </edge>
  <connection from="from_edge" to="to_edge" fromLane="0" toLane="0" via=":node_0_0" shape="1,0,0 1.5,0,0"/>
</net>
""",
                encoding="utf-8",
            )

            summary = _sync_internal_lane_shapes_from_connection_shapes(net_path)

            self.assertEqual(summary["synced_internal_lane_count"], 1)
            self.assertEqual(self._shape_xy(net_path, ":node_0_0"), [(1.0, 0.0), (1.5, 0.0)])

    def test_internal_connection_shape_aligns_to_net_lane_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="from_edge"><lane id="from_edge_0" index="0" speed="13.9" length="2.0" shape="0,0,0 2,0,0"/></edge>
  <edge id="to_edge"><lane id="to_edge_0" index="0" speed="13.9" length="1.0" shape="4,0,0 5,0,0"/></edge>
  <edge id=":node_0" function="internal">
    <lane id=":node_0_0" index="0" speed="13.9" length="3.0" shape="1,0,0 2,0,0 3,0,0 4,0,0"/>
  </edge>
  <connection from="from_edge" to="to_edge" fromLane="0" toLane="0" via=":node_0_0" shape="1,0,0 2,0,0 3,0,0 4,0,0" length="3.0"/>
</net>
""",
                encoding="utf-8",
            )

            summary = _align_internal_connection_shapes_to_net_lanes(net_path)

            self.assertEqual(summary["aligned_connection_count"], 1)
            self.assertEqual(summary["trimmed_connection_shape_count"], 1)
            self.assertEqual(self._shape_xy(net_path, ":node_0_0"), [(2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
            connection = ET.parse(net_path).getroot().find("connection")
            assert connection is not None
            self.assertEqual(connection.attrib["shape"], "2.000,0.000,0.000 3.000,0.000,0.000 4.000,0.000,0.000")
            self.assertEqual(connection.attrib["length"], "2.000")

    def test_joined_intersection_alignment_preserves_connection_shape_middle_points(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            connections_path = Path(temp_dir) / "network.con.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="from_edge"><lane id="from_edge_0" index="0" speed="13.9" length="1.0" shape="-1,0,0 0,0,0"/></edge>
  <edge id="to_edge"><lane id="to_edge_0" index="0" speed="13.9" length="1.0" shape="10,0,0 11,0,0"/></edge>
  <edge id=":ia_100_0" function="internal">
    <lane id=":ia_100_0_0" index="0" speed="13.9" length="1.0" shape="4,0,0 10,0,0"/>
  </edge>
  <connection from="from_edge" to="to_edge" fromLane="0" toLane="0" via=":ia_100_0_0" shape="4,0,0 10,0,0" length="6.0"/>
</net>
""",
                encoding="utf-8",
            )
            connections_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<connections>
  <connection from="from_edge" to="to_edge" fromLane="0" toLane="0" shape="4,0,0 5,5,0 10,0,0" length="8.1"/>
</connections>
""",
                encoding="utf-8",
            )

            summary = _align_internal_connection_shapes_to_net_lanes(
                net_path,
                plain_connections_path=connections_path,
            )

            self.assertEqual(summary["preserved_joined_internal_lane_count"], 1)
            self.assertEqual(summary["fallback_joined_internal_lane_count"], 0)
            self.assertEqual(summary["plain_joined_connection_shape_count"], 1)
            self.assertEqual(summary["max_joined_internal_endpoint_gap_after_m"], 0.0)
            self.assertEqual(self._shape_xy(net_path, ":ia_100_0_0"), [(0.0, 0.0), (4.0, 0.0), (5.0, 5.0), (10.0, 0.0)])
            connection = ET.parse(net_path).getroot().find("connection")
            assert connection is not None
            self.assertEqual(connection.attrib["shape"], "0.000,0.000,0.000 4.000,0.000,0.000 5.000,5.000,0.000 10.000,0.000,0.000")

    def test_joined_unmapped_connection_deletions_only_target_extra_joined_external_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            connections_path = Path(temp_dir) / "network.con.xml"
            delete_path = Path(temp_dir) / "network.joined-delete.con.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <connection from="edge_a" to="edge_b" fromLane="0" toLane="0" via=":ia_100_0_0"/>
  <connection from="edge_a" to="edge_b" fromLane="0" toLane="1" via=":ia_100_0_1"/>
  <connection from="edge_a" to="edge_c" fromLane="0" toLane="0" via=":node_1_0_0"/>
  <connection from=":ia_100_0" to="edge_b" fromLane="0" toLane="0"/>
</net>
""",
                encoding="utf-8",
            )
            connections_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<connections>
  <connection from="edge_a" to="edge_b" fromLane="0" toLane="0"/>
</connections>
""",
                encoding="utf-8",
            )

            summary = _write_joined_unmapped_connection_deletions(net_path, connections_path, delete_path)

            self.assertEqual(summary["joined_unmapped_connection_count_before"], 1)
            self.assertEqual(summary["deleted_joined_unmapped_connection_count"], 1)
            self.assertTrue(delete_path.exists())
            deletes = ET.parse(delete_path).getroot().findall("delete")
            self.assertEqual(len(deletes), 1)
            self.assertEqual(
                deletes[0].attrib,
                {"from": "edge_a", "to": "edge_b", "fromLane": "0", "toLane": "1"},
            )
            post_summary = _summarize_joined_unmapped_connections(net_path, connections_path)
            self.assertEqual(post_summary["joined_unmapped_connection_count"], 1)

    def test_internal_connection_shape_keeps_minimum_length_for_close_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="from_edge"><lane id="from_edge_0" index="0" speed="13.9" length="1.0" shape="0,0,0 1,0,0"/></edge>
  <edge id="to_edge"><lane id="to_edge_0" index="0" speed="13.9" length="1.0" shape="1.05,0,0 2,0,0"/></edge>
  <edge id=":node_0" function="internal">
    <lane id=":node_0_0" index="0" speed="13.9" length="0.05" shape="1,0,0 1.05,0,0"/>
  </edge>
  <connection from="from_edge" to="to_edge" fromLane="0" toLane="0" via=":node_0_0" shape="1,0,0 1.05,0,0" length="0.05"/>
</net>
""",
                encoding="utf-8",
            )

            summary = _align_internal_connection_shapes_to_net_lanes(net_path)

            self.assertEqual(summary["aligned_connection_count"], 1)
            self.assertEqual(summary["tangent_fallback_shape_count"], 1)
            self.assertEqual(self._shape_xy(net_path, ":node_0_0"), [(1.0, 0.0), (1.25, 0.0)])

    def test_connectivity_summary_writes_safe_randomtrips_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            net_path = out_dir / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="source"><lane id="source_0" index="0" speed="13.9" length="1.0" shape="0,0,0 1,0,0"/></edge>
  <edge id="middle"><lane id="middle_0" index="0" speed="13.9" length="2.0" shape="1,0,0 3,0,0"/></edge>
  <edge id="sink"><lane id="sink_0" index="0" speed="13.9" length="1.0" shape="3,0,0 4,0,0"/></edge>
  <connection from="source" to="middle" fromLane="0" toLane="0"/>
  <connection from="middle" to="sink" fromLane="0" toLane="0"/>
</net>
""",
                encoding="utf-8",
            )

            summary = _summarize_net_connectivity_and_write_safe_weights(net_path, out_dir)

            self.assertEqual(summary["safe_randomtrips_edge_count"], 1)
            self.assertEqual(summary["no_incoming_edge_ids"], ["source"])
            self.assertEqual(summary["no_outgoing_edge_ids"], ["sink"])
            weights_root = ET.parse(out_dir / "randomtrips.safe.src.xml").getroot()
            weights = {
                edge.attrib["id"]: float(edge.attrib["value"])
                for edge in weights_root.findall(".//edge")
            }
            self.assertEqual(weights["source"], 0.0)
            self.assertGreater(weights["middle"], 0.0)
            self.assertEqual(weights["sink"], 0.0)

    def test_japanese_tls_phase_patch_splits_crossing_axes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="edge_w"><lane id="edge_w_0" index="0" shape="-10,0,0 0,0,0"/></edge>
  <edge id="edge_e"><lane id="edge_e_0" index="0" shape="10,0,0 0,0,0"/></edge>
  <edge id="edge_s"><lane id="edge_s_0" index="0" shape="0,-10,0 0,0,0"/></edge>
  <edge id="edge_n"><lane id="edge_n_0" index="0" shape="0,10,0 0,0,0"/></edge>
  <edge id="out_w"><lane id="out_w_0" index="0" shape="0,0,0 -10,0,0"/></edge>
  <edge id="out_e">
    <lane id="out_e_0" index="0" shape="0,0,0 10,0,0"/>
    <lane id="out_e_1" index="1" shape="0,1,0 10,1,0"/>
  </edge>
  <edge id="out_s"><lane id="out_s_0" index="0" shape="0,0,0 0,-10,0"/></edge>
  <edge id="out_n"><lane id="out_n_0" index="0" shape="0,0,0 0,10,0"/></edge>
  <tlLogic id="tls_jp" type="static" programID="0" offset="0">
    <phase duration="90" state="GGGG"/>
  </tlLogic>
  <connection from="edge_w" to="out_e" fromLane="0" toLane="0" tl="tls_jp" linkIndex="0" dir="s"/>
  <connection from="edge_e" to="out_w" fromLane="0" toLane="0" tl="tls_jp" linkIndex="1" dir="s"/>
  <connection from="edge_s" to="out_n" fromLane="0" toLane="0" tl="tls_jp" linkIndex="2" dir="s"/>
  <connection from="edge_n" to="out_s" fromLane="0" toLane="0" tl="tls_jp" linkIndex="3" dir="s"/>
</net>
""",
                encoding="utf-8",
            )

            summary = _patch_net_japanese_tls_phases(net_path)

            self.assertEqual(summary["patched_tls_count"], 1)
            phases = ET.parse(net_path).getroot().findall(".//tlLogic[@id='tls_jp']/phase")
            states = [phase.attrib["state"] for phase in phases]
            self.assertIn("GGrr", states)
            self.assertIn("rrGG", states)
            self.assertNotIn("GGGG", states)

    def test_japanese_tls_phase_patch_syncs_joined_intersection_area_tls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="edge_w"><lane id="edge_w_0" index="0" shape="-10,0,0 0,0,0"/></edge>
  <edge id="edge_e"><lane id="edge_e_0" index="0" shape="10,0,0 0,0,0"/></edge>
  <edge id="edge_s"><lane id="edge_s_0" index="0" shape="0,-10,0 0,0,0"/></edge>
  <edge id="edge_n"><lane id="edge_n_0" index="0" shape="0,10,0 0,0,0"/></edge>
  <edge id="out_w"><lane id="out_w_0" index="0" shape="0,0,0 -10,0,0"/></edge>
  <edge id="out_e"><lane id="out_e_0" index="0" shape="0,0,0 10,0,0"/></edge>
  <edge id="out_s"><lane id="out_s_0" index="0" shape="0,0,0 0,-10,0"/></edge>
  <edge id="out_n"><lane id="out_n_0" index="0" shape="0,0,0 0,10,0"/></edge>
  <tlLogic id="tls_joined" type="static" programID="0" offset="0">
    <phase duration="90" state="GGGG"/>
  </tlLogic>
  <tlLogic id="tls_regular" type="static" programID="0" offset="0">
    <phase duration="90" state="GGGG"/>
  </tlLogic>
  <connection from="edge_w" to="out_e" fromLane="0" toLane="0" via=":ia_100_0_0" tl="tls_joined" linkIndex="0" dir="s"/>
  <connection from="edge_e" to="out_w" fromLane="0" toLane="0" via=":ia_100_1_0" tl="tls_joined" linkIndex="1" dir="s"/>
  <connection from="edge_s" to="out_n" fromLane="0" toLane="0" via=":ia_100_2_0" tl="tls_joined" linkIndex="2" dir="s"/>
  <connection from="edge_n" to="out_s" fromLane="0" toLane="0" via=":ia_100_3_0" tl="tls_joined" linkIndex="3" dir="s"/>
  <connection from="edge_w" to="out_e" fromLane="0" toLane="0" via=":node_1_0_0" tl="tls_regular" linkIndex="0" dir="s"/>
  <connection from="edge_e" to="out_w" fromLane="0" toLane="0" via=":node_1_1_0" tl="tls_regular" linkIndex="1" dir="s"/>
  <connection from="edge_s" to="out_n" fromLane="0" toLane="0" via=":node_1_2_0" tl="tls_regular" linkIndex="2" dir="s"/>
  <connection from="edge_n" to="out_s" fromLane="0" toLane="0" via=":node_1_3_0" tl="tls_regular" linkIndex="3" dir="s"/>
</net>
""",
                encoding="utf-8",
            )

            summary = _patch_net_japanese_tls_phases(net_path)

            root = ET.parse(net_path).getroot()
            joined_states = [
                phase.attrib["state"]
                for phase in root.findall(".//tlLogic[@id='tls_joined']/phase")
            ]
            regular_states = [
                phase.attrib["state"]
                for phase in root.findall(".//tlLogic[@id='tls_regular']/phase")
            ]
            self.assertEqual(summary["patched_tls_ids"], ["tls_joined", "tls_regular"])
            self.assertIn("GGrr", joined_states)
            self.assertIn("rrGG", joined_states)
            self.assertIn("GGrr", regular_states)
            self.assertIn("rrGG", regular_states)

    def test_japanese_tls_phase_patch_adds_permissive_and_protected_right_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="edge_w"><lane id="edge_w_0" index="0" shape="-10,0,0 0,0,0"/></edge>
  <edge id="edge_e"><lane id="edge_e_0" index="0" shape="10,0,0 0,0,0"/></edge>
  <edge id="edge_s"><lane id="edge_s_0" index="0" shape="0,-10,0 0,0,0"/></edge>
  <edge id="edge_n"><lane id="edge_n_0" index="0" shape="0,10,0 0,0,0"/></edge>
  <edge id="out_w"><lane id="out_w_0" index="0" shape="0,0,0 -10,0,0"/></edge>
  <edge id="out_e"><lane id="out_e_0" index="0" shape="0,0,0 10,0,0"/></edge>
  <edge id="out_s"><lane id="out_s_0" index="0" shape="0,0,0 0,-10,0"/></edge>
  <edge id="out_n"><lane id="out_n_0" index="0" shape="0,0,0 0,10,0"/></edge>
  <tlLogic id="tls_jp" type="static" programID="0" offset="0">
    <phase duration="90" state="GGGGG"/>
  </tlLogic>
  <connection from="edge_w" to="out_e" fromLane="0" toLane="0" tl="tls_jp" linkIndex="0" dir="s"/>
  <connection from="edge_e" to="out_w" fromLane="0" toLane="0" tl="tls_jp" linkIndex="1" dir="s"/>
  <connection from="edge_s" to="out_n" fromLane="0" toLane="0" tl="tls_jp" linkIndex="2" dir="s"/>
  <connection from="edge_n" to="out_s" fromLane="0" toLane="0" tl="tls_jp" linkIndex="3" dir="s"/>
  <connection from="edge_w" to="out_s" fromLane="0" toLane="0" tl="tls_jp" linkIndex="4" dir="r"/>
</net>
""",
                encoding="utf-8",
            )

            _patch_net_japanese_tls_phases(net_path)

            states = [
                phase.attrib["state"]
                for phase in ET.parse(net_path).getroot().findall(".//tlLogic[@id='tls_jp']/phase")
            ]
            self.assertIn("GGrrg", states)
            self.assertNotIn("rrrrG", states)

    def test_japanese_tls_phase_patch_keeps_same_incoming_lane_links_together(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="edge_w"><lane id="edge_w_0" index="0" shape="-10,0,0 0,0,0"/></edge>
  <edge id="edge_e"><lane id="edge_e_0" index="0" shape="10,0,0 0,0,0"/></edge>
  <edge id="edge_s"><lane id="edge_s_0" index="0" shape="0,-10,0 0,0,0"/></edge>
  <edge id="out_e"><lane id="out_e_0" index="0" shape="0,0,0 10,0,0"/></edge>
  <edge id="out_s"><lane id="out_s_0" index="0" shape="0,0,0 0,-10,0"/></edge>
  <edge id="out_w"><lane id="out_w_0" index="0" shape="0,0,0 -10,0,0"/></edge>
  <tlLogic id="tls_jp" type="static" programID="0" offset="0">
    <phase duration="90" state="Grr"/>
    <phase duration="90" state="rGr"/>
  </tlLogic>
  <connection from="edge_w" to="out_e" fromLane="0" toLane="0" tl="tls_jp" linkIndex="0" dir="s"/>
  <connection from="edge_w" to="out_s" fromLane="0" toLane="0" tl="tls_jp" linkIndex="1" dir="r"/>
  <connection from="edge_s" to="out_w" fromLane="0" toLane="0" tl="tls_jp" linkIndex="2" dir="s"/>
</net>
""",
                encoding="utf-8",
            )

            before = _summarize_tls_phase_sync(net_path)
            summary = _patch_net_japanese_tls_phases(net_path)
            after = _summarize_tls_phase_sync(net_path)

            self.assertEqual(before["mixed_same_incoming_lane_phase_count"], 2)
            self.assertEqual(summary["mixed_same_incoming_lane_phase_count_after"], 0)
            self.assertEqual(after["mixed_same_incoming_lane_phase_count"], 0)
            states = [
                phase.attrib["state"]
                for phase in ET.parse(net_path).getroot().findall(".//tlLogic[@id='tls_jp']/phase")
            ]
            self.assertIn("Ggr", states)
            self.assertNotIn("Grr", states)
            self.assertNotIn("rGr", states)

    def test_japanese_tls_phase_patch_syncs_same_heading_approaches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="edge_w0"><lane id="edge_w0_0" index="0" shape="-10,0,0 0,0,0"/></edge>
  <edge id="edge_w1"><lane id="edge_w1_0" index="0" shape="-10,1,0 0,1,0"/></edge>
  <edge id="edge_e"><lane id="edge_e_0" index="0" shape="10,0,0 0,0,0"/></edge>
  <edge id="edge_s"><lane id="edge_s_0" index="0" shape="0,-10,0 0,0,0"/></edge>
  <edge id="out_e"><lane id="out_e_0" index="0" shape="0,0,0 10,0,0"/></edge>
  <edge id="out_w"><lane id="out_w_0" index="0" shape="0,0,0 -10,0,0"/></edge>
  <edge id="out_n"><lane id="out_n_0" index="0" shape="0,0,0 0,10,0"/></edge>
  <tlLogic id="tls_jp" type="static" programID="0" offset="0">
    <phase duration="90" state="Grrr"/>
  </tlLogic>
  <connection from="edge_w0" to="out_e" fromLane="0" toLane="0" tl="tls_jp" linkIndex="0" dir="s"/>
  <connection from="edge_w1" to="out_e" fromLane="0" toLane="1" tl="tls_jp" linkIndex="1" dir="s"/>
  <connection from="edge_e" to="out_w" fromLane="0" toLane="0" tl="tls_jp" linkIndex="2" dir="s"/>
  <connection from="edge_s" to="out_n" fromLane="0" toLane="0" tl="tls_jp" linkIndex="3" dir="s"/>
</net>
""",
                encoding="utf-8",
            )

            summary = _patch_net_japanese_tls_phases(net_path)

            self.assertEqual(summary["mixed_same_approach_phase_count_after"], 0)
            states = [
                phase.attrib["state"]
                for phase in ET.parse(net_path).getroot().findall(".//tlLogic[@id='tls_jp']/phase")
            ]
            self.assertIn("GGGr", states)
            self.assertIn("rrrG", states)

    def test_japanese_tls_phase_patch_marks_shared_target_lane_links_permissive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="edge_w0"><lane id="edge_w0_0" index="0" shape="-10,0,0 0,0,0"/></edge>
  <edge id="edge_w1"><lane id="edge_w1_0" index="0" shape="-10,1,0 0,1,0"/></edge>
  <edge id="edge_s"><lane id="edge_s_0" index="0" shape="0,-10,0 0,0,0"/></edge>
  <edge id="out_e"><lane id="out_e_0" index="0" shape="0,0,0 10,0,0"/></edge>
  <edge id="out_n"><lane id="out_n_0" index="0" shape="0,0,0 0,10,0"/></edge>
  <tlLogic id="tls_jp" type="static" programID="0" offset="0">
    <phase duration="90" state="GGG"/>
  </tlLogic>
  <connection from="edge_w0" to="out_e" fromLane="0" toLane="0" tl="tls_jp" linkIndex="0" dir="s"/>
  <connection from="edge_w1" to="out_e" fromLane="0" toLane="0" tl="tls_jp" linkIndex="1" dir="s"/>
  <connection from="edge_s" to="out_n" fromLane="0" toLane="0" tl="tls_jp" linkIndex="2" dir="s"/>
</net>
""",
                encoding="utf-8",
            )

            _patch_net_japanese_tls_phases(net_path)

            states = [
                phase.attrib["state"]
                for phase in ET.parse(net_path).getroot().findall(".//tlLogic[@id='tls_jp']/phase")
            ]
            self.assertIn("ggr", states)
            self.assertNotIn("GGr", states)


if __name__ == "__main__":
    unittest.main()
