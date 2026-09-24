from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from ll2sumo.net_postprocess import (
    _align_internal_connection_shapes_to_net_lanes,
    _patch_net_japanese_tls_phases,
    _patch_net_lane_lengths_to_shape,
    _audit_degenerate_internal_lane_shapes,
    _summarize_joined_unmapped_connections,
    _summarize_tls_phase_sync,
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
  <connection from=":node_0" to="to_edge" fromLane="0" toLane="0"/>
</net>
""",
                encoding="utf-8",
            )

            summary = _align_internal_connection_shapes_to_net_lanes(net_path)

            self.assertEqual(summary["repaired_internal_lane_count"], 1)
            self.assertEqual(summary["unrepaired_connection_count"], 0)
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
  <connection from=":node_0" to="to_edge" fromLane="0" toLane="0"/>
</net>
""",
                encoding="utf-8",
            )

            summary = _align_internal_connection_shapes_to_net_lanes(net_path)

            self.assertEqual(summary["repaired_internal_lane_count"], 1)
            self.assertEqual(self._shape_xy(net_path, ":node_0_0"), [(1.0, 0.0), (1.25, 0.0)])

    def test_orphan_degenerate_lane_is_left_for_the_final_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "net.xml"
            path.write_text('<net><edge id=":orphan" function="internal"><lane id=":orphan_0" shape="0,0 0,0"/></edge></net>')
            _align_internal_connection_shapes_to_net_lanes(path)
            self.assertEqual(_audit_degenerate_internal_lane_shapes(path)["degenerate_internal_lane_count"], 1)

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
  <connection from=":node_0" to="to_edge" fromLane="0" toLane="0"/>
</net>
""",
                encoding="utf-8",
            )

            summary = _align_internal_connection_shapes_to_net_lanes(net_path)

            self.assertEqual(summary["repaired_internal_lane_count"], 1)
            self.assertEqual(self._shape_xy(net_path, ":node_0_0"), [(1.0, 0.0), (1.25, 0.0)])

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
  <connection from=":node_0" to="to_edge" fromLane="0" toLane="0"/>
</net>
""",
                encoding="utf-8",
            )

            summary = _align_internal_connection_shapes_to_net_lanes(net_path)

            self.assertEqual(summary["aligned_connection_count"], 1)
            self.assertEqual(self._shape_xy(net_path, ":node_0_0"), [(2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
            connection = ET.parse(net_path).getroot().find("connection")
            assert connection is not None
            self.assertEqual(connection.attrib["shape"], "2.000,0.000,0.000 3.000,0.000,0.000 4.000,0.000,0.000")
            self.assertEqual(connection.attrib["length"], "2.000")

    def test_source_curve_restores_normal_endpoints_and_split_vias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "net.xml"
            edges = Path(directory) / "edges.xml"
            connections = Path(directory) / "connections.xml"
            path.write_text('<net>\n<edge id="a"><lane id="a_0" index="0" shape="0,0 2,0"/></edge>\n<edge id="b"><lane id="b_0" index="0" shape="4,0 6,0"/></edge>\n<edge id=":x"><lane id=":x_0" index="0" shape="2,0 3,0.2"/></edge>\n<edge id=":y"><lane id=":y_0" index="0" shape="3,0.2 4,0"/></edge>\n<connection from="a" fromLane="0" to="b" toLane="0" via=":x_0"/>\n<connection from=":x" fromLane="0" to="b" toLane="0" via=":y_0"/>\n<connection from=":y" fromLane="0" to="b" toLane="0"/>\n</net>')
            edges.write_text('<edges><edge id="a"><lane index="0" shape="0,0 1,0"/></edge><edge id="b"><lane index="0" shape="5,0 6,0"/></edge></edges>')
            connections.write_text('<connections><connection from="a" to="b" fromLane="0" toLane="0" shape="1,0 2,0.1 3,0.2 4,0.1 5,0"/></connections>')
            summary = _align_internal_connection_shapes_to_net_lanes(path, connections, edges)
            self.assertEqual(summary["split_connection_count"], 1)
            self.assertEqual(self._shape_xy(path, "a_0"), [(0, 0), (1, 0)])
            self.assertEqual(self._shape_xy(path, "b_0"), [(5, 0), (6, 0)])
            self.assertEqual(self._shape_xy(path, ":x_0"), [(1, 0), (2, 0.1), (3, 0.2)])
            self.assertEqual(self._shape_xy(path, ":y_0"), [(3, 0.2), (4, 0.1), (5, 0)])
            self.assertNotIn("shape", ET.parse(path).getroot().findall("connection")[-1].attrib)

    def _write_split_reflection_fixture(self, directory, source_sign=1, via_sign=-1, error=(0, 0)):
        path, connections = Path(directory) / "net.xml", Path(directory) / "connections.xml"
        root = ET.Element("net")
        shapes = {
            "a_0": f"0,{9 * source_sign},4 1,{10 * source_sign},4",
            "a_1": f"0,{9 * source_sign},4 1,{10 * source_sign},4",
            "b_0": f"5,{14 * source_sign},4 6,{15 * source_sign},4",
            ":x_0": f"1,{10 * via_sign},4 3,{12 * via_sign + error[0]},{4 + error[1]}",
            ":y_0": f"3,{12 * via_sign + error[0]},{4 + error[1]} 5,{14 * via_sign},4",
            ":z_0": f"1,{10 * via_sign},4 5,{14 * via_sign},4",
        }
        for lane_id, shape in shapes.items():
            edge_id, index = lane_id.rsplit("_", 1)
            edge = root.find(f"edge[@id='{edge_id}']")
            if edge is None:
                edge = ET.SubElement(root, "edge", id=edge_id)
            ET.SubElement(edge, "lane", id=lane_id, index=index, shape=shape)
        plain = ET.Element("connections")
        for lane, via in (("0", ":x_0"), ("1", ":z_0")):
            attrs = dict(fromLane=lane, toLane="0", to="b", **{"from": "a"})
            shape = f"1,{10 * source_sign},4 3,{12 * source_sign},4 5,{14 * source_sign},4"
            net_shape = f"1,{10 * via_sign},4 3,{12 * via_sign},4 5,{14 * via_sign},4"
            ET.SubElement(root, "connection", **attrs, via=via, shape=net_shape)
            ET.SubElement(plain, "connection", **attrs, shape=shape)
        for edge, via in ((":x", ":y_0"), (":y", None), (":z", None)):
            link = ET.SubElement(root, "connection", **{"from": edge}, fromLane="0", to="b", toLane="0")
            if via:
                link.set("via", via)
        ET.ElementTree(root).write(path)
        ET.ElementTree(plain).write(connections)
        return path, connections

    def test_source_verified_y_reflection_restores_split_and_shared_successor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for source_sign, via_sign, corrected in ((1, -1, 3), (-1, 1, 3), (-1, -1, 0)):
                with self.subTest(source_sign=source_sign, via_sign=via_sign):
                    path, connections = self._write_split_reflection_fixture(directory, source_sign, via_sign)
                    summary = _align_internal_connection_shapes_to_net_lanes(path, connections)
                    self.assertEqual(summary["aligned_connection_count"], 2)
                    self.assertEqual(summary["unrepaired_connection_count"], 0)
                    self.assertEqual(summary["corrected_y_reflection_lane_count"], corrected)
                    self.assertEqual(self._shape_xy(path, ":x_0"), [(1, 10 * source_sign), (3, 12 * source_sign)])
                    self.assertEqual(self._shape_xy(path, ":y_0"), [(3, 12 * source_sign), (5, 14 * source_sign)])
                    self.assertEqual(self._shape_xy(path, ":z_0"), [(1, 10 * source_sign), (3, 12 * source_sign), (5, 14 * source_sign)])
                    self.assertEqual(_audit_degenerate_internal_lane_shapes(path)["discontinuities"], [])
                    once = path.read_bytes()
                    summary = _align_internal_connection_shapes_to_net_lanes(path, connections)
                    self.assertEqual(summary["corrected_y_reflection_lane_count"], 0)
                    self.assertEqual(path.read_bytes(), once)

    def test_reflection_requires_matching_source_curve_and_height(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for error, provide_source, reverse in (
                ((0.02, 0), True, False), ((0, 0.02), True, False),
                ((0, 0), False, False), ((0, 0), True, True),
            ):
                with self.subTest(error=error, provide_source=provide_source, reverse=reverse):
                    path, connections = self._write_split_reflection_fixture(directory, error=error)
                    tree = ET.parse(path)
                    tree.getroot().find(".//lane[@id=':z_0']").set("shape", "1,10,4 5,14,4")
                    if reverse:
                        first = tree.getroot().find(".//lane[@id=':x_0']")
                        first.set("shape", " ".join(reversed(first.get("shape").split())))
                    tree.write(path)
                    before = [(e.tag, dict(e.attrib)) for e in tree.getroot().iter()]
                    summary = _align_internal_connection_shapes_to_net_lanes(path, connections if provide_source else None)
                    self.assertEqual(summary["corrected_y_reflection_lane_count"], 0)
                    self.assertEqual(summary["unrepaired_connection_count"], 2)
                    self.assertEqual([(e.tag, e.attrib) for e in ET.parse(path).getroot().iter()], before)

    def test_verified_reflection_survives_unrelated_successor_group_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, connections = self._write_split_reflection_fixture(directory, error=(0, 0.02))
            before = self._shape_xy(path, ":x_0")
            summary = _align_internal_connection_shapes_to_net_lanes(path, connections)
            self.assertEqual(summary["unrepaired_connection_count"], 2)
            self.assertEqual(summary["corrected_y_reflection_lane_count"], 1)
            self.assertEqual(self._shape_xy(path, ":x_0"), before)
            self.assertEqual(self._shape_xy(path, ":z_0"), [(1, 10), (5, 14)])
            owner = ET.parse(path).getroot().find("connection[@fromLane='1']")
            self.assertEqual(owner.get("shape"), "1.000,10.000,4.000 3.000,12.000,4.000 5.000,14.000,4.000")

    def test_shared_successor_stub_follows_slope_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "net.xml"
            path.write_text('<net>\n<edge id="a"><lane id="a_0" index="0" shape="0,0,0 1,0,0"/><lane id="a_1" index="1" shape="0,-0.1,0 1,0,0"/></edge>\n<edge id="b"><lane id="b_0" index="0" shape="1,0,0 2,0,0.1"/></edge>\n<edge id=":x"><lane id=":x_0" index="0" shape="1,0,0 1,0,0"/><lane id=":x_1" index="1" shape="1,0,0 0.9,0,0"/></edge>\n<connection from="a" fromLane="0" to="b" toLane="0" via=":x_0"/>\n<connection from="a" fromLane="1" to="b" toLane="0" via=":x_1"/>\n<connection from=":x" fromLane="0" to="b" toLane="0"/>\n<connection from=":x" fromLane="1" to="b" toLane="0"/>\n</net>')
            summary = _align_internal_connection_shapes_to_net_lanes(path)
            self.assertEqual(summary["allocated_successor_stub_count"], 1)
            self.assertEqual(summary["repaired_internal_lane_count"], 2)
            self.assertEqual(self._shape_xy(path, "b_0")[0], (1.25, 0.0))
            for lane_id in (":x_0", ":x_1"):
                self.assertEqual(self._shape_xy(path, lane_id), [(1.0, 0.0), (1.25, 0.0)])
            lane = ET.parse(path).getroot().find(".//lane[@id='b_0']")
            self.assertEqual(lane.get("shape"), "1.250,0.000,0.025 2.000,0.000,0.100")
            before = path.read_bytes()
            _align_internal_connection_shapes_to_net_lanes(path)
            self.assertEqual(path.read_bytes(), before)

    def test_split_outside_source_curve_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "net.xml"
            path.write_text('<net><edge id="a"><lane id="a_0" index="0" shape="0,0 1,0"/></edge><edge id="b"><lane id="b_0" index="0" shape="3,0 4,0"/></edge><edge id=":x"><lane id=":x_0" index="0" shape="1,0 5,0"/></edge><edge id=":y"><lane id=":y_0" index="0" shape="5,0 3,0"/></edge><connection from="a" to="b" fromLane="0" toLane="0" via=":x_0"/><connection from=":x" to="b" fromLane="0" toLane="0" via=":y_0"/><connection from=":y" to="b" fromLane="0" toLane="0"/></net>')
            before = [(element.tag, element.attrib.copy()) for element in ET.parse(path).getroot().iter()]
            summary = _align_internal_connection_shapes_to_net_lanes(path)
            self.assertEqual(summary["reason_counts"], {"ambiguous_internal_split": 1})
            self.assertEqual([(element.tag, element.attrib) for element in ET.parse(path).getroot().iter()], before)

    def test_short_stub_does_not_restore_a_large_source_height_difference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, edges = Path(directory) / "net.xml", Path(directory) / "edges.xml"
            path.write_text('<net><edge id="a"><lane id="a_0" index="0" shape="0,0,0 1,0,0"/></edge><edge id="b"><lane id="b_0" index="0" shape="1,0,0.2 2,0,0.2"/></edge><edge id=":x"><lane id=":x_0" index="0" shape="1,0,0 1.2,0,0.2"/></edge><connection from="a" to="b" fromLane="0" toLane="0" via=":x_0"/><connection from=":x" to="b" fromLane="0" toLane="0"/></net>')
            edges.write_text('<edges><edge id="b"><lane index="0" shape="1,0,0 2,0,0"/></edge></edges>')
            before = [element.attrib.copy() for element in ET.parse(path).getroot().iter()]
            summary = _align_internal_connection_shapes_to_net_lanes(path, plain_edges_path=edges)
            self.assertEqual(summary["reason_counts"], {"stub_height_change_exceeds_limit": 1})
            self.assertEqual([element.attrib for element in ET.parse(path).getroot().iter()], before)

    def test_short_stub_checks_entry_height_and_each_segment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, edges = Path(directory) / "net.xml", Path(directory) / "edges.xml"
            for original_height, successor, reason in (
                (0.2, "1,0,0 2,0,0", "stub_height_change_exceeds_limit"),
                (0, "1,0,0 1.02,0,0 2,0,0", "stub_segment_too_short"),
            ):
                with self.subTest(reason=reason):
                    path.write_text(f'<net><edge id="a"><lane id="a_0" index="0" shape="0,0,{original_height} 1,0,{original_height}"/></edge><edge id="b"><lane id="b_0" index="0" shape="{successor}"/></edge><edge id=":x"><lane id=":x_0" index="0" shape="1,0,{original_height} 1.1,0,0"/></edge><connection from="a" to="b" fromLane="0" toLane="0" via=":x_0"/><connection from=":x" to="b" fromLane="0" toLane="0"/></net>')
                    edges.write_text('<edges><edge id="a"><lane index="0" shape="0,0,0 1,0,0"/></edge></edges>')
                    before = [element.attrib.copy() for element in ET.parse(path).getroot().iter()]
                    summary = _align_internal_connection_shapes_to_net_lanes(path, plain_edges_path=edges)
                    self.assertEqual(summary["reason_counts"], {reason: 1})
                    self.assertEqual([element.attrib for element in ET.parse(path).getroot().iter()], before)

    def test_ambiguous_shared_via_preserves_its_other_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "net.xml"
            path.write_text('<net><edge id="a"><lane id="a_0" index="0" shape="0,0 1,0"/></edge><edge id="b"><lane id="b_0" index="0" shape="3,0 4,0"/></edge><edge id="c"><lane id="c_0" index="0" shape="3,0 4,1"/></edge><edge id=":x"><lane id=":x_0" index="0" shape="1,0 2,0"/></edge><connection from="a" to="b" fromLane="0" toLane="0" via=":x_0"/><connection from="a" to="c" fromLane="0" toLane="0" via=":x_0"/><connection from=":x" to="b" fromLane="0" toLane="0"/></net>')
            before = [element.attrib.copy() for element in ET.parse(path).getroot().iter()]
            summary = _align_internal_connection_shapes_to_net_lanes(path)
            self.assertEqual(summary["unrepaired_connection_count"], 2)
            self.assertEqual(summary["aligned_connection_count"], 0)
            self.assertEqual([element.attrib for element in ET.parse(path).getroot().iter()], before)

    def test_ambiguous_successor_group_preserves_all_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "net.xml"
            for successor in ("1,0,0 1,1,0", "1,0,0 2,0,1"):
                with self.subTest(successor=successor):
                    path.write_text(f'<net>\n<edge id="a"><lane id="a_0" index="0" shape="0,0,0 1,0,0"/></edge>\n<edge id="b"><lane id="b_0" index="0" shape="{successor}"/></edge>\n<edge id=":x"><lane id=":x_0" index="0" shape="1,0,0 1.1,0,0"/></edge>\n<connection from="a" fromLane="0" to="b" toLane="0" via=":x_0"/>\n<connection from=":x" fromLane="0" to="b" toLane="0"/>\n</net>')
                    before = [element.attrib.copy() for element in ET.parse(path).getroot().iter()]
                    summary = _align_internal_connection_shapes_to_net_lanes(path)
                    self.assertEqual(summary["unrepaired_connection_count"], 1)
                    self.assertEqual([element.attrib for element in ET.parse(path).getroot().iter()], before)

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

    def test_close_endpoints_do_not_extend_past_successor(self) -> None:
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
  <connection from=":node_0" to="to_edge" fromLane="0" toLane="0"/>
</net>
""",
                encoding="utf-8",
            )

            summary = _align_internal_connection_shapes_to_net_lanes(net_path)

            self.assertEqual(summary["aligned_connection_count"], 1)
            self.assertEqual(self._shape_xy(net_path, ":node_0_0"), [(1.0, 0.0), (1.05, 0.0)])

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
