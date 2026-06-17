from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from ll2sumo import convert_map


REPO_ROOT = Path(__file__).resolve().parents[1]
MINIMAL_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "minimal_road_lanelet.osm"
MINIMAL_GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "minimal_signal_none_skip_netconvert"

MINIMAL_SIGNAL_OSM = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="ll2sumo-test">
  <node id="1"><tag k="local_x" v="0.0" /><tag k="local_y" v="1.0" /><tag k="local_z" v="0.0" /></node>
  <node id="2"><tag k="local_x" v="10.0" /><tag k="local_y" v="1.0" /><tag k="local_z" v="0.0" /></node>
  <node id="3"><tag k="local_x" v="0.0" /><tag k="local_y" v="-1.0" /><tag k="local_z" v="0.0" /></node>
  <node id="4"><tag k="local_x" v="10.0" /><tag k="local_y" v="-1.0" /><tag k="local_z" v="0.0" /></node>
  <node id="5"><tag k="local_x" v="20.0" /><tag k="local_y" v="1.0" /><tag k="local_z" v="0.0" /></node>
  <node id="6"><tag k="local_x" v="20.0" /><tag k="local_y" v="-1.0" /><tag k="local_z" v="0.0" /></node>
  <node id="7"><tag k="local_x" v="10.0" /><tag k="local_y" v="2.0" /><tag k="local_z" v="0.0" /></node>
  <node id="8"><tag k="local_x" v="10.0" /><tag k="local_y" v="-2.0" /><tag k="local_z" v="0.0" /></node>
  <way id="10"><nd ref="1" /><nd ref="2" /><tag k="type" v="line_thin" /><tag k="subtype" v="solid" /></way>
  <way id="20"><nd ref="3" /><nd ref="4" /><tag k="type" v="road_border" /></way>
  <way id="30"><nd ref="2" /><nd ref="5" /><tag k="type" v="line_thin" /><tag k="subtype" v="solid" /></way>
  <way id="40"><nd ref="4" /><nd ref="6" /><tag k="type" v="road_border" /></way>
  <way id="300"><nd ref="2" /><nd ref="5" /><nd ref="6" /><nd ref="4" /><nd ref="2" /><tag k="type" v="intersection_area" /></way>
  <way id="700"><nd ref="7" /><nd ref="8" /><tag k="subtype" v="red_yellow_green" /></way>
  <way id="800"><nd ref="7" /><nd ref="8" /><tag k="subtype" v="stop_line" /></way>
  <relation id="100">
    <member type="way" ref="10" role="left" />
    <member type="way" ref="20" role="right" />
    <member type="relation" ref="900" role="regulatory_element" />
    <tag k="type" v="lanelet" /><tag k="subtype" v="road" /><tag k="one_way" v="yes" /><tag k="speed_limit" v="50" />
  </relation>
  <relation id="200">
    <member type="way" ref="30" role="left" />
    <member type="way" ref="40" role="right" />
    <tag k="type" v="lanelet" /><tag k="subtype" v="road" /><tag k="one_way" v="yes" /><tag k="speed_limit" v="50" />
    <tag k="intersection_area" v="300" />
  </relation>
  <relation id="900">
    <member type="way" ref="700" role="refers" />
    <member type="way" ref="800" role="ref_line" />
    <tag k="type" v="regulatory_element" />
    <tag k="subtype" v="traffic_light" />
  </relation>
</osm>
"""

CONVERSION_REPORT_COUNT_KEYS = (
    "lanelet_count",
    "road_lanelet_count",
    "lane_group_count",
    "edge_count",
    "node_count",
    "connection_count",
    "merged_serial_group_count",
    "merged_parallel_group_count",
    "intersection_area_cluster_count",
    "collapsed_intersection_group_count",
    "dropped_self_loop_lanelet_count",
    "dropped_tiny_lanelet_count",
)


def _xml_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    return {
        "edge_count": len(root.findall(".//edge")),
        "lane_count": len(root.findall(".//lane")),
        "node_count": len(root.findall(".//node")),
        "junction_count": len(root.findall(".//junction")),
        "connection_count": len(root.findall(".//connection")),
        "tlLogic_count": len(root.findall(".//tlLogic")),
        "phase_count": len(root.findall(".//phase")),
    }


def _plain_primary_counts(out_dir: Path) -> dict[str, int | str]:
    node_counts = _xml_counts(out_dir / "network.nod.xml")
    edge_counts = _xml_counts(out_dir / "network.edg.xml")
    connection_counts = _xml_counts(out_dir / "network.con.xml")
    return {
        "source": "plain SUMO XML files",
        "edge_count": edge_counts["edge_count"],
        "lane_count": edge_counts["lane_count"],
        "node_count": node_counts["node_count"],
        "junction_count": 0,
        "connection_count": connection_counts["connection_count"],
        "tlLogic_count": 0,
        "phase_count": 0,
    }


def _conversion_report_counts(out_dir: Path) -> dict[str, int]:
    report = json.loads((out_dir / "conversion.report.json").read_text(encoding="utf-8"))
    return {
        key: report[key]
        for key in CONVERSION_REPORT_COUNT_KEYS
        if key in report
    }


class PublicConversionRegressionTest(unittest.TestCase):
    def test_convert_map_public_api_matches_minimal_plain_xml_golden(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)

            result = convert_map(
                MINIMAL_FIXTURE,
                out_dir,
                lane_change_mode="lanelet-infer",
                signal_mode="none",
                run_netconvert=False,
            )

            self.assertEqual(result["net_path"], None)
            self.assertNotIn("tll_path", result)
            self.assertEqual(Path(result["nodes_path"]), out_dir / "network.nod.xml")
            self.assertEqual(Path(result["edges_path"]), out_dir / "network.edg.xml")
            self.assertEqual(Path(result["connections_path"]), out_dir / "network.con.xml")
            self.assertEqual(Path(result["sidecar_path"]), out_dir / "retention.sidecar.json")
            self.assertEqual(result["signal_mapping_path"], None)
            self.assertEqual(Path(result["report_path"]), out_dir / "conversion.report.json")
            self.assertFalse((out_dir / "signal_id_mapping.json").exists())

            for file_name in ("network.nod.xml", "network.edg.xml", "network.con.xml"):
                with self.subTest(file_name=file_name):
                    actual = (out_dir / file_name).read_text(encoding="utf-8").strip()
                    expected = (MINIMAL_GOLDEN_DIR / file_name).read_text(encoding="utf-8").strip()
                    self.assertEqual(actual, expected)

            self.assertEqual(
                _plain_primary_counts(out_dir),
                {
                    "source": "plain SUMO XML files",
                    "edge_count": 1,
                    "lane_count": 1,
                    "node_count": 2,
                    "junction_count": 0,
                    "connection_count": 0,
                    "tlLogic_count": 0,
                    "phase_count": 0,
                },
            )
            self.assertEqual(
                _conversion_report_counts(out_dir),
                {
                    "lanelet_count": 1,
                    "road_lanelet_count": 1,
                    "lane_group_count": 1,
                    "edge_count": 1,
                    "node_count": 2,
                    "connection_count": 0,
                    "merged_serial_group_count": 0,
                    "merged_parallel_group_count": 0,
                    "intersection_area_cluster_count": 0,
                    "collapsed_intersection_group_count": 0,
                    "dropped_self_loop_lanelet_count": 0,
                    "dropped_tiny_lanelet_count": 0,
                },
            )

    def test_cli_plain_conversion_emits_result_json_and_expected_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ll2sumo.convert",
                    "--input",
                    str(MINIMAL_FIXTURE),
                    "--out-dir",
                    str(out_dir),
                    "--lane-change-mode",
                    "lanelet-infer",
                    "--signal-mode",
                    "none",
                    "--skip-netconvert",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertEqual(result["net_path"], None)
            self.assertNotIn("tll_path", result)
            self.assertEqual(result["signal_mapping_path"], None)
            for key in ("nodes_path", "edges_path", "connections_path", "sidecar_path", "report_path"):
                self.assertTrue(Path(result[key]).exists(), key)
            self.assertFalse((out_dir / "signal_id_mapping.json").exists())
            self.assertEqual(_plain_primary_counts(out_dir)["edge_count"], 1)
            self.assertEqual(_plain_primary_counts(out_dir)["connection_count"], 0)

    def test_jp_static_skip_netconvert_writes_planned_signal_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "minimal_signal.osm"
            out_dir = temp_path / "out"
            input_path.write_text(MINIMAL_SIGNAL_OSM, encoding="utf-8")

            result = convert_map(
                input_path,
                out_dir,
                lane_change_mode="lanelet-infer",
                signal_mode="jp-static",
                run_netconvert=False,
            )

            mapping_path = out_dir / "signal_id_mapping.json"
            self.assertEqual(Path(result["signal_mapping_path"]), mapping_path)
            self.assertTrue(mapping_path.exists())
            document = json.loads(mapping_path.read_text(encoding="utf-8"))
            self.assertEqual(document["schema_version"], 2)
            self.assertEqual(document["summary"]["planned_only_record_count"], 1)
            self.assertEqual(document["summary"]["mapped_record_count"], 0)
            self.assertEqual(document["summary"]["sumo_link_mapping_count"], 0)
            self.assertEqual(document["sumo_link_to_lanelet_signal"], [])
            self.assertEqual(document["lanelet_signal_to_sumo_links"], {})
            self.assertEqual(document["lanelet_to_sumo"][0]["lanelet_regulatory_element_ids"], ["900"])
            self.assertEqual(document["lanelet_to_sumo"][0]["lanelet_traffic_light_way_ids"], ["700"])
            self.assertEqual(document["lanelet_to_sumo"][0]["lanelet_ref_line_way_ids"], ["800"])
            self.assertEqual(document["lanelet_to_sumo"][0]["planned_sumo_tls_id"], "tls_300")
            self.assertEqual(document["lanelet_to_sumo"][0]["resolution_status"], "planned_only")
            report = json.loads((out_dir / "conversion.report.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(report["signal_id_mapping_path"]), mapping_path)
            self.assertEqual(report["signal_id_mapping_summary"]["planned_only_record_count"], 1)


if __name__ == "__main__":
    unittest.main()
