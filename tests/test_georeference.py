from __future__ import annotations

import tempfile
import textwrap
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from ll2sumo.georeference import infer_geo_reference, patch_net_location, project_wgs84_to_utm
from ll2sumo.model import GeoPoint, Point3D
from ll2sumo.parser import parse_lanelet_map


class UTMProjectionTest(unittest.TestCase):
    def test_projects_odaiba_node_to_utm_zone_54(self) -> None:
        easting, northing = project_wgs84_to_utm(35.62318359651, 139.77840697094)

        self.assertAlmostEqual(easting, 389376.6261, places=3)
        self.assertAlmostEqual(northing, 3942842.2608, places=3)


class GeoReferenceInferenceTest(unittest.TestCase):
    def test_infers_mgrs_local_offset_from_lanelet2_coordinates(self) -> None:
        nodes = {
            "3": Point3D(89376.6261, 42842.2608, 6.706),
            "4": Point3D(89376.5511, 42839.0494, 6.678),
        }
        node_geo = {
            "3": GeoPoint(lat=35.62318359651, lon=139.77840697094),
            "4": GeoPoint(lat=35.62315463909, lon=139.77840658327),
        }

        geo_reference = infer_geo_reference(nodes, node_geo)

        self.assertIsNotNone(geo_reference)
        assert geo_reference is not None
        self.assertEqual(geo_reference.utm_zone, 54)
        self.assertEqual(geo_reference.local_to_projected_offset_x, 300000.0)
        self.assertEqual(geo_reference.local_to_projected_offset_y, 3900000.0)
        self.assertLess(geo_reference.max_error_m, 0.01)

    def test_patches_sumo_net_location(self) -> None:
        nodes = {"n": Point3D(89376.6261, 42842.2608, 6.706)}
        node_geo = {"n": GeoPoint(lat=35.62318359651, lon=139.77840697094)}
        geo_reference = infer_geo_reference(nodes, node_geo)

        with tempfile.TemporaryDirectory() as temp_dir:
            net_path = Path(temp_dir) / "network.net.xml"
            net_path.write_text(
                textwrap.dedent(
                    """\
                    <?xml version="1.0" encoding="UTF-8"?>
                    <net>
                      <location netOffset="0.00,0.00" convBoundary="89376.63,42842.26,89376.63,42842.26" origBoundary="89376.63,42842.26,89376.63,42842.26" projParameter="!"/>
                      <junction id="n" x="89376.63" y="42842.26"/>
                    </net>
                    """
                ),
                encoding="utf-8",
            )

            patched = patch_net_location(net_path, geo_reference)
            location = ET.parse(net_path).getroot().find("location")

        self.assertTrue(patched)
        self.assertIsNotNone(location)
        assert location is not None
        self.assertEqual(location.attrib["netOffset"], "-300000,-3900000")
        self.assertEqual(location.attrib["origBoundary"], "389376.63,3942842.26,389376.63,3942842.26")
        self.assertIn("+proj=utm", location.attrib["projParameter"])
        self.assertIn("+zone=54", location.attrib["projParameter"])


class ParserGeoFallbackTest(unittest.TestCase):
    def test_parser_uses_utm_coordinates_when_local_xy_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            map_path = Path(temp_dir) / "map.osm"
            map_path.write_text(
                textwrap.dedent(
                    """\
                    <?xml version="1.0" encoding="UTF-8"?>
                    <osm>
                      <node id="1" lat="35.62318359651" lon="139.77840697094"/>
                      <node id="2" lat="35.62315463909" lon="139.77840658327"/>
                    </osm>
                    """
                ),
                encoding="utf-8",
            )

            lanelet_map = parse_lanelet_map(map_path)

        self.assertAlmostEqual(lanelet_map.nodes["1"].x, 389376.6261, places=3)
        self.assertAlmostEqual(lanelet_map.nodes["1"].y, 3942842.2608, places=3)
        self.assertIsNotNone(lanelet_map.geo_reference)
        assert lanelet_map.geo_reference is not None
        self.assertAlmostEqual(lanelet_map.geo_reference.local_to_projected_offset_x, 0.0, places=6)
        self.assertAlmostEqual(lanelet_map.geo_reference.local_to_projected_offset_y, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
