from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import Counter
from functools import lru_cache
from pathlib import Path
from statistics import median

from pyproj import CRS, Transformer

from ll2sumo.model import GeoPoint, GeoReference, Point3D


def utm_zone_for_lon(lon: float) -> int:
    return int((lon + 180.0) // 6.0) + 1


def utm_proj_parameter(zone: int, northern: bool) -> str:
    parts = [
        "+proj=utm",
        f"+zone={zone}",
        "+ellps=WGS84",
        "+datum=WGS84",
        "+units=m",
        "+no_defs",
    ]
    if not northern:
        parts.insert(2, "+south")
    return " ".join(parts)


@lru_cache(maxsize=None)
def _utm_transformer(zone: int, northern: bool) -> Transformer:
    """Return a cached WGS84 to UTM transformer.

    Building a transformer is far more expensive than using one, and a map is
    projected node by node, so the instances are cached per UTM frame.
    """

    return Transformer.from_crs(
        "EPSG:4326",
        CRS.from_proj4(utm_proj_parameter(zone, northern)),
        always_xy=True,
    )


def _utm_frame(lat: float, lon: float, zone: int | None, northern: bool | None) -> tuple[int, bool]:
    return zone or utm_zone_for_lon(lon), (lat >= 0.0) if northern is None else northern


def project_wgs84_to_utm(
    lat: float,
    lon: float,
    zone: int | None = None,
    northern: bool | None = None,
) -> tuple[float, float]:
    easting, northing = _utm_transformer(*_utm_frame(lat, lon, zone, northern)).transform(lon, lat)
    return easting, northing


def project_many_wgs84_to_utm(
    coordinates: list[GeoPoint],
    zone: int,
    northern: bool,
) -> tuple[list[float], list[float]]:
    """Project a batch of points through one transformer call."""

    if not coordinates:
        return [], []
    eastings, northings = _utm_transformer(zone, northern).transform(
        [geo.lon for geo in coordinates],
        [geo.lat for geo in coordinates],
    )
    return list(eastings), list(northings)


def infer_geo_reference(
    nodes: dict[str, Point3D],
    node_geo: dict[str, GeoPoint],
    max_error_m: float = 1.0,
) -> GeoReference | None:
    usable_ids = [node_id for node_id in nodes if node_id in node_geo]
    if not usable_ids:
        return None

    zone_counts = Counter(utm_zone_for_lon(node_geo[node_id].lon) for node_id in usable_ids)
    zone = zone_counts.most_common(1)[0][0]
    northern = sum(1 for node_id in usable_ids if node_geo[node_id].lat >= 0.0) >= len(usable_ids) / 2.0

    eastings, northings = project_many_wgs84_to_utm(
        [node_geo[node_id] for node_id in usable_ids],
        zone,
        northern,
    )
    offset_x_samples = [easting - nodes[node_id].x for node_id, easting in zip(usable_ids, eastings)]
    offset_y_samples = [northing - nodes[node_id].y for node_id, northing in zip(usable_ids, northings)]

    offset_x = _snap_offset(offset_x_samples)
    offset_y = _snap_offset(offset_y_samples)
    errors = [
        math.hypot(offset_x_sample - offset_x, offset_y_sample - offset_y)
        for offset_x_sample, offset_y_sample in zip(offset_x_samples, offset_y_samples)
    ]
    max_error = max(errors)
    if max_error > max_error_m:
        return None

    return GeoReference(
        proj_parameter=utm_proj_parameter(zone, northern),
        utm_zone=zone,
        hemisphere="north" if northern else "south",
        local_to_projected_offset_x=offset_x,
        local_to_projected_offset_y=offset_y,
        sample_count=len(usable_ids),
        max_error_m=max_error,
        mean_error_m=sum(errors) / len(errors),
    )


def patch_net_location(net_path: str | Path, geo_reference: GeoReference | None) -> bool:
    if geo_reference is None:
        return False

    path = Path(net_path)
    tree = ET.parse(path)
    root = tree.getroot()
    location = root.find("location")
    if location is None:
        location = ET.Element("location")
        root.insert(0, location)

    conv_boundary = _parse_boundary(location.attrib.get("convBoundary"))
    if conv_boundary is None:
        conv_boundary = _network_boundary(root)
    if conv_boundary is None:
        return False

    net_offset_x = geo_reference.net_offset_x
    net_offset_y = geo_reference.net_offset_y
    orig_boundary = (
        conv_boundary[0] - net_offset_x,
        conv_boundary[1] - net_offset_y,
        conv_boundary[2] - net_offset_x,
        conv_boundary[3] - net_offset_y,
    )

    location.set("netOffset", f"{_format_float(net_offset_x)},{_format_float(net_offset_y)}")
    location.set("convBoundary", _format_boundary(conv_boundary))
    location.set("origBoundary", _format_boundary(orig_boundary))
    location.set("projParameter", geo_reference.proj_parameter)

    ET.indent(tree, space="    ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return True


def _snap_offset(samples: list[float]) -> float:
    value = median(samples)
    for grid, tolerance in ((100000.0, 0.5), (1.0, 0.05)):
        snapped = round(value / grid) * grid
        if max(abs(sample - snapped) for sample in samples) <= tolerance:
            return float(snapped)
    return float(value)


def _parse_boundary(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 4:
        return None
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def _network_boundary(root: ET.Element) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for element in root.iter():
        if "shape" in element.attrib:
            for token in element.attrib["shape"].split():
                coords = token.split(",")
                if len(coords) < 2:
                    continue
                xs.append(float(coords[0]))
                ys.append(float(coords[1]))
        if "x" in element.attrib and "y" in element.attrib:
            xs.append(float(element.attrib["x"]))
            ys.append(float(element.attrib["y"]))
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _format_boundary(boundary: tuple[float, float, float, float]) -> str:
    return ",".join(_format_float(value) for value in boundary)


def _format_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text or "0"
