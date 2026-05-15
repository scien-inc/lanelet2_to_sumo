from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from statistics import median

from ll2sumo.model import GeoPoint, GeoReference, Point3D

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
UTM_K0 = 0.9996


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


def project_wgs84_to_utm(
    lat: float,
    lon: float,
    zone: int | None = None,
    northern: bool | None = None,
) -> tuple[float, float]:
    zone = zone or utm_zone_for_lon(lon)
    northern = (lat >= 0.0) if northern is None else northern
    false_northing = 0.0 if northern else 10000000.0

    e2 = WGS84_F * (2.0 - WGS84_F)
    ep2 = e2 / (1.0 - e2)
    phi = math.radians(lat)
    lam = math.radians(lon)
    lam0 = math.radians((zone - 1) * 6 - 180 + 3)

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    tan_phi = math.tan(phi)
    n = WGS84_A / math.sqrt(1.0 - e2 * sin_phi * sin_phi)
    t = tan_phi * tan_phi
    c = ep2 * cos_phi * cos_phi
    a = cos_phi * (lam - lam0)
    m = WGS84_A * (
        (1.0 - e2 / 4.0 - 3.0 * e2**2 / 64.0 - 5.0 * e2**3 / 256.0) * phi
        - (3.0 * e2 / 8.0 + 3.0 * e2**2 / 32.0 + 45.0 * e2**3 / 1024.0) * math.sin(2.0 * phi)
        + (15.0 * e2**2 / 256.0 + 45.0 * e2**3 / 1024.0) * math.sin(4.0 * phi)
        - (35.0 * e2**3 / 3072.0) * math.sin(6.0 * phi)
    )

    easting = UTM_K0 * n * (
        a
        + (1.0 - t + c) * a**3 / 6.0
        + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * ep2) * a**5 / 120.0
    ) + 500000.0
    northing = UTM_K0 * (
        m
        + n
        * tan_phi
        * (
            a * a / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c * c) * a**4 / 24.0
            + (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * ep2) * a**6 / 720.0
        )
    ) + false_northing
    return easting, northing


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

    offset_x_samples: list[float] = []
    offset_y_samples: list[float] = []
    for node_id in usable_ids:
        geo = node_geo[node_id]
        easting, northing = project_wgs84_to_utm(geo.lat, geo.lon, zone=zone, northern=northern)
        point = nodes[node_id]
        offset_x_samples.append(easting - point.x)
        offset_y_samples.append(northing - point.y)

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
