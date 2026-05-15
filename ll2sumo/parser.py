from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from ll2sumo.georeference import infer_geo_reference, project_wgs84_to_utm, utm_zone_for_lon
from ll2sumo.geometry import average_polylines, heading_deg, orient_lanelet, polyline_length
from ll2sumo.model import GeoPoint, Lanelet, LaneletMap, Point3D, RegulatoryElement, Way


def _node_tags(node: ET.Element) -> dict[str, str]:
    return {tag.attrib["k"]: tag.attrib["v"] for tag in node.findall("tag")}


def _node_geo(node: ET.Element) -> GeoPoint | None:
    if "lat" not in node.attrib or "lon" not in node.attrib:
        return None
    return GeoPoint(lat=float(node.attrib["lat"]), lon=float(node.attrib["lon"]))


def _node_point(
    node: ET.Element,
    tags: dict[str, str],
    fallback_utm_zone: int | None,
    fallback_northern: bool | None,
) -> Point3D:
    z = float(tags.get("ele", tags.get("local_z", "0")))
    if "local_x" in tags and "local_y" in tags:
        return Point3D(
            x=float(tags["local_x"]),
            y=float(tags["local_y"]),
            z=z,
        )

    geo = _node_geo(node)
    if geo is None:
        raise ValueError(f"Node {node.attrib.get('id', '<unknown>')} has neither local_x/local_y nor lat/lon coordinates")

    x, y = project_wgs84_to_utm(
        geo.lat,
        geo.lon,
        zone=fallback_utm_zone,
        northern=fallback_northern,
    )
    return Point3D(x=x, y=y, z=z)


def _fallback_utm_frame(raw_geo_nodes: dict[str, GeoPoint]) -> tuple[int | None, bool | None]:
    if not raw_geo_nodes:
        return None, None
    zone = Counter(utm_zone_for_lon(geo.lon) for geo in raw_geo_nodes.values()).most_common(1)[0][0]
    northern = sum(1 for geo in raw_geo_nodes.values() if geo.lat >= 0.0) >= len(raw_geo_nodes) / 2.0
    return zone, northern


def parse_lanelet_map(path: str | Path) -> LaneletMap:
    tree = ET.parse(path)
    root = tree.getroot()

    nodes: dict[str, Point3D] = {}
    node_geo: dict[str, GeoPoint] = {}
    ways: dict[str, Way] = {}
    lanelets: dict[str, Lanelet] = {}
    regulatory_elements: dict[str, RegulatoryElement] = {}

    raw_node_tags: dict[str, dict[str, str]] = {}
    for node in root.findall("node"):
        node_id = node.attrib["id"]
        raw_node_tags[node_id] = _node_tags(node)
        geo = _node_geo(node)
        if geo is not None:
            node_geo[node_id] = geo

    fallback_utm_zone, fallback_northern = _fallback_utm_frame(node_geo)
    for node in root.findall("node"):
        node_id = node.attrib["id"]
        nodes[node_id] = _node_point(node, raw_node_tags[node_id], fallback_utm_zone, fallback_northern)

    for way in root.findall("way"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
        ways[way.attrib["id"]] = Way(
            id=way.attrib["id"],
            node_ids=tuple(nd.attrib["ref"] for nd in way.findall("nd")),
            tags=tags,
        )

    for relation in root.findall("relation"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in relation.findall("tag")}
        if tags.get("type") == "regulatory_element":
            members_by_role: dict[str, list[str]] = {}
            for member in relation.findall("member"):
                members_by_role.setdefault(member.attrib.get("role", ""), []).append(member.attrib["ref"])
            regulatory_elements[relation.attrib["id"]] = RegulatoryElement(
                id=relation.attrib["id"],
                subtype=tags.get("subtype", ""),
                tags=tags,
                members_by_role={role: tuple(refs) for role, refs in members_by_role.items()},
            )
        if tags.get("type") != "lanelet":
            continue

        left_way_id = ""
        right_way_id = ""
        regulatory_ids: list[str] = []
        for member in relation.findall("member"):
            role = member.attrib.get("role")
            if role == "left":
                left_way_id = member.attrib["ref"]
            elif role == "right":
                right_way_id = member.attrib["ref"]
            elif role == "regulatory_element":
                regulatory_ids.append(member.attrib["ref"])

        if not left_way_id or not right_way_id:
            continue

        left_way = ways[left_way_id]
        right_way = ways[right_way_id]
        left_boundary = tuple(nodes[node_id] for node_id in left_way.node_ids)
        right_boundary = tuple(nodes[node_id] for node_id in right_way.node_ids)
        left_node_ids, right_node_ids, left_boundary, right_boundary = orient_lanelet(
            left_way.node_ids,
            right_way.node_ids,
            left_boundary,
            right_boundary,
        )
        centerline = average_polylines(left_boundary, right_boundary)
        lanelets[relation.attrib["id"]] = Lanelet(
            id=relation.attrib["id"],
            subtype=tags.get("subtype", "unknown"),
            tags=tags,
            left_way_id=left_way_id,
            right_way_id=right_way_id,
            regulatory_ids=tuple(regulatory_ids),
            left_node_ids=left_node_ids,
            right_node_ids=right_node_ids,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            centerline=centerline,
            start=centerline[0],
            end=centerline[-1],
            avg_heading_deg=heading_deg(centerline[0], centerline[-1]),
            length_m=polyline_length(centerline),
        )

    return LaneletMap(
        nodes=nodes,
        ways=ways,
        lanelets=lanelets,
        regulatory_elements=regulatory_elements,
        node_geo=node_geo,
        geo_reference=infer_geo_reference(nodes, node_geo),
    )
