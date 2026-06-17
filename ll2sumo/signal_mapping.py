from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from ll2sumo.model import Lanelet, RegulatoryElement
from ll2sumo.sumo_xml import id_sort_key as _sort_key


SIGNAL_MAPPING_SCHEMA_VERSION = 2
MAX_SIGNAL_MAPPING_EXAMPLES = 20

SignalMappingRecord = dict[str, object]
SignalLinkMappingRecord = dict[str, object]


class LaneGroupLike(Protocol):
    edge_id: str
    lanelet_paths: tuple[tuple[str, ...], ...]


def _intersection_area_id(lanelet: Lanelet) -> str | None:
    return lanelet.tags.get("intersection_area")


def _traffic_light_way_ids_for_relations(
    traffic_light_regs: dict[str, RegulatoryElement],
    relation_ids: Iterable[str],
    role: str,
) -> tuple[str, ...]:
    way_ids: set[str] = set()
    for relation_id in relation_ids:
        regulatory_element = traffic_light_regs.get(relation_id)
        if regulatory_element is None:
            continue
        way_ids.update(regulatory_element.members_by_role.get(role, ()))
    return tuple(sorted(way_ids, key=_sort_key))


def _signal_mapping_record(
    *,
    relation_ids: Iterable[str],
    traffic_light_regs: dict[str, RegulatoryElement],
    attached_lanelet_ids: Iterable[str],
    intersection_area_id: str | None,
    planned_sumo_tls_id: str | None,
    planned_sumo_node_ids: Iterable[str] = (),
    resolution_status: str,
    reason: str | None = None,
) -> SignalMappingRecord:
    sorted_relation_ids = tuple(sorted(relation_ids, key=_sort_key))
    record: SignalMappingRecord = {
        "lanelet_regulatory_element_ids": list(sorted_relation_ids),
        "lanelet_traffic_light_way_ids": list(
            _traffic_light_way_ids_for_relations(traffic_light_regs, sorted_relation_ids, "refers")
        ),
        "lanelet_ref_line_way_ids": list(
            _traffic_light_way_ids_for_relations(traffic_light_regs, sorted_relation_ids, "ref_line")
        ),
        "attached_lanelet_ids": list(sorted(attached_lanelet_ids, key=_sort_key)),
        "intersection_area_id": intersection_area_id,
        "planned_sumo_tls_id": planned_sumo_tls_id,
        "planned_sumo_node_ids": list(sorted(planned_sumo_node_ids, key=_sort_key)),
        "actual_sumo_tls_ids": [],
        "actual_sumo_junction_ids": [],
        "actual_sumo_connection_count": 0,
        "resolution_status": resolution_status,
    }
    if reason is not None:
        record["reason"] = reason
    return record


def _signal_mapping_record_sort_key(record: SignalMappingRecord) -> tuple[object, ...]:
    relation_ids = record.get("lanelet_regulatory_element_ids", [])
    first_relation_id = relation_ids[0] if isinstance(relation_ids, list) and relation_ids else ""
    return (
        _sort_key(str(record.get("intersection_area_id") or "")),
        _sort_key(str(first_relation_id)),
        _sort_key(str(record.get("planned_sumo_tls_id") or "")),
    )


def _record_string_list(record: dict[str, object], key: str) -> list[str]:
    values = record.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def _net_tls_ids(root: ET.Element) -> set[str]:
    return {element.attrib["id"] for element in root.findall("tlLogic") if "id" in element.attrib}


def _net_tls_connection_counts(root: ET.Element) -> dict[str, int]:
    tls_ids = _net_tls_ids(root)
    counts: Counter[str] = Counter()
    for connection in root.findall("connection"):
        tls_id = connection.attrib.get("tl")
        if tls_id in tls_ids:
            counts[str(tls_id)] += 1
    return dict(counts)


def _net_internal_lane_to_junction_id(root: ET.Element) -> dict[str, str]:
    lane_to_junction: dict[str, str] = {}
    for junction_element in root.findall("junction"):
        junction_id = junction_element.attrib.get("id")
        if not junction_id:
            continue
        if not junction_element.attrib.get("type", "").startswith("traffic_light"):
            continue
        for lane_id in junction_element.attrib.get("intLanes", "").split():
            lane_to_junction[lane_id] = junction_id
    return lane_to_junction


def _resolve_signal_mapping_records(
    records: list[SignalMappingRecord],
    net_path: str | Path | None,
) -> list[SignalMappingRecord]:
    if net_path is None:
        return [dict(record) for record in records]
    path = Path(net_path)
    if not path.exists():
        return [dict(record) for record in records]

    root = ET.parse(path).getroot()
    tls_ids = _net_tls_ids(root)
    connections = root.findall("connection")
    internal_lane_to_junction_id = _net_internal_lane_to_junction_id(root)
    resolved_records: list[SignalMappingRecord] = []

    for record in records:
        resolved = dict(record)
        if resolved.get("resolution_status") == "unmapped":
            resolved_records.append(resolved)
            continue

        intersection_area_id = resolved.get("intersection_area_id")
        planned_node_ids = _record_string_list(resolved, "planned_sumo_node_ids")
        planned_tls_id = resolved.get("planned_sumo_tls_id")
        actual_tls_ids: set[str] = set()

        if intersection_area_id:
            via_prefix = f":ia_{intersection_area_id}_"
            actual_tls_ids.update(
                connection.attrib["tl"]
                for connection in connections
                if connection.attrib.get("via", "").startswith(via_prefix)
                and connection.attrib.get("tl") in tls_ids
            )

        for node_id in planned_node_ids:
            via_prefix = f":{node_id}_"
            actual_tls_ids.update(
                connection.attrib["tl"]
                for connection in connections
                if connection.attrib.get("via", "").startswith(via_prefix)
                and connection.attrib.get("tl") in tls_ids
            )

        if isinstance(planned_tls_id, str) and planned_tls_id in tls_ids:
            actual_tls_ids.add(planned_tls_id)

        sorted_actual_tls_ids = sorted(actual_tls_ids, key=_sort_key)
        if not sorted_actual_tls_ids:
            resolved["actual_sumo_tls_ids"] = []
            resolved["actual_sumo_junction_ids"] = []
            resolved["actual_sumo_connection_count"] = 0
            resolved["resolution_status"] = "unmapped"
            resolved["reason"] = "actual_sumo_tls_not_found"
            resolved_records.append(resolved)
            continue

        actual_connections = [
            connection
            for connection in connections
            if connection.attrib.get("tl") in actual_tls_ids
        ]
        actual_junction_ids = {
            junction_id
            for connection in actual_connections
            for junction_id in [internal_lane_to_junction_id.get(connection.attrib.get("via", ""))]
            if junction_id is not None
        }
        resolved["actual_sumo_tls_ids"] = sorted_actual_tls_ids
        resolved["actual_sumo_junction_ids"] = sorted(actual_junction_ids, key=_sort_key)
        resolved["actual_sumo_connection_count"] = len(actual_connections)
        resolved["resolution_status"] = "mapped"
        resolved.pop("reason", None)
        resolved_records.append(resolved)

    return sorted(resolved_records, key=_signal_mapping_record_sort_key)


def _connection_link_index(connection: ET.Element) -> int | None:
    try:
        return int(connection.attrib["linkIndex"])
    except (KeyError, ValueError):
        return None


def _link_mapping_record_sort_key(record: SignalLinkMappingRecord) -> tuple[object, ...]:
    relation_ids = record.get("lanelet_regulatory_element_ids", [])
    first_relation_id = relation_ids[0] if isinstance(relation_ids, list) and relation_ids else ""
    link_index = record.get("linkIndex")
    return (
        _sort_key(str(record.get("actual_sumo_tls_id") or "")),
        int(link_index) if isinstance(link_index, int) else -1,
        _sort_key(str(record.get("from") or "")),
        _sort_key(str(record.get("fromLane") or "")),
        _sort_key(str(record.get("to") or "")),
        _sort_key(str(record.get("toLane") or "")),
        _sort_key(str(record.get("planned_sumo_tls_id") or "")),
        _sort_key(str(first_relation_id)),
    )


def _lanelet_paths_by_sumo_lane_key(
    lane_groups: list[LaneGroupLike],
    lanelet_to_lane_index: dict[str, int],
) -> dict[tuple[str, str], tuple[str, ...]]:
    lanelet_paths: dict[tuple[str, str], tuple[str, ...]] = {}
    for group in lane_groups:
        for lanelet_path in group.lanelet_paths:
            lane_index = next(
                (lanelet_to_lane_index[lanelet_id] for lanelet_id in lanelet_path if lanelet_id in lanelet_to_lane_index),
                None,
            )
            if lane_index is None:
                continue
            lanelet_paths[(group.edge_id, str(lane_index))] = tuple(lanelet_path)
    return lanelet_paths


def _extend_lanelet_paths_to_intersection_entry(
    lanelet_paths_by_lane_key: dict[tuple[str, str], tuple[str, ...]],
    successors: dict[str, list[str]],
    road_lanelets: dict[str, Lanelet],
    max_depth: int = 12,
) -> dict[tuple[str, str], tuple[str, ...]]:
    extended_paths: dict[tuple[str, str], tuple[str, ...]] = {}
    for lane_key, lanelet_path in lanelet_paths_by_lane_key.items():
        extended: list[str] = list(lanelet_path)
        visited: set[str] = set(lanelet_path)
        queue: list[tuple[str, int]] = [(lanelet_id, 0) for lanelet_id in lanelet_path]

        while queue:
            lanelet_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            lanelet = road_lanelets.get(lanelet_id)
            if lanelet is not None and _intersection_area_id(lanelet) is not None:
                continue
            for successor_lanelet_id in sorted(successors.get(lanelet_id, []), key=_sort_key):
                if successor_lanelet_id in visited or successor_lanelet_id not in road_lanelets:
                    continue
                visited.add(successor_lanelet_id)
                extended.append(successor_lanelet_id)
                successor_lanelet = road_lanelets[successor_lanelet_id]
                if _intersection_area_id(successor_lanelet) is None:
                    queue.append((successor_lanelet_id, depth + 1))

        extended_paths[lane_key] = tuple(extended)
    return extended_paths


def _signal_mapping_records_by_actual_tls(
    records: list[SignalMappingRecord],
) -> dict[str, list[SignalMappingRecord]]:
    records_by_tls: dict[str, list[SignalMappingRecord]] = defaultdict(list)
    for record in records:
        if record.get("resolution_status") != "mapped":
            continue
        for tls_id in _record_string_list(record, "actual_sumo_tls_ids"):
            records_by_tls[tls_id].append(record)
    return {
        tls_id: sorted(tls_records, key=_signal_mapping_record_sort_key)
        for tls_id, tls_records in records_by_tls.items()
    }


def _sumo_link_signal_mapping_record(
    connection: ET.Element,
    link_index: int,
    source_lanelet_ids: tuple[str, ...],
    mapping_record: SignalMappingRecord | None,
    match_status: str,
) -> SignalLinkMappingRecord:
    record: SignalLinkMappingRecord = {
        "actual_sumo_tls_id": connection.attrib.get("tl", ""),
        "linkIndex": link_index,
        "from": connection.attrib.get("from", ""),
        "to": connection.attrib.get("to", ""),
        "fromLane": connection.attrib.get("fromLane", ""),
        "toLane": connection.attrib.get("toLane", ""),
        "via": connection.attrib.get("via", ""),
        "dir": connection.attrib.get("dir", ""),
        "source_lanelet_ids": list(source_lanelet_ids),
        "lanelet_regulatory_element_ids": [],
        "lanelet_traffic_light_way_ids": [],
        "lanelet_ref_line_way_ids": [],
        "intersection_area_id": None,
        "planned_sumo_tls_id": None,
        "match_status": match_status,
    }
    if mapping_record is None:
        return record

    for key in (
        "lanelet_regulatory_element_ids",
        "lanelet_traffic_light_way_ids",
        "lanelet_ref_line_way_ids",
    ):
        record[key] = _record_string_list(mapping_record, key)
    record["intersection_area_id"] = mapping_record.get("intersection_area_id")
    record["planned_sumo_tls_id"] = mapping_record.get("planned_sumo_tls_id")
    return record


def _build_sumo_link_signal_mapping_records(
    records: list[SignalMappingRecord],
    net_path: str | Path | None,
    lanelet_paths_by_lane_key: dict[tuple[str, str], tuple[str, ...]],
) -> list[SignalLinkMappingRecord]:
    if net_path is None:
        return []
    path = Path(net_path)
    if not path.exists():
        return []

    root = ET.parse(path).getroot()
    tls_ids = _net_tls_ids(root)
    records_by_tls = _signal_mapping_records_by_actual_tls(records)
    link_records: list[SignalLinkMappingRecord] = []

    for connection in root.findall("connection"):
        tls_id = connection.attrib.get("tl")
        if tls_id not in tls_ids:
            continue
        link_index = _connection_link_index(connection)
        if link_index is None:
            continue
        source_lanelet_ids = lanelet_paths_by_lane_key.get(
            (connection.attrib.get("from", ""), connection.attrib.get("fromLane", "")),
            tuple(),
        )
        candidates = records_by_tls.get(str(tls_id), [])
        source_lanelet_id_set = set(source_lanelet_ids)
        matched = [
            record
            for record in candidates
            if source_lanelet_id_set.intersection(_record_string_list(record, "attached_lanelet_ids"))
        ]
        if matched:
            for record in matched:
                link_records.append(
                    _sumo_link_signal_mapping_record(
                        connection,
                        link_index,
                        source_lanelet_ids,
                        record,
                        "matched",
                    )
                )
            continue
        if candidates:
            for record in candidates:
                link_records.append(
                    _sumo_link_signal_mapping_record(
                        connection,
                        link_index,
                        source_lanelet_ids,
                        record,
                        "tls_group_fallback",
                    )
                )
            continue
        link_records.append(
            _sumo_link_signal_mapping_record(
                connection,
                link_index,
                source_lanelet_ids,
                None,
                "unmapped",
            )
        )

    return sorted(link_records, key=_link_mapping_record_sort_key)


def _lanelet_signal_to_sumo_links(
    link_records: list[SignalLinkMappingRecord],
) -> dict[str, list[dict[str, object]]]:
    entries: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[object, ...]] = set()
    for record in sorted(link_records, key=_link_mapping_record_sort_key):
        for way_id in _record_string_list(record, "lanelet_traffic_light_way_ids"):
            entry = {
                "actual_sumo_tls_id": record.get("actual_sumo_tls_id"),
                "linkIndex": record.get("linkIndex"),
                "from": record.get("from"),
                "to": record.get("to"),
                "fromLane": record.get("fromLane"),
                "toLane": record.get("toLane"),
                "via": record.get("via"),
                "dir": record.get("dir"),
                "match_status": record.get("match_status"),
                "lanelet_regulatory_element_ids": _record_string_list(record, "lanelet_regulatory_element_ids"),
                "intersection_area_id": record.get("intersection_area_id"),
                "planned_sumo_tls_id": record.get("planned_sumo_tls_id"),
            }
            key = (
                way_id,
                entry["actual_sumo_tls_id"],
                entry["linkIndex"],
                entry["from"],
                entry["to"],
                entry["fromLane"],
                entry["toLane"],
                tuple(entry["lanelet_regulatory_element_ids"]),
            )
            if key in seen:
                continue
            seen.add(key)
            entries[way_id].append(entry)
    return {way_id: entries[way_id] for way_id in sorted(entries, key=_sort_key)}


def _net_tls_phase_states(root: ET.Element) -> dict[str, list[str]]:
    states: dict[str, list[str]] = {}
    for tl_logic in root.findall("tlLogic"):
        tls_id = tl_logic.attrib.get("id")
        if not tls_id:
            continue
        states[tls_id] = [phase.attrib.get("state", "") for phase in tl_logic.findall("phase")]
    return states


def _tls_state_category(state: str) -> str:
    value = state.lower()
    if value == "r":
        return "red"
    if value == "y":
        return "yellow"
    if value == "g":
        return "green"
    if value == "o":
        return "off"
    return value or "unknown"


def _audit_mixed_lanelet_signal_phase_states(
    link_records: list[SignalLinkMappingRecord],
    tls_phase_states: dict[str, list[str]],
) -> tuple[int, list[dict[str, object]]]:
    links_by_way_and_tls: dict[tuple[str, str], set[int]] = defaultdict(set)
    for record in link_records:
        tls_id = record.get("actual_sumo_tls_id")
        link_index = record.get("linkIndex")
        if not isinstance(tls_id, str) or not isinstance(link_index, int):
            continue
        if record.get("match_status") == "unmapped":
            continue
        for way_id in _record_string_list(record, "lanelet_traffic_light_way_ids"):
            links_by_way_and_tls[(way_id, tls_id)].add(link_index)

    mixed_count = 0
    examples: list[dict[str, object]] = []
    for (way_id, tls_id), link_indices in sorted(
        links_by_way_and_tls.items(),
        key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1])),
    ):
        for phase_index, state in enumerate(tls_phase_states.get(tls_id, [])):
            categories = {
                _tls_state_category(state[link_index])
                for link_index in link_indices
                if 0 <= link_index < len(state)
            }
            if len(categories) <= 1:
                continue
            mixed_count += 1
            if len(examples) < MAX_SIGNAL_MAPPING_EXAMPLES:
                examples.append(
                    {
                        "type": "mixed_lanelet_signal_phase",
                        "lanelet_traffic_light_way_id": way_id,
                        "actual_sumo_tls_id": tls_id,
                        "phase_index": phase_index,
                        "linkIndices": sorted(link_indices),
                        "state": state,
                        "categories": sorted(categories),
                    }
                )
    return mixed_count, examples


def _signal_mapping_summary(
    records: list[SignalMappingRecord],
    sumo_to_lanelet: dict[str, dict[str, object]],
    link_records: list[SignalLinkMappingRecord],
    net_tls_ids: set[str] | None = None,
    mixed_phase_count: int = 0,
    mixed_phase_examples: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    regulatory_element_ids = {
        relation_id
        for record in records
        for relation_id in _record_string_list(record, "lanelet_regulatory_element_ids")
    }
    traffic_light_way_ids = {
        way_id
        for record in records
        for way_id in _record_string_list(record, "lanelet_traffic_light_way_ids")
    }
    status_counts = Counter(str(record.get("resolution_status", "")) for record in records)
    link_status_counts = Counter(str(record.get("match_status", "")) for record in link_records)
    covered_tls_ids = set(sumo_to_lanelet)
    covered_tls_ids.update(
        str(record.get("actual_sumo_tls_id"))
        for record in link_records
        if record.get("match_status") != "unmapped" and record.get("actual_sumo_tls_id")
    )
    unmapped_tls_ids = sorted((net_tls_ids or set()) - covered_tls_ids, key=_sort_key)
    examples = list(mixed_phase_examples or [])
    for tls_id in unmapped_tls_ids:
        if len(examples) >= MAX_SIGNAL_MAPPING_EXAMPLES:
            break
        examples.append(
            {
                "type": "unmapped_actual_sumo_tls",
                "actual_sumo_tls_id": tls_id,
            }
        )
    return {
        "record_count": len(records),
        "mapped_record_count": status_counts.get("mapped", 0),
        "unmapped_record_count": status_counts.get("unmapped", 0),
        "planned_only_record_count": status_counts.get("planned_only", 0),
        "actual_sumo_tls_count": len(sumo_to_lanelet),
        "lanelet_regulatory_element_count": len(regulatory_element_ids),
        "lanelet_traffic_light_way_count": len(traffic_light_way_ids),
        "sumo_link_mapping_count": len(link_records),
        "matched_sumo_link_count": link_status_counts.get("matched", 0),
        "fallback_sumo_link_count": link_status_counts.get("tls_group_fallback", 0),
        "unmapped_sumo_link_count": link_status_counts.get("unmapped", 0),
        "unmapped_actual_sumo_tls_count": len(unmapped_tls_ids),
        "mixed_lanelet_signal_phase_count": mixed_phase_count,
        "examples": examples,
    }


def _sumo_to_lanelet_mapping(
    records: list[SignalMappingRecord],
    connection_count_by_tls: dict[str, int] | None = None,
) -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    for record in records:
        if record.get("resolution_status") != "mapped":
            continue
        for tls_id in _record_string_list(record, "actual_sumo_tls_ids"):
            entry = entries.setdefault(
                tls_id,
                {
                    "lanelet_regulatory_element_ids": [],
                    "lanelet_traffic_light_way_ids": [],
                    "lanelet_ref_line_way_ids": [],
                    "intersection_area_ids": [],
                    "planned_sumo_tls_ids": [],
                    "actual_sumo_junction_ids": [],
                    "record_count": 0,
                    "actual_sumo_connection_count": 0,
                },
            )
            entry["record_count"] = int(entry["record_count"]) + 1
            if connection_count_by_tls is not None and tls_id in connection_count_by_tls:
                entry["actual_sumo_connection_count"] = connection_count_by_tls[tls_id]
            else:
                entry["actual_sumo_connection_count"] = max(
                    int(entry["actual_sumo_connection_count"]),
                    int(record.get("actual_sumo_connection_count", 0)),
                )
            for target_key, source_key in (
                ("lanelet_regulatory_element_ids", "lanelet_regulatory_element_ids"),
                ("lanelet_traffic_light_way_ids", "lanelet_traffic_light_way_ids"),
                ("lanelet_ref_line_way_ids", "lanelet_ref_line_way_ids"),
                ("actual_sumo_junction_ids", "actual_sumo_junction_ids"),
            ):
                values = set(_record_string_list(record, source_key))
                values.update(str(value) for value in entry[target_key])
                entry[target_key] = sorted(values, key=_sort_key)
            intersection_area_id = record.get("intersection_area_id")
            if intersection_area_id is not None:
                values = set(str(value) for value in entry["intersection_area_ids"])
                values.add(str(intersection_area_id))
                entry["intersection_area_ids"] = sorted(values, key=_sort_key)
            planned_tls_id = record.get("planned_sumo_tls_id")
            if planned_tls_id is not None:
                values = set(str(value) for value in entry["planned_sumo_tls_ids"])
                values.add(str(planned_tls_id))
                entry["planned_sumo_tls_ids"] = sorted(values, key=_sort_key)

    return {tls_id: entries[tls_id] for tls_id in sorted(entries, key=_sort_key)}


def _write_signal_id_mapping_json(
    path: str | Path,
    signal_mode: str,
    records: list[SignalMappingRecord],
    link_records: list[SignalLinkMappingRecord] | None = None,
    net_path: str | Path | None = None,
) -> dict[str, object]:
    sorted_records = sorted(records, key=_signal_mapping_record_sort_key)
    sorted_link_records = sorted(link_records or [], key=_link_mapping_record_sort_key)
    root = ET.parse(net_path).getroot() if net_path is not None and Path(net_path).exists() else None
    net_tls_ids = _net_tls_ids(root) if root is not None else set()
    connection_count_by_tls = _net_tls_connection_counts(root) if root is not None else None
    tls_phase_states = _net_tls_phase_states(root) if root is not None else {}
    sumo_to_lanelet = _sumo_to_lanelet_mapping(sorted_records, connection_count_by_tls)
    lanelet_signal_to_sumo_links = _lanelet_signal_to_sumo_links(sorted_link_records)
    mixed_phase_count, mixed_phase_examples = _audit_mixed_lanelet_signal_phase_states(
        sorted_link_records,
        tls_phase_states,
    )
    summary = _signal_mapping_summary(
        sorted_records,
        sumo_to_lanelet,
        sorted_link_records,
        net_tls_ids=net_tls_ids,
        mixed_phase_count=mixed_phase_count,
        mixed_phase_examples=mixed_phase_examples,
    )
    document = {
        "schema_version": SIGNAL_MAPPING_SCHEMA_VERSION,
        "signal_mode": signal_mode,
        "lanelet_to_sumo": sorted_records,
        "sumo_to_lanelet": sumo_to_lanelet,
        "sumo_link_to_lanelet_signal": sorted_link_records,
        "lanelet_signal_to_sumo_links": lanelet_signal_to_sumo_links,
        "summary": summary,
    }
    Path(path).write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return summary
