"""Render Lanelet2 signal positions with animated SUMO-derived timing."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from ll2sumo.model import Point3D
from ll2sumo.parser import parse_lanelet_map


STATE_PRIORITY = {
    "off": 0,
    "green": 1,
    "yellow": 2,
    "red": 3,
    "unknown": 4,
}


def _state_category(value: str) -> str:
    if value in {"r", "R"}:
        return "red"
    if value in {"y", "Y"}:
        return "yellow"
    if value in {"g", "G", "s"}:
        return "green"
    if value in {"o", "O"}:
        return "off"
    return "unknown"


def _aggregate_categories(categories: list[str]) -> str:
    if not categories:
        return "unknown"
    return max(categories, key=lambda category: STATE_PRIORITY[category])


def _sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):020d}")
    except ValueError:
        return (1, value)


def _polyline_midpoint(points: list[Point3D]) -> Point3D | None:
    if not points:
        return None
    if len(points) == 1:
        return points[0]

    segment_lengths: list[float] = []
    total = 0.0
    for a, b in zip(points, points[1:]):
        length = ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5
        segment_lengths.append(length)
        total += length
    if total <= 0:
        return points[len(points) // 2]

    target = total / 2.0
    travelled = 0.0
    for index, length in enumerate(segment_lengths):
        if travelled + length >= target:
            a = points[index]
            b = points[index + 1]
            ratio = (target - travelled) / length if length > 0 else 0.0
            return Point3D(
                x=a.x + (b.x - a.x) * ratio,
                y=a.y + (b.y - a.y) * ratio,
                z=a.z + (b.z - a.z) * ratio,
            )
        travelled += length
    return points[-1]


def _load_tl_logics(net_path: Path) -> dict[str, list[dict[str, Any]]]:
    root = ET.parse(net_path).getroot()
    tl_logics: dict[str, list[dict[str, Any]]] = {}
    for tl_logic in root.findall("tlLogic"):
        tls_id = tl_logic.get("id")
        if not tls_id:
            continue
        phases: list[dict[str, Any]] = []
        for phase in tl_logic.findall("phase"):
            duration_text = phase.get("duration", "0")
            try:
                duration = float(duration_text)
            except ValueError:
                duration = 0.0
            phases.append({"duration": max(duration, 0.0), "state": phase.get("state", "")})
        tl_logics[tls_id] = phases
    return tl_logics


def _load_mapping(mapping_path: Path) -> dict[str, Any]:
    with mapping_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _vehicle_signal_way_info(mapping: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return Lanelet2 vehicle traffic_light way IDs that should be drawn."""
    info: dict[str, dict[str, Any]] = {}
    for record in mapping.get("lanelet_to_sumo", []):
        if not isinstance(record, dict):
            continue
        if record.get("resolution_status") == "excluded_non_vehicle":
            continue
        way_ids = [str(way_id) for way_id in record.get("lanelet_traffic_light_way_ids", [])]
        for way_id in way_ids:
            way_info = info.setdefault(
                way_id,
                {
                    "regulatory_element_ids": set(),
                    "intersection_area_ids": set(),
                    "planned_sumo_tls_ids": set(),
                    "actual_sumo_tls_ids": set(),
                    "resolution_statuses": Counter(),
                },
            )
            way_info["regulatory_element_ids"].update(str(item) for item in record.get("lanelet_regulatory_element_ids", []))
            if record.get("intersection_area_id"):
                way_info["intersection_area_ids"].add(str(record["intersection_area_id"]))
            if record.get("planned_sumo_tls_id"):
                way_info["planned_sumo_tls_ids"].add(str(record["planned_sumo_tls_id"]))
            way_info["actual_sumo_tls_ids"].update(str(item) for item in record.get("actual_sumo_tls_ids", []))
            way_info["resolution_statuses"][str(record.get("resolution_status", "unknown"))] += 1
    return info


def _direct_matched_links_by_way(mapping: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return only direct SUMO link matches, intentionally excluding fallback records."""
    links_by_way: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in mapping.get("sumo_link_to_lanelet_signal", []):
        if not isinstance(record, dict):
            continue
        if record.get("match_status") != "matched":
            continue
        for way_id in record.get("lanelet_traffic_light_way_ids", []):
            links_by_way[str(way_id)].append(record)
    return links_by_way


def _screen_point(point: Point3D, max_y: float) -> dict[str, float]:
    return {"x": round(point.x, 3), "y": round(max_y - point.y, 3)}


def _bounds(points: list[Point3D]) -> dict[str, float]:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    margin = max(min(width, height) * 0.03, 20.0)
    return {
        "min_x": min_x - margin,
        "max_x": max_x + margin,
        "min_y": min_y - margin,
        "max_y": max_y + margin,
        "width": width + margin * 2.0,
        "height": height + margin * 2.0,
        "margin": margin,
    }


def _build_signal_phase_groups(
    links: list[dict[str, Any]],
    tl_logics: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    links_by_tls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for link in links:
        tls_id = link.get("actual_sumo_tls_id")
        if tls_id:
            links_by_tls[str(tls_id)].append(link)

    phase_groups: list[dict[str, Any]] = []
    for tls_id, tls_links in sorted(links_by_tls.items()):
        phases = tl_logics.get(tls_id)
        if not phases:
            continue
        states: list[dict[str, Any]] = []
        for phase in phases:
            state_text = phase["state"]
            categories: list[str] = []
            raw_values: list[str] = []
            for link in tls_links:
                try:
                    link_index = int(link.get("linkIndex"))
                except (TypeError, ValueError):
                    continue
                if 0 <= link_index < len(state_text):
                    raw_value = state_text[link_index]
                    raw_values.append(raw_value)
                    categories.append(_state_category(raw_value))
            states.append(
                {
                    "duration": phase["duration"],
                    "category": _aggregate_categories(categories),
                    "raw": dict(Counter(raw_values)),
                }
            )
        phase_groups.append(
            {
                "tlsId": tls_id,
                "cycle": round(sum(state["duration"] for state in states), 3),
                "linkIndices": sorted(
                    {
                        int(link["linkIndex"])
                        for link in tls_links
                        if str(link.get("linkIndex", "")).lstrip("-").isdigit()
                    }
                ),
                "syncStatus": dict(Counter(str(link.get("sync_status", "unknown")) for link in tls_links)),
                "matchStatus": dict(Counter(str(link.get("match_status", "unknown")) for link in tls_links)),
                "phases": states,
            }
        )
    return phase_groups


def _build_payload(osm_path: Path, mapping_path: Path, net_path: Path) -> tuple[dict[str, Any], list[str]]:
    lanelet_map = parse_lanelet_map(osm_path)
    mapping = _load_mapping(mapping_path)
    tl_logics = _load_tl_logics(net_path)
    way_info = _vehicle_signal_way_info(mapping)
    direct_links_by_way = _direct_matched_links_by_way(mapping)
    if not way_info:
        raise ValueError("signal_id_mapping.json does not contain vehicle Lanelet2 signal ways")

    used_way_ids = set(way_info)
    road_points: list[Point3D] = []
    for lanelet in lanelet_map.lanelets.values():
        if lanelet.subtype == "road":
            road_points.extend(lanelet.centerline)

    signal_points: list[Point3D] = []
    signal_way_points: dict[str, list[Point3D]] = {}
    warnings: list[str] = []
    for way_id in used_way_ids:
        way = lanelet_map.ways.get(way_id)
        if way is None:
            warnings.append(f"Lanelet2 traffic_light way not found in OSM: {way_id}")
            continue
        points = [lanelet_map.nodes[node_id] for node_id in way.node_ids if node_id in lanelet_map.nodes]
        midpoint = _polyline_midpoint(points)
        if midpoint is None:
            warnings.append(f"Lanelet2 traffic_light way has no usable nodes: {way_id}")
            continue
        signal_points.append(midpoint)
        signal_way_points[way_id] = points

    all_points = road_points + signal_points
    if not all_points:
        raise ValueError("No drawable Lanelet2 points found")
    raw_bounds = _bounds(all_points)
    max_y = raw_bounds["max_y"]

    road_paths: list[list[dict[str, float]]] = []
    for lanelet in sorted(lanelet_map.lanelets.values(), key=lambda item: _sort_key(item.id)):
        if lanelet.subtype != "road":
            continue
        if len(lanelet.centerline) < 2:
            continue
        road_paths.append([_screen_point(point, max_y) for point in lanelet.centerline])

    signals: list[dict[str, Any]] = []
    unmapped_way_count = 0
    for way_id, info in sorted(way_info.items(), key=lambda item: _sort_key(str(item[0]))):
        points = signal_way_points.get(str(way_id))
        if not points:
            continue
        midpoint = _polyline_midpoint(points)
        if midpoint is None:
            continue
        typed_links = [link for link in direct_links_by_way.get(str(way_id), []) if isinstance(link, dict)]
        reg_ids = sorted(
            set(info["regulatory_element_ids"])
            | {str(reg_id) for link in typed_links for reg_id in link.get("lanelet_regulatory_element_ids", [])},
            key=_sort_key,
        )
        intersections = sorted(
            set(info["intersection_area_ids"])
            | {str(link["intersection_area_id"]) for link in typed_links if link.get("intersection_area_id")},
            key=_sort_key,
        )
        phase_groups = _build_signal_phase_groups(typed_links, tl_logics)
        has_direct_match = bool(phase_groups)
        if not has_direct_match:
            unmapped_way_count += 1
        signals.append(
            {
                "wayId": str(way_id),
                "position": _screen_point(midpoint, max_y),
                "line": [_screen_point(point, max_y) for point in points],
                "hasDirectMatch": has_direct_match,
                "regulatoryElementIds": reg_ids,
                "intersectionAreaIds": intersections,
                "plannedSumoTlsIds": sorted(info["planned_sumo_tls_ids"]),
                "actualSumoTlsIds": sorted(info["actual_sumo_tls_ids"]),
                "resolutionStatuses": dict(info["resolution_statuses"]),
                "linkCount": len(typed_links),
                "phaseGroups": phase_groups,
            }
        )

    viewport = {
        "minX": round(raw_bounds["min_x"], 3),
        "minY": round(max_y - raw_bounds["max_y"], 3),
        "width": round(raw_bounds["width"], 3),
        "height": round(raw_bounds["height"], 3),
    }
    payload = {
        "meta": {
            "osmPath": str(osm_path),
            "netPath": str(net_path),
            "mappingPath": str(mapping_path),
            "schemaVersion": mapping.get("schema_version"),
            "summary": mapping.get("summary", {}),
            "signalCount": len(signals),
            "directMatchedSignalCount": len(signals) - unmapped_way_count,
            "unmappedSignalCount": unmapped_way_count,
            "tlsCount": len({group["tlsId"] for signal in signals for group in signal["phaseGroups"]}),
            "roadPathCount": len(road_paths),
            "visualizationMode": "direct_matched_only",
        },
        "viewport": viewport,
        "roadPaths": road_paths,
        "signals": signals,
    }
    return payload, warnings


def _html_document(payload: dict[str, Any], warnings: list[str]) -> str:
    payload_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    warnings_text = json.dumps(warnings, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lanelet2 Signal Timing Map</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --border: #d7dbe2;
      --text: #20242b;
      --muted: #626b78;
      --road: #a7afb9;
      --signal-line: #4b5563;
      --red: #d62525;
      --yellow: #e3b514;
      --green: #14a553;
      --off: #8f98a4;
      --unknown: #6b4cb3;
      --missing: #111827;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      background: var(--bg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }}
    header {{
      display: flex;
      align-items: center;
      gap: 16px;
      min-height: 64px;
      padding: 12px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
      white-space: nowrap;
    }}
    button {{
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      cursor: pointer;
      font: inherit;
      height: 34px;
      padding: 0 12px;
    }}
    label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      white-space: nowrap;
    }}
    input[type="range"] {{ width: 140px; }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: calc(100vh - 64px);
    }}
    #mapWrap {{
      position: relative;
      overflow: hidden;
      min-height: calc(100vh - 64px);
      background: #fbfcfd;
    }}
    svg {{
      display: block;
      width: 100%;
      height: calc(100vh - 64px);
    }}
    .road {{
      fill: none;
      stroke: var(--road);
      stroke-width: 1.25;
      stroke-linecap: round;
      stroke-linejoin: round;
      opacity: 0.5;
      vector-effect: non-scaling-stroke;
    }}
    .signal-line {{
      fill: none;
      stroke: var(--signal-line);
      stroke-width: 2.6;
      stroke-linecap: round;
      vector-effect: non-scaling-stroke;
    }}
    #markerLayer {{
      position: absolute;
      inset: 0;
      pointer-events: none;
    }}
    .signal-marker {{
      position: absolute;
      width: 6px;
      height: 6px;
      min-width: 0;
      padding: 0;
      border: 1px solid rgba(17, 24, 39, 0.72);
      border-radius: 999px;
      appearance: none;
      cursor: pointer;
      font-size: 0;
      pointer-events: auto;
      transform: translate(-50%, -50%);
      box-shadow: none;
    }}
    .signal-marker.selected {{
      width: 12px;
      height: 12px;
      border: 2px solid #111827;
      outline: 2px solid rgba(17, 24, 39, 0.35);
      outline-offset: 1px;
      z-index: 4;
    }}
    .signal-marker.missing {{
      width: 10px;
      height: 10px;
      border: 0;
      border-radius: 0;
      background: transparent;
      box-shadow: none;
    }}
    .signal-marker.missing::before,
    .signal-marker.missing::after {{
      content: "";
      position: absolute;
      left: 4px;
      top: -1px;
      width: 2px;
      height: 12px;
      border-radius: 999px;
      background: var(--missing);
    }}
    .signal-marker.missing::before {{ transform: rotate(45deg); }}
    .signal-marker.missing::after {{ transform: rotate(-45deg); }}
    .signal-marker.state-red {{ background: var(--red); }}
    .signal-marker.state-yellow {{ background: var(--yellow); }}
    .signal-marker.state-green {{ background: var(--green); }}
    .signal-marker.state-off {{ background: var(--off); }}
    .signal-marker.state-unknown {{ background: var(--unknown); }}
    .signal-marker.state-missing {{ background: transparent; }}
    .signal-marker[data-density="dense"] {{
      width: 5px;
      height: 5px;
      border-width: 1px;
    }}
    aside {{
      border-left: 1px solid var(--border);
      background: var(--panel);
      padding: 16px;
      overflow: auto;
      max-height: calc(100vh - 64px);
    }}
    .stat {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 5px 0;
      border-bottom: 1px solid #edf0f4;
    }}
    .stat span:first-child, .hint, .small {{
      color: var(--muted);
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
    }}
    .dot {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    .detail {{
      margin-top: 16px;
      border-top: 1px solid var(--border);
      padding-top: 14px;
    }}
    .phaseBar {{
      display: flex;
      overflow: hidden;
      border: 1px solid var(--border);
      border-radius: 5px;
      height: 22px;
      margin: 6px 0 10px;
    }}
    .phaseSeg {{
      min-width: 12px;
      border-right: 1px solid rgba(255,255,255,0.55);
    }}
    .warnings {{
      margin-top: 14px;
      color: #7c2d12;
      background: #fff7ed;
      border: 1px solid #fed7aa;
      border-radius: 6px;
      padding: 10px;
    }}
    @media (max-width: 980px) {{
      header {{ flex-wrap: wrap; }}
      main {{ grid-template-columns: 1fr; }}
      aside {{
        border-left: 0;
        border-top: 1px solid var(--border);
        max-height: none;
      }}
      svg {{ height: 70vh; }}
      #mapWrap {{ min-height: 70vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Lanelet2 Signal Timing Map</h1>
    <button id="playPause">Pause</button>
    <label>speed <input id="speed" type="range" min="0" max="20" step="0.5" value="1"><span id="speedLabel">1.0x</span></label>
    <label>time <input id="timeSlider" type="range" min="0" max="120" step="0.1" value="0"><span id="timeLabel">0.0s</span></label>
  </header>
  <main>
    <div id="mapWrap">
      <svg id="map" aria-label="Lanelet2 signal timing map"></svg>
      <div id="markerLayer" aria-label="Lanelet2 signal markers"></div>
    </div>
    <aside>
      <section>
        <h2>Summary</h2>
        <div id="summary"></div>
        <div class="legend">
          <span class="chip"><span class="dot" style="background:var(--red)"></span>red</span>
          <span class="chip"><span class="dot" style="background:var(--yellow)"></span>yellow</span>
          <span class="chip"><span class="dot" style="background:var(--green)"></span>green</span>
          <span class="chip"><span class="dot" style="background:var(--off)"></span>off</span>
          <span class="chip"><span style="color:var(--missing);font-weight:800">×</span>no direct match</span>
        </div>
        <p class="hint">Signals are placed from Lanelet2 <code>traffic_light/refers</code> way geometry. Colors use only direct SUMO link matches. Fallback mappings are intentionally excluded; signals without a direct matched SUMO link are shown as <strong>×</strong>.</p>
      </section>
      <section class="detail">
        <h2>Selected Signal</h2>
        <div id="detail" class="hint">Click a signal marker.</div>
      </section>
      <section id="warnings" class="warnings" hidden></section>
    </aside>
  </main>
  <script>
    const payload = {payload_text};
    const warnings = {warnings_text};
    const priority = {{off: 0, green: 1, yellow: 2, red: 3, unknown: 4, missing: 5}};
    const cssClass = {{red: "state-red", yellow: "state-yellow", green: "state-green", off: "state-off", unknown: "state-unknown", missing: "state-missing"}};
    let playing = true;
    let simTime = 0;
    let lastFrame = performance.now();
    let selectedWayId = null;

    const svg = document.getElementById("map");
    const summary = document.getElementById("summary");
    const detail = document.getElementById("detail");
    const playPause = document.getElementById("playPause");
    const speed = document.getElementById("speed");
    const speedLabel = document.getElementById("speedLabel");
    const timeSlider = document.getElementById("timeSlider");
    const timeLabel = document.getElementById("timeLabel");
    const warningsBox = document.getElementById("warnings");
    const markerLayer = document.getElementById("markerLayer");

    const maxCycle = Math.max(1, ...payload.signals.flatMap(signal => signal.phaseGroups.map(group => group.cycle || 0)));
    timeSlider.max = String(maxCycle);

    function pathData(points) {{
      return points.map((p, index) => `${{index === 0 ? "M" : "L"}}${{p.x}},${{p.y}}`).join(" ");
    }}

    function escapeText(value) {{
      return String(value).replace(/[&<>"']/g, ch => ({{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}}[ch]));
    }}

    function stateAt(group, t) {{
      if (!group.phases.length || !group.cycle) return "unknown";
      let local = ((t % group.cycle) + group.cycle) % group.cycle;
      for (const phase of group.phases) {{
        if (local < phase.duration) return phase.category;
        local -= phase.duration;
      }}
      return group.phases[group.phases.length - 1].category || "unknown";
    }}

    function signalState(signal, t) {{
      if (!signal.hasDirectMatch) return "missing";
      const states = signal.phaseGroups.map(group => stateAt(group, t));
      return states.reduce((best, state) => priority[state] > priority[best] ? state : best, "off");
    }}

    function projectToMarkerLayer(point) {{
      const svgPoint = svg.createSVGPoint();
      svgPoint.x = point.x;
      svgPoint.y = point.y;
      const ctm = svg.getScreenCTM();
      if (!ctm) return {{x: 0, y: 0}};
      const screenPoint = svgPoint.matrixTransform(ctm);
      const layerRect = markerLayer.getBoundingClientRect();
      return {{
        x: screenPoint.x - layerRect.left,
        y: screenPoint.y - layerRect.top,
      }};
    }}

    function positionMarkers() {{
      for (const marker of markerLayer.querySelectorAll(".signal-marker")) {{
        const wayId = marker.getAttribute("data-way-id");
        const signal = payload.signals.find(item => item.wayId === wayId);
        if (!signal) continue;
        const point = projectToMarkerLayer(signal.position);
        marker.style.left = `${{point.x}}px`;
        marker.style.top = `${{point.y}}px`;
      }}
    }}

    function renderMap() {{
      const vp = payload.viewport;
      svg.setAttribute("viewBox", `${{vp.minX}} ${{vp.minY}} ${{vp.width}} ${{vp.height}}`);

      const roadGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      for (const road of payload.roadPaths) {{
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("class", "road");
        path.setAttribute("d", pathData(road));
        roadGroup.appendChild(path);
      }}
      svg.appendChild(roadGroup);

      const lineGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      for (const signal of payload.signals) {{
        const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
        line.setAttribute("class", "signal-line");
        line.setAttribute("d", pathData(signal.line));
        lineGroup.appendChild(line);
      }}
      svg.appendChild(lineGroup);

      markerLayer.innerHTML = "";
      for (const signal of payload.signals) {{
        const marker = document.createElement("button");
        marker.type = "button";
        marker.className = signal.hasDirectMatch ? "signal-marker state-unknown" : "signal-marker state-missing missing";
        marker.dataset.wayId = signal.wayId;
        marker.title = `Lanelet2 signal way ${{signal.wayId}}`;
        marker.setAttribute("aria-label", `Lanelet2 signal way ${{signal.wayId}}`);
        marker.addEventListener("click", () => {{
          selectedWayId = signal.wayId;
          update(simTime);
        }});
        markerLayer.appendChild(marker);
      }}
      positionMarkers();
    }}

    function setClass(element, state, selected) {{
      const missing = state === "missing";
      element.setAttribute(
        "class",
        `signal-marker ${{cssClass[state] || cssClass.unknown}}${{missing ? " missing" : ""}}${{selected ? " selected" : ""}}`
      );
    }}

    function renderSummary() {{
      const meta = payload.meta;
      const s = meta.summary || {{}};
      const rows = [
        ["Lanelet2 signal ways", meta.signalCount],
        ["direct matched ways", meta.directMatchedSignalCount],
        ["no direct match", meta.unmappedSignalCount],
        ["SUMO TLS groups", meta.tlsCount],
        ["road centerlines", meta.roadPathCount],
        ["mixed sync count", s.mixed_lanelet_signal_phase_count ?? "-"],
        ["diagnostic mixed count", s.diagnostic_mixed_lanelet_signal_phase_count ?? "-"],
        ["visualization mode", meta.visualizationMode],
      ];
      summary.innerHTML = rows.map(([k, v]) => `<div class="stat"><span>${{escapeText(k)}}</span><strong>${{escapeText(v)}}</strong></div>`).join("");
      if (warnings.length) {{
        warningsBox.hidden = false;
        warningsBox.innerHTML = `<strong>Warnings</strong><ul>${{warnings.map(w => `<li>${{escapeText(w)}}</li>`).join("")}}</ul>`;
      }}
    }}

    function renderPhaseBar(group) {{
      const cycle = group.cycle || 1;
      return `<div class="phaseBar">${{group.phases.map(phase => {{
        const width = Math.max(0.75, phase.duration / cycle * 100);
        return `<span class="phaseSeg ${{cssClass[phase.category] || cssClass.unknown}}" style="width:${{width}}%" title="${{escapeText(phase.category)}} ${{phase.duration}}s"></span>`;
      }}).join("")}}</div>`;
    }}

    function renderDetail(signal, state) {{
      if (!signal) {{
        detail.className = "hint";
        detail.textContent = "Click a signal marker.";
        return;
      }}
      detail.className = "";
      detail.innerHTML = `
        <div class="stat"><span>current state</span><strong>${{escapeText(state)}}</strong></div>
        <div class="stat"><span>Lanelet2 refers way</span><code>${{escapeText(signal.wayId)}}</code></div>
        <div class="stat"><span>regulatory elements</span><code>${{escapeText(signal.regulatoryElementIds.join(", ") || "-")}}</code></div>
        <div class="stat"><span>intersection areas</span><code>${{escapeText(signal.intersectionAreaIds.join(", ") || "-")}}</code></div>
        <div class="stat"><span>direct matched links</span><strong>${{escapeText(signal.linkCount)}}</strong></div>
        <div class="stat"><span>planned SUMO TLS</span><code>${{escapeText(signal.plannedSumoTlsIds.join(", ") || "-")}}</code></div>
        <div class="stat"><span>actual SUMO TLS</span><code>${{escapeText(signal.actualSumoTlsIds.join(", ") || "-")}}</code></div>
        <div class="stat"><span>group status</span><code>${{escapeText(Object.entries(signal.resolutionStatuses).map(([k, v]) => `${{k}}:${{v}}`).join(", ") || "-")}}</code></div>
        <h3>Direct Matched SUMO TLS</h3>
        ${{signal.hasDirectMatch ? signal.phaseGroups.map(group => `
          <div>
            <code>${{escapeText(group.tlsId)}}</code>
            <div class="small">cycle ${{escapeText(group.cycle)}}s / linkIndex ${{escapeText(group.linkIndices.join(", "))}}</div>
            ${{renderPhaseBar(group)}}
          </div>
        `).join("") : `<p class="hint">No direct matched SUMO link. This signal may only be covered by fallback mapping, so it is not colored in this view.</p>`}}
      `;
    }}

    function update(t) {{
      const markers = markerLayer.querySelectorAll(".signal-marker");
      let selectedSignal = null;
      let selectedState = "unknown";
      for (const marker of markers) {{
        const wayId = marker.getAttribute("data-way-id");
        const signal = payload.signals.find(item => item.wayId === wayId);
        const state = signalState(signal, t);
        const selected = wayId === selectedWayId;
        setClass(marker, state, selected);
        if (selected) {{
          selectedSignal = signal;
          selectedState = state;
        }}
      }}
      if (selectedSignal) renderDetail(selectedSignal, selectedState);
      timeLabel.textContent = `${{(t % maxCycle).toFixed(1)}}s`;
      if (Math.abs(Number(timeSlider.value) - (t % maxCycle)) > 0.15) {{
        timeSlider.value = String(t % maxCycle);
      }}
    }}

    playPause.addEventListener("click", () => {{
      playing = !playing;
      playPause.textContent = playing ? "Pause" : "Play";
      lastFrame = performance.now();
    }});
    speed.addEventListener("input", () => {{
      speedLabel.textContent = `${{Number(speed.value).toFixed(1)}}x`;
    }});
    window.addEventListener("resize", positionMarkers);
    timeSlider.addEventListener("input", () => {{
      simTime = Number(timeSlider.value);
      playing = false;
      playPause.textContent = "Play";
      update(simTime);
    }});

    function tick(now) {{
      const elapsed = (now - lastFrame) / 1000;
      lastFrame = now;
      if (playing) {{
        simTime += elapsed * Number(speed.value);
        update(simTime);
      }}
      requestAnimationFrame(tick);
    }}

    renderMap();
    renderSummary();
    update(simTime);
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""


def render_html(osm_path: Path, mapping_path: Path, net_path: Path, html_path: Path) -> dict[str, Any]:
    payload, warnings = _build_payload(osm_path, mapping_path, net_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_html_document(payload, warnings), encoding="utf-8")
    return {
        "html_path": str(html_path),
        "lanelet_signal_way_count": payload["meta"]["signalCount"],
        "direct_matched_signal_way_count": payload["meta"]["directMatchedSignalCount"],
        "no_direct_match_signal_way_count": payload["meta"]["unmappedSignalCount"],
        "displayed_tls_count": payload["meta"]["tlsCount"],
        "road_path_count": payload["meta"]["roadPathCount"],
        "warning_count": len(warnings),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osm", type=Path, required=True, help="Lanelet2 OSM path used for signal positions.")
    parser.add_argument("--out-dir", type=Path, help="Directory containing network.net.xml and signal_id_mapping.json.")
    parser.add_argument("--net", type=Path, help="SUMO network.net.xml path.")
    parser.add_argument("--mapping", type=Path, help="signal_id_mapping.json path.")
    parser.add_argument("--html", type=Path, help="Output HTML path.")
    args = parser.parse_args(argv)

    net_path = args.net
    mapping_path = args.mapping
    html_path = args.html
    if args.out_dir is not None:
        net_path = net_path or args.out_dir / "network.net.xml"
        mapping_path = mapping_path or args.out_dir / "signal_id_mapping.json"
        html_path = html_path or args.out_dir / "signal_timing_map.html"
    if net_path is None or mapping_path is None:
        parser.error("provide --out-dir or both --net and --mapping")
    if html_path is None:
        html_path = mapping_path.with_name("signal_timing_map.html")

    result = render_html(args.osm, mapping_path, net_path, html_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
