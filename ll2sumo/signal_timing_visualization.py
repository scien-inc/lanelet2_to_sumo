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
    reverse_mapping = mapping.get("lanelet_signal_to_sumo_links", {})
    if not isinstance(reverse_mapping, dict):
        raise ValueError("signal_id_mapping.json does not contain lanelet_signal_to_sumo_links")

    used_way_ids = {str(way_id) for way_id in reverse_mapping}
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
    for way_id, links in sorted(reverse_mapping.items(), key=lambda item: _sort_key(str(item[0]))):
        if not isinstance(links, list):
            continue
        points = signal_way_points.get(str(way_id))
        if not points:
            continue
        midpoint = _polyline_midpoint(points)
        if midpoint is None:
            continue
        typed_links = [link for link in links if isinstance(link, dict)]
        reg_ids = sorted(
            {str(reg_id) for link in typed_links for reg_id in link.get("lanelet_regulatory_element_ids", [])},
            key=_sort_key,
        )
        intersections = sorted(
            {str(link["intersection_area_id"]) for link in typed_links if link.get("intersection_area_id")},
            key=_sort_key,
        )
        phase_groups = _build_signal_phase_groups(typed_links, tl_logics)
        if not phase_groups:
            warnings.append(f"No usable SUMO phase group for Lanelet2 traffic_light way: {way_id}")
            continue
        signals.append(
            {
                "wayId": str(way_id),
                "position": _screen_point(midpoint, max_y),
                "line": [_screen_point(point, max_y) for point in points],
                "regulatoryElementIds": reg_ids,
                "intersectionAreaIds": intersections,
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
            "tlsCount": len({group["tlsId"] for signal in signals for group in signal["phaseGroups"]}),
            "roadPathCount": len(road_paths),
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
    .signal {{
      cursor: pointer;
      stroke: #1f2328;
      stroke-width: 1.8;
      vector-effect: non-scaling-stroke;
    }}
    .signal.selected {{
      stroke-width: 4.0;
      stroke: #111827;
    }}
    .state-red {{ fill: var(--red); }}
    .state-yellow {{ fill: var(--yellow); }}
    .state-green {{ fill: var(--green); }}
    .state-off {{ fill: var(--off); }}
    .state-unknown {{ fill: var(--unknown); }}
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
        </div>
        <p class="hint">Signals are placed from Lanelet2 <code>traffic_light/refers</code> way geometry. Colors are computed from SUMO <code>tlLogic id + linkIndex</code> via the sync-safe mapping.</p>
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
    const priority = {{off: 0, green: 1, yellow: 2, red: 3, unknown: 4}};
    const cssClass = {{red: "state-red", yellow: "state-yellow", green: "state-green", off: "state-off", unknown: "state-unknown"}};
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
      const states = signal.phaseGroups.map(group => stateAt(group, t));
      return states.reduce((best, state) => priority[state] > priority[best] ? state : best, "off");
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

      const signalGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      for (const signal of payload.signals) {{
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("class", "signal state-unknown");
        circle.setAttribute("data-way-id", signal.wayId);
        circle.setAttribute("cx", signal.position.x);
        circle.setAttribute("cy", signal.position.y);
        circle.setAttribute("r", 7);
        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        title.textContent = `Lanelet2 signal way ${{signal.wayId}}`;
        circle.appendChild(title);
        circle.addEventListener("click", () => {{
          selectedWayId = signal.wayId;
          update(simTime);
        }});
        signalGroup.appendChild(circle);
      }}
      svg.appendChild(signalGroup);
    }}

    function setClass(element, state, selected) {{
      element.setAttribute("class", `signal ${{cssClass[state] || cssClass.unknown}}${{selected ? " selected" : ""}}`);
    }}

    function renderSummary() {{
      const meta = payload.meta;
      const s = meta.summary || {{}};
      const rows = [
        ["Lanelet2 signal ways", meta.signalCount],
        ["SUMO TLS groups", meta.tlsCount],
        ["road centerlines", meta.roadPathCount],
        ["mixed sync count", s.mixed_lanelet_signal_phase_count ?? "-"],
        ["diagnostic mixed count", s.diagnostic_mixed_lanelet_signal_phase_count ?? "-"],
        ["sync link count", s.sync_eligible_sumo_link_count ?? "-"],
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
        <div class="stat"><span>sync links</span><strong>${{escapeText(signal.linkCount)}}</strong></div>
        <h3>SUMO TLS</h3>
        ${{signal.phaseGroups.map(group => `
          <div>
            <code>${{escapeText(group.tlsId)}}</code>
            <div class="small">cycle ${{escapeText(group.cycle)}}s / linkIndex ${{escapeText(group.linkIndices.join(", "))}}</div>
            ${{renderPhaseBar(group)}}
          </div>
        `).join("")}}
      `;
    }}

    function update(t) {{
      const circles = svg.querySelectorAll(".signal");
      let selectedSignal = null;
      let selectedState = "unknown";
      for (const circle of circles) {{
        const wayId = circle.getAttribute("data-way-id");
        const signal = payload.signals.find(item => item.wayId === wayId);
        const state = signalState(signal, t);
        const selected = wayId === selectedWayId;
        setClass(circle, state, selected);
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
