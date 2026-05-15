from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float


@dataclass(frozen=True)
class GeoReference:
    proj_parameter: str
    utm_zone: int
    hemisphere: str
    local_to_projected_offset_x: float
    local_to_projected_offset_y: float
    sample_count: int
    max_error_m: float
    mean_error_m: float

    @property
    def net_offset_x(self) -> float:
        return -self.local_to_projected_offset_x

    @property
    def net_offset_y(self) -> float:
        return -self.local_to_projected_offset_y


@dataclass(frozen=True)
class Way:
    id: str
    node_ids: tuple[str, ...]
    tags: dict[str, str]


@dataclass(frozen=True)
class Lanelet:
    id: str
    subtype: str
    tags: dict[str, str]
    left_way_id: str
    right_way_id: str
    regulatory_ids: tuple[str, ...]
    left_node_ids: tuple[str, ...]
    right_node_ids: tuple[str, ...]
    left_boundary: tuple[Point3D, ...]
    right_boundary: tuple[Point3D, ...]
    centerline: tuple[Point3D, ...]
    start: Point3D
    end: Point3D
    avg_heading_deg: float
    length_m: float


@dataclass(frozen=True)
class RegulatoryElement:
    id: str
    subtype: str
    tags: dict[str, str]
    members_by_role: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class LaneletMap:
    nodes: dict[str, Point3D]
    ways: dict[str, Way]
    lanelets: dict[str, Lanelet]
    regulatory_elements: dict[str, RegulatoryElement] = field(default_factory=dict)
    node_geo: dict[str, GeoPoint] = field(default_factory=dict)
    geo_reference: GeoReference | None = None


@dataclass(frozen=True)
class BoundaryNeighbor:
    boundary_id: str
    left_lanelet_id: str
    right_lanelet_id: str
    start_gap_m: float
    end_gap_m: float
    heading_diff_deg: float
    z_gap_m: float
    bundle_eligible: bool


@dataclass(frozen=True)
class LaneChangeDecision:
    source_lanelet_id: str
    target_lanelet_id: str
    direction: str
    boundary_id: str
    allowed: bool
    reason: str
    source: str


@dataclass
class LaneChangeAnalysis:
    neighbors: list[BoundaryNeighbor] = field(default_factory=list)
    decisions: dict[tuple[str, str], LaneChangeDecision] = field(default_factory=dict)
    summary: dict[str, int] = field(default_factory=dict)
