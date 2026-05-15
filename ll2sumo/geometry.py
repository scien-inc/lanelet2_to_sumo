from __future__ import annotations

import math

from ll2sumo.model import Point3D


def distance_2d(a: Point3D, b: Point3D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def distance_3d(a: Point3D, b: Point3D) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def angle_diff_deg(lhs: float, rhs: float) -> float:
    diff = (lhs - rhs + 180.0) % 360.0 - 180.0
    return abs(diff)


def heading_deg(start: Point3D, end: Point3D) -> float:
    return math.degrees(math.atan2(end.y - start.y, end.x - start.x))


def polyline_length(points: tuple[Point3D, ...] | list[Point3D]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(distance_3d(points[index], points[index + 1]) for index in range(len(points) - 1))


def point_along_direction(origin: Point3D, direction_start: Point3D, direction_end: Point3D, distance_m: float) -> Point3D | None:
    direction_length = distance_2d(direction_start, direction_end)
    if direction_length <= 0.0:
        return None
    ratio = distance_m / direction_length
    return Point3D(
        x=origin.x + (direction_end.x - direction_start.x) * ratio,
        y=origin.y + (direction_end.y - direction_start.y) * ratio,
        z=origin.z,
    )


def first_nonzero_segment(points: tuple[Point3D, ...], min_length_m: float = 0.01) -> tuple[Point3D, Point3D] | None:
    for start, end in zip(points, points[1:]):
        if distance_2d(start, end) > min_length_m:
            return start, end
    return None


def last_nonzero_segment(points: tuple[Point3D, ...], min_length_m: float = 0.01) -> tuple[Point3D, Point3D] | None:
    for start, end in zip(reversed(points[:-1]), reversed(points[1:])):
        if distance_2d(start, end) > min_length_m:
            return start, end
    return None


def _cumulative_lengths(points: tuple[Point3D, ...] | list[Point3D]) -> list[float]:
    lengths = [0.0]
    for index in range(1, len(points)):
        lengths.append(lengths[-1] + distance_3d(points[index - 1], points[index]))
    return lengths


def _interpolate(a: Point3D, b: Point3D, ratio: float) -> Point3D:
    return Point3D(
        x=a.x + (b.x - a.x) * ratio,
        y=a.y + (b.y - a.y) * ratio,
        z=a.z + (b.z - a.z) * ratio,
    )


def interpolate_polyline(points: tuple[Point3D, ...] | list[Point3D], distance_m: float) -> Point3D:
    if len(points) == 1:
        return points[0]
    cumulative = _cumulative_lengths(points)
    total = cumulative[-1]
    if total <= 0.0:
        return points[0]
    target = min(max(distance_m, 0.0), total)
    for index in range(1, len(points)):
        if cumulative[index] < target:
            continue
        span = cumulative[index] - cumulative[index - 1]
        if span <= 0.0:
            return points[index]
        ratio = (target - cumulative[index - 1]) / span
        return _interpolate(points[index - 1], points[index], ratio)
    return points[-1]


def resample_polyline(points: tuple[Point3D, ...] | list[Point3D], sample_count: int) -> tuple[Point3D, ...]:
    if not points:
        return tuple()
    if len(points) == 1 or sample_count <= 1:
        return (points[0],)
    total = polyline_length(points)
    if total <= 0.0:
        return tuple(points[0] for _ in range(sample_count))
    return tuple(interpolate_polyline(points, total * index / (sample_count - 1)) for index in range(sample_count))


def parallel_polylines_compatible(
    lhs: tuple[Point3D, ...],
    rhs: tuple[Point3D, ...],
    *,
    max_length_ratio: float,
    max_mean_gap_m: float,
    max_sample_gap_m: float,
    max_segment_heading_diff_deg: float,
) -> bool:
    shorter_length, longer_length = sorted((polyline_length(lhs), polyline_length(rhs)))
    if shorter_length <= 0.0:
        return False
    if longer_length / shorter_length > max_length_ratio:
        return False

    sample_count = max(min(max(len(lhs), len(rhs)), 21), 5)
    lhs_samples = resample_polyline(lhs, sample_count)
    rhs_samples = resample_polyline(rhs, sample_count)
    gaps = [distance_2d(lhs_point, rhs_point) for lhs_point, rhs_point in zip(lhs_samples, rhs_samples)]
    if max(gaps) > max_sample_gap_m:
        return False
    if sum(gaps) / len(gaps) > max_mean_gap_m:
        return False

    for index in range(sample_count - 1):
        lhs_heading = heading_deg(lhs_samples[index], lhs_samples[index + 1])
        rhs_heading = heading_deg(rhs_samples[index], rhs_samples[index + 1])
        if angle_diff_deg(lhs_heading, rhs_heading) > max_segment_heading_diff_deg:
            return False

    return True


def orient_boundaries(
    left_boundary: tuple[Point3D, ...],
    right_boundary: tuple[Point3D, ...],
) -> tuple[tuple[Point3D, ...], tuple[Point3D, ...]]:
    same_score = distance_2d(left_boundary[0], right_boundary[0]) + distance_2d(left_boundary[-1], right_boundary[-1])
    flipped_score = distance_2d(left_boundary[0], right_boundary[-1]) + distance_2d(left_boundary[-1], right_boundary[0])
    if flipped_score < same_score:
        return left_boundary, tuple(reversed(right_boundary))
    return left_boundary, right_boundary


def orient_lanelet(
    left_node_ids: tuple[str, ...],
    right_node_ids: tuple[str, ...],
    left_boundary: tuple[Point3D, ...],
    right_boundary: tuple[Point3D, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[Point3D, ...], tuple[Point3D, ...]]:
    aligned_left_boundary, aligned_right_boundary = orient_boundaries(left_boundary, right_boundary)
    aligned_left_node_ids = left_node_ids
    aligned_right_node_ids = right_node_ids
    if aligned_right_boundary != right_boundary:
        aligned_right_node_ids = tuple(reversed(aligned_right_node_ids))

    candidates = (
        (
            aligned_left_node_ids,
            aligned_right_node_ids,
            aligned_left_boundary,
            aligned_right_boundary,
        ),
        (
            tuple(reversed(aligned_left_node_ids)),
            tuple(reversed(aligned_right_node_ids)),
            tuple(reversed(aligned_left_boundary)),
            tuple(reversed(aligned_right_boundary)),
        ),
    )

    return max(candidates, key=_lanelet_orientation_score)


def _lanelet_orientation_score(
    candidate: tuple[tuple[str, ...], tuple[str, ...], tuple[Point3D, ...], tuple[Point3D, ...]],
) -> float:
    _, _, left_boundary, right_boundary = candidate
    sample_count = max(len(left_boundary), len(right_boundary), 8)
    centerline = average_polylines(left_boundary, right_boundary, sample_count=sample_count)
    left_samples = resample_polyline(left_boundary, sample_count)
    right_samples = resample_polyline(right_boundary, sample_count)

    score = 0.0
    for index in range(sample_count):
        if index == 0:
            heading = heading_deg(centerline[0], centerline[1])
        elif index == sample_count - 1:
            heading = heading_deg(centerline[-2], centerline[-1])
        else:
            heading = heading_deg(centerline[index - 1], centerline[index + 1])

        score += signed_lateral_offset(left_samples[index], centerline[index], heading)
        score -= signed_lateral_offset(right_samples[index], centerline[index], heading)

    return score


def average_polylines(
    left_boundary: tuple[Point3D, ...],
    right_boundary: tuple[Point3D, ...],
    sample_count: int | None = None,
) -> tuple[Point3D, ...]:
    sample_count = sample_count or max(len(left_boundary), len(right_boundary), 2)
    left_samples = resample_polyline(left_boundary, sample_count)
    right_samples = resample_polyline(right_boundary, sample_count)
    return tuple(
        Point3D(
            x=(left_point.x + right_point.x) / 2.0,
            y=(left_point.y + right_point.y) / 2.0,
            z=(left_point.z + right_point.z) / 2.0,
        )
        for left_point, right_point in zip(left_samples, right_samples)
    )


def average_polyline_many(polylines: list[tuple[Point3D, ...]]) -> tuple[Point3D, ...]:
    if not polylines:
        return tuple()
    sample_count = max(max(len(polyline) for polyline in polylines), 2)
    samples = [resample_polyline(polyline, sample_count) for polyline in polylines]
    averaged: list[Point3D] = []
    for sample_index in range(sample_count):
        xs = [sample[sample_index].x for sample in samples]
        ys = [sample[sample_index].y for sample in samples]
        zs = [sample[sample_index].z for sample in samples]
        averaged.append(
            Point3D(
                x=sum(xs) / len(xs),
                y=sum(ys) / len(ys),
                z=sum(zs) / len(zs),
            )
        )
    return tuple(averaged)


def max_z_gap(lhs: tuple[Point3D, ...], rhs: tuple[Point3D, ...], sample_count: int = 5) -> float:
    lhs_samples = resample_polyline(lhs, sample_count)
    rhs_samples = resample_polyline(rhs, sample_count)
    return max(abs(lhs_point.z - rhs_point.z) for lhs_point, rhs_point in zip(lhs_samples, rhs_samples))


def midpoint(points: tuple[Point3D, ...]) -> Point3D:
    return interpolate_polyline(points, polyline_length(points) / 2.0)


def signed_lateral_offset(point: Point3D, origin: Point3D, heading: float) -> float:
    angle_rad = math.radians(heading)
    left_normal_x = -math.sin(angle_rad)
    left_normal_y = math.cos(angle_rad)
    return (point.x - origin.x) * left_normal_x + (point.y - origin.y) * left_normal_y
