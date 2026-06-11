from __future__ import annotations

import unittest

from ll2sumo.geometry import orient_lanelet, parallel_polylines_compatible
from ll2sumo.model import Point3D


class OrientLaneletTest(unittest.TestCase):
    def test_reverses_both_bounds_when_left_boundary_is_on_wrong_side(self) -> None:
        left_node_ids = ("l0", "l1")
        right_node_ids = ("r0", "r1")
        left_boundary = (
            Point3D(10.0, 1.0, 0.0),
            Point3D(0.0, 1.0, 0.0),
        )
        right_boundary = (
            Point3D(0.0, -1.0, 0.0),
            Point3D(10.0, -1.0, 0.0),
        )

        oriented_left_node_ids, oriented_right_node_ids, oriented_left_boundary, oriented_right_boundary = orient_lanelet(
            left_node_ids,
            right_node_ids,
            left_boundary,
            right_boundary,
        )

        self.assertEqual(oriented_left_node_ids, ("l1", "l0"))
        self.assertEqual(oriented_right_node_ids, ("r0", "r1"))
        self.assertEqual(oriented_left_boundary[0], Point3D(0.0, 1.0, 0.0))
        self.assertEqual(oriented_right_boundary[0], Point3D(0.0, -1.0, 0.0))

    def test_keeps_curved_lanelet_orientation_when_local_left_stays_left(self) -> None:
        left_node_ids = tuple(f"l{index}" for index in range(8))
        right_node_ids = tuple(f"r{index}" for index in range(7))
        left_boundary = (
            Point3D(87508.34, 44532.93, 38.98),
            Point3D(87375.46, 44553.75, 33.18),
            Point3D(87274.70, 44464.44, 28.31),
            Point3D(87311.67, 44343.43, 23.99),
            Point3D(87426.65, 44318.43, 20.04),
            Point3D(87511.06, 44408.40, 15.97),
            Point3D(87522.32, 44548.02, 13.81),
            Point3D(87522.40, 44558.30, 13.79),
        )
        right_boundary = (
            Point3D(87509.19, 44535.90, 39.05),
            Point3D(87379.55, 44557.94, 33.71),
            Point3D(87275.62, 44479.65, 29.23),
            Point3D(87301.57, 44347.61, 24.73),
            Point3D(87421.51, 44312.85, 20.64),
            Point3D(87515.78, 44411.01, 16.10),
            Point3D(87525.48, 44558.21, 13.88),
        )

        oriented_left_node_ids, oriented_right_node_ids, oriented_left_boundary, oriented_right_boundary = orient_lanelet(
            left_node_ids,
            right_node_ids,
            left_boundary,
            right_boundary,
        )

        self.assertEqual(oriented_left_node_ids, left_node_ids)
        self.assertEqual(oriented_right_node_ids, right_node_ids)
        self.assertEqual(oriented_left_boundary[0], left_boundary[0])
        self.assertEqual(oriented_right_boundary[0], right_boundary[0])


class ParallelPolylinesCompatibleTest(unittest.TestCase):
    def test_accepts_nearby_parallel_lines(self) -> None:
        self.assertTrue(
            parallel_polylines_compatible(
                (Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0)),
                (Point3D(0.0, 4.0, 0.0), Point3D(10.0, 4.0, 0.0)),
                max_length_ratio=1.35,
                max_mean_gap_m=8.0,
                max_sample_gap_m=12.0,
                max_segment_heading_diff_deg=45.0,
            )
        )

    def test_rejects_excessive_length_ratio(self) -> None:
        self.assertFalse(
            parallel_polylines_compatible(
                (Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0)),
                (Point3D(0.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0)),
                max_length_ratio=1.35,
                max_mean_gap_m=100.0,
                max_sample_gap_m=100.0,
                max_segment_heading_diff_deg=180.0,
            )
        )

    def test_rejects_large_sample_gap(self) -> None:
        self.assertFalse(
            parallel_polylines_compatible(
                (Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0)),
                (Point3D(0.0, 13.0, 0.0), Point3D(10.0, 13.0, 0.0)),
                max_length_ratio=1.35,
                max_mean_gap_m=100.0,
                max_sample_gap_m=12.0,
                max_segment_heading_diff_deg=45.0,
            )
        )

    def test_rejects_segment_heading_mismatch(self) -> None:
        self.assertFalse(
            parallel_polylines_compatible(
                (Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0)),
                (Point3D(0.0, 0.0, 0.0), Point3D(0.0, 10.0, 0.0)),
                max_length_ratio=1.35,
                max_mean_gap_m=100.0,
                max_sample_gap_m=100.0,
                max_segment_heading_diff_deg=45.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
