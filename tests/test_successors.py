from __future__ import annotations

import unittest

from ll2sumo.convert import _successor_candidates
from ll2sumo.model import Lanelet, Point3D


def make_lanelet(
    lanelet_id: str,
    *,
    left_node_ids: tuple[str, str],
    right_node_ids: tuple[str, str],
    start: tuple[float, float, float] = (0.0, 0.0, 0.0),
    end: tuple[float, float, float] = (1.0, 0.0, 0.0),
    tags: dict[str, str] | None = None,
) -> Lanelet:
    start_point = Point3D(*start)
    end_point = Point3D(*end)
    centerline = (start_point, end_point)
    lanelet_tags = {"type": "lanelet", "subtype": "road", "speed_limit": "60", "location": "urban", "one_way": "yes"}
    if tags:
        lanelet_tags.update(tags)
    return Lanelet(
        id=lanelet_id,
        subtype="road",
        tags=lanelet_tags,
        left_way_id=f"{lanelet_id}_left",
        right_way_id=f"{lanelet_id}_right",
        regulatory_ids=tuple(),
        left_node_ids=left_node_ids,
        right_node_ids=right_node_ids,
        left_boundary=centerline,
        right_boundary=centerline,
        centerline=centerline,
        start=start_point,
        end=end_point,
        avg_heading_deg=0.0,
        length_m=1.0,
    )


def make_map(*lanelets: Lanelet) -> dict[str, Lanelet]:
    return {lanelet.id: lanelet for lanelet in lanelets}


class SuccessorCandidateTest(unittest.TestCase):
    def test_shared_endpoint_nodes_create_successor(self) -> None:
        source = make_lanelet(
            "source",
            left_node_ids=("a0", "a1"),
            right_node_ids=("b0", "b1"),
        )
        target = make_lanelet(
            "target",
            left_node_ids=("a1", "a2"),
            right_node_ids=("b1", "b2"),
        )

        successors = _successor_candidates(make_map(source, target))

        self.assertEqual(successors["source"], ["target"])

    def test_geometrically_close_without_shared_nodes_is_not_a_successor(self) -> None:
        source = make_lanelet(
            "source",
            left_node_ids=("a0", "a1"),
            right_node_ids=("b0", "b1"),
            start=(0.0, 0.0, 0.0),
            end=(10.0, 0.0, 0.0),
        )
        target = make_lanelet(
            "target",
            left_node_ids=("c0", "c1"),
            right_node_ids=("d0", "d1"),
            start=(10.0, 0.0, 0.0),
            end=(20.0, 0.1, 0.0),
        )

        successors = _successor_candidates(make_map(source, target))

        self.assertNotIn("source", successors)


if __name__ == "__main__":
    unittest.main()
