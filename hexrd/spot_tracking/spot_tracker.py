from dataclasses import dataclass

import rtree

from .spot import Spot


@dataclass
class TrackedSpot(Spot):
    frame_index: int
    missing_count: int

    @staticmethod
    def from_spot(spot: Spot, frame_index: int):
        return TrackedSpot(
            spot.i,
            spot.j,
            spot.w,
            spot.bounding_box,
            spot.max,
            spot.sum,
            frame_index,
            0,
        )

    def update(self, spot: Spot, frame_index: int):
        self.i = spot.i
        self.j = spot.j
        self.w = spot.w
        self.bounding_box = spot.bounding_box
        self.max = spot.max
        self.sum = spot.sum
        self.frame_index = frame_index
        self.missing_count = 0


class SpotTracker:
    def __init__(
        self,
        overlap_threshold: float = 0.001,
        missing_frame_threshold: int = 1,
    ):
        self.overlap_threshold = overlap_threshold
        self.missing_frame_threshold = missing_frame_threshold

        self.current_spots: dict[int, TrackedSpot] = {}

        self.next_id = 0

        self.spot_index = rtree.index.Index()

    def overlap(self, a: Spot, b: Spot) -> float:
        '''
        Returns the fraction of overlap between two rectangles given by their center and width
        '''

        i1, j1, w1 = a.i, a.j, a.w
        i2, j2, w2 = b.i, b.j, b.w

        lx = max(i1 - w1 / 2, i2 - w2 / 2)
        ly = max(j1 - w1 / 2, j2 - w2 / 2)
        hx = min(i1 + w1 / 2, i2 + w2 / 2)
        hy = min(j1 + w1 / 2, j2 + w2 / 2)

        dx = max(0, hx - lx)
        dy = max(0, hy - ly)

        return dx * dy / (w1 * w1 + w2 * w2 - dx * dy)

    def track_spots(
        self, spots: list[Spot], frame_index: int
    ) -> list[tuple[int, TrackedSpot]]:
        '''
        Compares the spots to the currently tracked spots and updates the current spots with the new ones.

        Returns the currently tracked spots and their ids.
        '''

        for spot in spots:
            hits = list(self.spot_index.intersection(spot.bounding_box))

            for hit in hits:
                existing_spot = self.current_spots[hit]

                if existing_spot.missing_count == 0:
                    continue

                if self.overlap(spot, existing_spot) > self.overlap_threshold:
                    self.spot_index.delete(hit, existing_spot.bounding_box)
                    existing_spot.update(spot, frame_index)
                    self.spot_index.insert(hit, existing_spot.bounding_box)
                    break
            else:
                self.current_spots[self.next_id] = TrackedSpot.from_spot(
                    spot, frame_index
                )
                self.spot_index.insert(self.next_id, spot.bounding_box)
                self.next_id += 1

        for i, spot in list(self.current_spots.items()):
            spot.missing_count += 1
            if spot.missing_count > self.missing_frame_threshold:
                del self.current_spots[i]
                self.spot_index.delete(i, spot.bounding_box)

        return list(self.current_spots.items())
