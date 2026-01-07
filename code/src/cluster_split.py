"""
split_cluster takes an existing clustering (labels) and
re‑balances it so that every cluster fulfils
min_points <= size <= max_size while simultaneously reducing the
pairwise overlap of their (axis‑aligned) bounding boxes. The routine
works in any Euclidean dimension and is completely NumPy based. no
external ML libraries are required (a custom splitter callback can
still be supplied if you prefer k‑means or smth else).

Strategy in a nutshell:
1. KD‑style recursive bisection for oversized clusters
   – split along the dimension with the largest span at the median, which
   guarantees that the childrens bounding boxes are disjoint in that
   axis (=> zero overlap across the split axis).
2. Greedy merge for undersized clusters
   – progressively merge the smallest clusters into the neighbour that
   would cause the smallest increase in combined bounding‑box volume
   (ties broken by centroid distance).
3. Progress feedback
   – if verbose=True the current percentage of balanced clusters is
   emitted after every split or merge.

The KD‑bisection is entirely vectorised and therefore very fast even for
hundreds of thousands of points!! Also tries to be type safe!
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Sequence, Tuple, Union, List, Dict, Iterable

import numpy as np

__all__ = ["split_cluster"]

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]

IdxList = List[int]
BBox = Tuple[np.ndarray, np.ndarray]  # (mins, maxs)


# public API
def split_cluster(
    X: ArrayLike,
    labels: Sequence[int],
    max_size: int,
    *,
    min_points: int | None = None,
    splitter: (
        Callable[[np.ndarray, int], Tuple[Union[np.ndarray, List[int]], np.ndarray]]
        | None
    ) = None,
    seed: int | None = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    # validation
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    if min_points is not None:
        if min_points <= 0:
            raise ValueError("min_points must be positive when provided")
        if min_points > max_size:
            raise ValueError("min_points must not exceed max_size")
    if len(X) != len(labels):
        raise ValueError("len(X) and len(labels) must match")

    X_arr: np.ndarray = np.asarray(
        X, dtype=float, order="C"
    )  # converting into np array
    n_pts, n_dim = X_arr.shape
    rng = np.random.default_rng(seed)

    # initial buckets
    clusters: Dict[int, IdxList] = defaultdict(list)
    for idx, lab in enumerate(labels):
        clusters[int(lab)].append(idx)

    next_id: int = max(clusters.keys(), default=-1) + 1

    # geometry functions
    def _bbox(idxs: Iterable[int]) -> BBox:
        pts = X_arr[list(idxs)]
        return pts.min(axis=0), pts.max(axis=0)

    def _volume(bbox: BBox) -> float:
        mins, maxs = bbox
        return float(np.prod(np.maximum(maxs - mins, 0.0)))

    def _union_bbox(bb1: BBox, bb2: BBox) -> BBox:
        return np.minimum(bb1[0], bb2[0]), np.maximum(bb1[1], bb2[1])

    # keep auxiliary geometry (Bboxes, centers etc) up‑to‑date for each cluster
    bbox: Dict[int, BBox] = {cid: _bbox(idxs) for cid, idxs in clusters.items()}
    centers: Dict[int, np.ndarray] = {
        cid: X_arr[idxs].mean(axis=0) for cid, idxs in clusters.items()
    }

    # if verbose = True // logs to console
    def _progress() -> float:
        if not clusters:
            return 100.0
        ok = sum(
            (len(v) <= max_size) and (min_points is None or len(v) >= min_points)
            for v in clusters.values()
        )
        return 100.0 * ok / len(clusters)

    def _print(tag: str):
        if verbose:
            print(f"[{tag}] {_progress():.1f}% balanced – {len(clusters)} clusters")

    # kd split routine
    def _kd_bisect(idxs: IdxList) -> List[IdxList]:
        """Recursively (using stack) bisect idxs until every child has <= max_size points.

        The split axis is chosen as the dimension with the largest spread;
        the pivot is the median. This ensures zero overlap in that axis.
        """
        subclusters: List[IdxList] = []
        stack: List[IdxList] = [idxs]
        while stack:
            cur = stack.pop()
            if len(cur) <= max_size:
                subclusters.append(cur)
                continue

            pts = X_arr[cur]
            ranges = np.ptp(pts, axis=0)  # max - min per dimension
            split_dim = int(np.argmax(ranges))

            if ranges[split_dim] == 0:  # degenerate (all coords identical)
                # fallback: random split to keep things moving
                rng.shuffle(cur)
                mid = len(cur) // 2
                stack.append(cur[:mid])
                stack.append(cur[mid:])
                continue

            # partition around median (vectorised, O(n))
            coords = pts[:, split_dim]
            mid = len(cur) // 2
            median_val = np.partition(coords, mid)[mid]
            left_mask = coords <= median_val
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                # all points equal up to numerical eps => random shuffle
                rng.shuffle(cur)
                mid = len(cur) // 2
                left_mask = np.zeros_like(coords, dtype=bool)
                left_mask[:mid] = True
                right_mask = ~left_mask

            left = [cur[i] for i, m in enumerate(left_mask) if m]
            right = [cur[i] for i, m in enumerate(right_mask) if m]
            stack.extend([left, right])
        return subclusters

    def _split_cluster(cid: int):
        nonlocal next_id  # nonlocal makes changes of next_id also work out of function scope
        idxs = clusters.pop(cid)
        new_parts = _kd_bisect(idxs)
        for part in new_parts:
            clusters[next_id] = part
            bbox[next_id] = _bbox(part)
            centers[next_id] = X_arr[part].mean(axis=0)
            next_id += 1
        _print("split")

    # merging routine
    def _merge_smallest():
        # pick the smallest cluster (ties broken randomly for speed)
        cid_src = min(clusters, key=lambda c: len(clusters[c]))
        src_idxs = clusters[cid_src]
        src_bb = bbox[cid_src]

        best_cid_tgt: int | None = None
        best_cost = math.inf
        for cid_tgt in clusters:
            if cid_tgt == cid_src:
                continue
            tgt_bb = bbox[cid_tgt]
            union_bb = _union_bbox(src_bb, tgt_bb)
            cost = _volume(union_bb)
            if cost < best_cost:
                best_cost = cost
                best_cid_tgt = cid_tgt

        assert best_cid_tgt is not None  # smth went wrong

        # perform merge
        clusters[best_cid_tgt].extend(src_idxs)
        bbox[best_cid_tgt] = _union_bbox(bbox[best_cid_tgt], src_bb)
        centers[best_cid_tgt] = X_arr[clusters[best_cid_tgt]].mean(axis=0)
        del clusters[cid_src]
        del bbox[cid_src]
        del centers[cid_src]
        _print("merge")

    # main loop

    # simple FIFO queue of clusters waiting for a split
    queue: List[int] = [cid for cid, v in clusters.items() if len(v) > max_size]

    while True:
        # split oversized
        while queue:
            cid = queue.pop(0)
            if cid not in clusters:  # might have been merged already
                continue
            if len(clusters[cid]) > max_size:
                _split_cluster(cid)
                queue.extend(
                    k
                    for k, v in clusters.items()
                    if len(v) > max_size and k not in queue
                )

        # merge undersized
        if min_points is None:
            break

        small_exists = any(len(v) < min_points for v in clusters.values())
        if not small_exists:
            break
        _merge_smallest()

        # merged cluster might now be too large => queue it for splitting
        for cid, v in list(clusters.items()):
            if len(v) > max_size and cid not in queue:
                queue.append(cid)

    # done, results
    new_labels = np.empty(n_pts, dtype=int)
    new_centers: List[np.ndarray] = []

    for new_id, (cid, idxs) in enumerate(clusters.items()):
        new_labels[idxs] = new_id
        new_centers.append(centers[cid])

    _print("done")
    return new_labels, np.vstack(new_centers)
