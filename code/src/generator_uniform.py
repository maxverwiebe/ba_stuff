import numpy as np
from typing import List, Tuple, Union, Sequence, Optional
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def generate_uniform_clusters(
    n_points: Union[int, List[int]] = 500,
    num_clusters: int = 2,
    num_dims: int = 2,
    spacing: Union[float, Sequence[float], str] = "auto",
    ranges: Optional[Sequence[Tuple[float, float]]] = None,
    gap_factor: float = 0.1,
    seed: Optional[int] = None,
    plot: bool = True,
):
    """Generate one or more independent, axis‑aligned uniform
    point clouds in arbitrary dimensions.

    Returns
    -------
    clusters : list[np.ndarray]
        List of arrays each shaped (n_points, num_dims)
    """
    rng = np.random.default_rng(seed)

    if isinstance(n_points, int):
        points_per_cluster = [n_points] * num_clusters
        actual_num_clusters = num_clusters
    else:
        points_per_cluster = list(n_points)
        actual_num_clusters = len(points_per_cluster)
        print(
            f"Inferred {actual_num_clusters} clusters from n_points list: {points_per_cluster}"
        )

    if ranges is None:
        ranges = [(0.0, 1.0) for _ in range(num_dims)]
    if len(ranges) != num_dims:
        raise ValueError("Length of 'ranges' must match 'num_dims'.")

    mins = np.array([lo for lo, _ in ranges], dtype=float)
    maxs = np.array([hi for _, hi in ranges], dtype=float)
    side_lengths = maxs - mins

    if spacing == "auto":
        cluster_width = side_lengths[0]
        gap = cluster_width * gap_factor
        auto_spacing = cluster_width + gap

        shift_vec = np.zeros(num_dims)
        shift_vec[0] = auto_spacing

        print(
            f"Auto spacing: {auto_spacing:.3f} (cluster width: {cluster_width:.3f}, gap: {gap:.3f})"
        )
    else:
        shift_vec = np.zeros(num_dims)
        if np.isscalar(spacing):
            shift_vec[0] = float(spacing)
        else:
            # copy into shift vector (ignore remainder if longer than num_dims)
            spacing = list(spacing)
            shift_vec[: len(spacing)] = spacing[:num_dims]

    clusters: List[np.ndarray] = []

    for k in range(actual_num_clusters):
        # sample base cube uniformly and apply k‑th shift
        n_pts = points_per_cluster[k]
        pts = rng.random((n_pts, num_dims)) * side_lengths + mins
        pts += k * shift_vec
        clusters.append(pts)

    if plot and num_dims == 2:
        if True:
            plt.figure(figsize=(10, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, actual_num_clusters))

            for i, pts in enumerate(clusters):
                plt.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=10,
                    alpha=0.6,
                    color=colors[i],
                    label=f"Cluster {i+1} (n={len(pts)})",
                )

            for i, pts in enumerate(clusters):
                x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
                y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

                from matplotlib.patches import Rectangle

                rect = Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=2,
                    edgecolor=colors[i],
                    facecolor="none",
                    alpha=0.8,
                    linestyle="--",
                )
                plt.gca().add_patch(rect)

            plt.title(f"{actual_num_clusters} gleichverteilte Cluster (d={num_dims})")
            plt.xlabel("x₀")
            plt.ylabel("x₁")
            plt.legend()
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
            plt.tight_layout()
            plt.show()

    return clusters


def export_points_txt(
    points: Union[Sequence[Sequence[float]], np.ndarray],
    file_path: Union[str, Path],
    *,
    fmt: str = "%.15g",
    delimiter: str = " ",
    header: Optional[str] = None,
) -> None:
    points = np.asarray(points)
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        file_path,
        points,
        fmt=fmt,
        delimiter=delimiter,
        header=header or "",
        comments="",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate uniform clusters and export to a text file."
    )

    parser.add_argument(
        "--n_points",
        type=int,
        nargs="+",
        default=[1000],
        help="Number of points per cluster.",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=2,
        help="Number of uniform clusters to create.",
    )
    parser.add_argument(
        "--num_dims",
        type=int,
        default=2,
        help="Number of dimensions for the clusters",
    )

    parser.add_argument(
        "--spacing",
        type=str,
        default="auto",
        help="Spacing between clusters: 'auto' or numeric value",
    )
    parser.add_argument(
        "--gap_factor",
        type=float,
        default=0.1,
        help="Gap between clusters as fraction of cluster size",
    )

    parser.add_argument(
        "--range_min",
        type=float,
        default=0.0,
        help="Minimum value for cluster ranges",
    )
    parser.add_argument(
        "--range_max",
        type=float,
        default=10.0,
        help="Maximum value for cluster ranges",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument("--plot", action="store_true", help="Show visualization plot")

    parser.add_argument(
        "--output_file",
        type=str,
        default="uniform_clusters.txt",
        help="Output file path for the clusters",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for the generated files",
    )

    args = parser.parse_args()

    if args.spacing.lower() == "auto":
        spacing = "auto"
    else:
        try:
            spacing = float(args.spacing)
        except ValueError:
            print(f"Warning: Invalid spacing value '{args.spacing}', using 'auto'")
            spacing = "auto"

    if len(args.n_points) == 1 and args.num_clusters > 1:
        # single value provided, replicate for all clusters
        n_points = args.n_points[0]
        num_clusters = args.num_clusters
        print(f"Using {n_points} points for each of {num_clusters} clusters")
    else:
        # multiple values provided, use as is lol
        n_points = args.n_points
        num_clusters = len(args.n_points)
        print(
            f"Using variable cluster sizes: {n_points} (total {num_clusters} clusters)"
        )

    # create ranges for all dimensions
    ranges = [(args.range_min, args.range_max) for _ in range(args.num_dims)]

    # Generate clusters
    clusters = generate_uniform_clusters(
        n_points=n_points,
        num_clusters=num_clusters,
        num_dims=args.num_dims,
        spacing=spacing,
        ranges=ranges,
        gap_factor=args.gap_factor,
        seed=args.seed,
        plot=args.plot,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_file == "uniform_clusters.txt":
        output_file = output_dir / "uniform_clusters.txt"
    else:
        output_file = output_dir / args.output_file

    all_points = np.vstack(clusters)
    export_points_txt(all_points, output_file)
