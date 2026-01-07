import numpy as np
import matplotlib.pyplot as plt
from typing import List, Sequence, Optional, Union
import argparse
from pathlib import Path


def generate_gaussian_clusters(
    n_points: Union[int, Sequence[int]] = 1000,
    means: Optional[Sequence[Sequence[float]]] = None,
    covs: Optional[Sequence[np.ndarray]] = None,
    num_dims: int = 2,
    noise_scale: float = 0.0,
    spacing: float = 4.0,
    seed: Optional[int] = None,
    plot: bool = True,
):
    """Generate a mixture of multivariate Gaussian clusters with optional isotropic noise.

    Returns
    -------
    clusters : list[np.ndarray]
        List of arrays, each shaped (n_points_k, num_dims).
    """

    rng = np.random.default_rng(seed)

    # cluster size
    if np.isscalar(n_points):
        n_points = int(n_points)
        sizes = [n_points]
    else:
        sizes = list(map(int, n_points))

    num_clusters = len(sizes)

    # prepare means
    if means is None:
        means = [np.zeros(num_dims) + k * spacing for k in range(num_clusters)]
    if len(means) != num_clusters:
        raise ValueError("Length of 'means' must equal number of clusters.")
    means = [np.asarray(m, dtype=float) for m in means]

    # prepare covariances
    if covs is None:
        covs = [np.eye(num_dims)] * num_clusters
    if len(covs) != num_clusters:
        raise ValueError("Length of 'covs' must equal number of clusters.")
    covs = [np.asarray(c, dtype=float) for c in covs]

    clusters: List[np.ndarray] = []

    for size, mu, sigma in zip(sizes, means, covs):
        pts = rng.multivariate_normal(mean=mu, cov=sigma, size=size)
        if noise_scale > 0:
            pts += rng.normal(loc=0.0, scale=noise_scale, size=pts.shape)
        clusters.append(pts)

    # optional 2d plot
    if plot and num_dims == 2:
        plt.figure(figsize=(6, 5))
        for k, pts in enumerate(clusters):
            plt.scatter(pts[:, 0], pts[:, 1], s=8, label=f"Cluster {k}")
        plt.title(f"{num_clusters} 2D-Gauss-Cluster (noise σ={noise_scale})")
        plt.xlabel("x₀")
        plt.ylabel("x₁")
        plt.gca().set_aspect("equal", "box")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()
        plt.show()

    return clusters


def export_points_txt(
    points: Sequence[Sequence[float]] | np.ndarray,
    file_path: str | Path,
    *,
    fmt: str = "%.15g",
    delimiter: str = " ",
    header: str | None = None,
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
        description="Generate Gaussian clusters and export to a text file."
    )
    parser.add_argument(
        "--n_points",
        type=int,
        nargs="+",
        default=[400, 400],
        help="Number of points per cluster (list for multiple clusters).",
    )
    parser.add_argument(
        "--num_dims",
        type=int,
        default=3,
        help="Number of dimensions for the clusters.",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.1,
        help="Std dev of additional isotropic jitter added to points.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=4.0,
        help="Spacing between cluster means (used when means=None).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the clusters (works for num_dims==2).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="clusters.txt",
        help="Output file path for the clusters.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    means = [rng.uniform(-3, 3, size=args.num_dims) for _ in range(len(args.n_points))]

    covs = [np.eye(args.num_dims) for _ in range(len(args.n_points))]

    clusters = generate_gaussian_clusters(
        n_points=args.n_points,
        means=means,
        covs=covs,
        num_dims=args.num_dims,
        noise_scale=args.noise_scale,
        spacing=args.spacing,
        seed=args.seed,
        plot=args.plot,
    )

    export_points_txt(np.vstack(clusters), args.output_file)
    print("Clusters exported to " + args.output_file)
