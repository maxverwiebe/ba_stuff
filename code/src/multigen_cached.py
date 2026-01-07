from tqdm import tqdm
import itertools
import os
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
import sys
import logging
import subprocess
import shutil

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)
logger = logging.getLogger()


def run_command(command, case_name=None):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        prefix = f"[{case_name}] " if case_name else ""
        if result.stdout:
            logger.info(f"{prefix}STDOUT:\n{result.stdout.strip()}")
        if result.stderr:
            logger.warning(f"{prefix}STDERR:\n{result.stderr.strip()}")
    except Exception as e:
        logger.error(f"{prefix}Command failed: {command}\nException: {str(e)}")


# data class
class TestCase:
    def __init__(
        self,
        name,
        num_clusters,
        num_points,
        distribution="gaussian",
        noise_scale=0.0,
        num_dims=2,
        num_points_is_total=False,
        rtree_variant=0,
        min_max_points_split=None,
    ):
        self.name = name
        self.num_clusters = num_clusters
        self.num_points = num_points
        self.distribution = distribution
        self.noise_scale = noise_scale
        self.num_dims = num_dims
        self.num_points_is_total = num_points_is_total
        self.rtree_variant = rtree_variant
        self.min_max_points_split = min_max_points_split


# generates a combinatoric of all possible test cases
def generate_test_cases(
    num_clusters_list,
    num_points_list,
    distribution_list,
    noise_scale_list,
    num_dims_list,
    rtree_variant_list,
    min_max_points_split_list,
):
    test_cases = {}
    case_counter = 1
    combinations = itertools.product(
        num_clusters_list,
        num_points_list,
        distribution_list,
        num_dims_list,
        rtree_variant_list,
        min_max_points_split_list,
    )
    for (
        num_clusters,
        num_points,
        distribution,
        num_dims,
        rtree_variant,
        min_max_points,
    ) in combinations:
        current_noise_scales = [0.0] if distribution == "uniform" else noise_scale_list
        for noise_scale in current_noise_scales:
            case_name = f"case{case_counter}"
            test_cases[case_name] = TestCase(
                name=case_name,
                num_clusters=num_clusters,
                num_points=num_points,
                distribution=distribution,
                noise_scale=noise_scale,
                num_dims=num_dims,
                num_points_is_total=True,
                rtree_variant=rtree_variant,
                min_max_points_split=min_max_points,
            )
            case_counter += 1
    return test_cases


# just a mapping from integer -> string
RTREE_VARIANTS = ["Linear", "Quadratic", "RStar"]

# CONFIG for the test cases
num_clusters_options = [1, 2, 5]
num_points_options = [5000, 10000, 20000, 50000, 100000, 200000]
distribution_options = ["gaussian", "uniform"]
noise_scale_options = [0.1, 0.3, 0.5, 0.7]
num_dims_options = [2, 5, 10, 20]
rtree_variants = [0, 1, 2]

# relative to max num_points
min_max_points_split = [(0.005, 0.01)]


TEST_CASES = generate_test_cases(
    num_clusters_options,
    num_points_options,
    distribution_options,
    noise_scale_options,
    num_dims_options,
    rtree_variants,
    min_max_points_split,
)

# creates a dir to store test cases info
# absoluetly not the best solution performance-wise
# but yeah, fck it - it works
if os.path.exists("test_cases"):
    shutil.rmtree("test_cases")
os.makedirs("test_cases")

CACHE_DIR = "distribution_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# STEP 1: Generate the datasets for each testcases
for case_name, test_case in tqdm(TEST_CASES.items(), desc="Generating datasets"):
    case_dir = os.path.join("test_cases", test_case.name)
    os.makedirs(case_dir, exist_ok=True)

    # calculate points per cluster for this case
    points_per_cluster = test_case.num_points // test_case.num_clusters

    # write info file for the test case into its dir
    info_file_path = os.path.join(case_dir, "info.txt")
    with open(info_file_path, "w") as f:
        f.write(f"Test Case: {test_case.name}\n")
        f.write(f"Number of Clusters: {test_case.num_clusters}\n")
        f.write(f"Number of Points per Cluster: {points_per_cluster}\n")
        f.write(f"Distribution: {test_case.distribution}\n")
        f.write(f"Noise Scale: {test_case.noise_scale}\n")
        f.write(f"Number of Dimensions: {test_case.num_dims}\n")
        f.write(f"Rtree Variant: {RTREE_VARIANTS[test_case.rtree_variant]}\n")
        f.write(f"Min-Max Points (Relative): {test_case.min_max_points_split}\n")

        min_abs = test_case.min_max_points_split[0] * points_per_cluster
        max_abs = test_case.min_max_points_split[1] * points_per_cluster
        f.write(f"Min-Max Points (Absolute): {min_abs} - {max_abs}\n")

    # creates the data distribution and caches it in /distribution_cache
    cache_filename = (
        f"c{test_case.num_clusters}_p{points_per_cluster}_d-{test_case.distribution}_"
        f"n{test_case.noise_scale}_dim{test_case.num_dims}.txt"
    )
    cached_file_path = os.path.join(CACHE_DIR, cache_filename)
    target_points_file = os.path.join(case_dir, "points.txt")

    if os.path.exists(cached_file_path):
        # cache hit
        logger.info(f"[{case_name}] Cache HIT. Copying from {cached_file_path}")
        shutil.copy(cached_file_path, target_points_file)
    else:
        # cache miss
        logger.info(
            f"[{case_name}] Cache MISS. Generating new dataset: {cached_file_path}"
        )
        n_points_arg = " ".join([str(points_per_cluster)] * test_case.num_clusters)

        # weird solution I know
        if test_case.distribution == "gaussian":
            cmd = (
                f"python3.11 code/src/generator_gaussian.py --n_points {n_points_arg} "
                f"--num_dims {test_case.num_dims} --noise_scale {test_case.noise_scale} --output_file {cached_file_path}"
            )
        else:  # uniform
            cmd = (
                f"python3.11 code/src/generator_uniform.py --n_points {n_points_arg} "
                f"--num_dims {test_case.num_dims} --spacing 0 --output_file {cached_file_path}"
            )

        run_command(cmd, case_name=case_name)

        if os.path.exists(cached_file_path):
            shutil.copy(cached_file_path, target_points_file)
        else:
            logger.error(
                f"[{case_name}] Failed to generate cached file: {cached_file_path}"
            )


# generates the queries, we wanna use on the R-Tree of a test case
def generate_queries(points_file, num_queries=1000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    points = pd.read_csv(points_file, header=None, sep="\\s+").values
    indices = np.random.choice(points.shape[0], num_queries, replace=False)
    return points[indices]


for case_name in tqdm(TEST_CASES.keys(), desc="Generating queries"):
    case_dir = os.path.join("test_cases", case_name)
    points_file = os.path.join(case_dir, "points.txt")
    queries_file = os.path.join(case_dir, "queries.txt")
    if not os.path.exists(points_file):
        logger.warning(
            f"[{case_name}] Points file not found, skipping query generation."
        )
        continue
    queries = generate_queries(points_file)
    np.savetxt(queries_file, queries, delimiter=" ")


# runs the k-Means clustering for each test-case
results = {}
for case_name, test_case in tqdm(TEST_CASES.items(), desc="Running KMeans"):
    points_file = os.path.join("test_cases", case_name, "points.txt")
    if not os.path.exists(points_file):
        logger.warning(f"[{case_name}] Points file not found, skipping KMeans.")
        continue
    data = pd.read_csv(points_file, header=None, sep="\\s+")
    k = max(
        1,
        int(
            test_case.num_points
            // (test_case.num_points * test_case.min_max_points_split[1])
        ),
    )
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data)
    results[case_name] = {
        "labels": kmeans.labels_,
        "centers": kmeans.cluster_centers_,
        "data": data,
    }

# splits the cluster using our algorithm
# sys.path.append("/Users/maximilianverwiebe/codingprojects/bachelor-thesis/src/utils/")
from cluster_split import split_cluster

split_results = {}
for case_name, result in tqdm(results.items(), desc="Splitting clusters"):
    test_case = TEST_CASES[case_name]
    points_per_cluster = test_case.num_points // test_case.num_clusters

    max_points = int(test_case.min_max_points_split[1] * points_per_cluster)
    min_points = int(test_case.min_max_points_split[0] * points_per_cluster)

    data = result["data"]
    labels = result["labels"]
    labels_split, centers_split = split_cluster(
        X=data, labels=labels, max_size=max_points, min_points=min_points, seed=42
    )
    split_results[case_name] = {
        "labels_split": labels_split,
        "centers_split": centers_split,
    }


# computes the bboxes
def export_bboxes_txt(bounding_boxes, file_path):
    with open(file_path, "w") as f:
        for label in sorted(bounding_boxes):
            lo, hi = bounding_boxes[label]
            f.write(f"{label} {' '.join(map(str, lo))} {' '.join(map(str, hi))}\n")


bounding_boxes_split_results = {}
for case_name in tqdm(TEST_CASES.keys(), desc="Computing bounding boxes"):
    if case_name not in results or case_name not in split_results:
        logger.warning(f"[{case_name}] Missing results for bbox generation, skipping.")
        continue
    data = results[case_name]["data"]
    labels_split = split_results[case_name]["labels_split"]
    case_bboxes = {}
    for label in np.unique(labels_split):
        cluster_points = data[labels_split == label]
        if not cluster_points.empty:
            case_bboxes[label] = (
                cluster_points.min().values,
                cluster_points.max().values,
            )
    bounding_boxes_split_results[case_name] = case_bboxes
    bbox_file = os.path.join("test_cases", case_name, "bboxes_split.txt")
    export_bboxes_txt(case_bboxes, bbox_file)

# builds R-Tree and uses the queries
points_exec = "/Users/maximilianverwiebe/codingprojects/bachelor-thesis/src/rtree_cpp/batch_query/test_points/rtree_pointsvariant"
bboxes_exec = "/Users/maximilianverwiebe/codingprojects/bachelor-thesis/src/rtree_cpp/batch_query/test_bboxes/rtree_bboxesvariant"

for case_name in tqdm(TEST_CASES.keys(), desc="Running point variant"):
    case = TEST_CASES[case_name]
    points_per_cluster = case.num_points // case.num_clusters
    max_points = int(case.min_max_points_split[1] * points_per_cluster)

    case_dir = os.path.join("test_cases", case_name)
    input_file = os.path.join(case_dir, "points.txt")
    query_file = os.path.join(case_dir, "queries.txt")
    output_file = os.path.join(case_dir, "out_points_only.txt")
    if not os.path.exists(input_file):
        logger.warning(f"[{case_name}] Input file not found, skipping point variant.")
        continue
    cmd = f"{points_exec} {input_file} {query_file} {output_file} {case.num_dims} {max_points} {case.rtree_variant}"
    run_command(cmd, case_name=case_name)

for case_name in tqdm(TEST_CASES.keys(), desc="Running bbox variant"):
    case = TEST_CASES[case_name]
    points_per_cluster = case.num_points // case.num_clusters
    max_points = int(case.min_max_points_split[1] * points_per_cluster)

    case_dir = os.path.join("test_cases", case_name)
    bbox_file = os.path.join(case_dir, "bboxes_split.txt")
    input_file = os.path.join(case_dir, "points.txt")
    query_file = os.path.join(case_dir, "queries.txt")
    output_file = os.path.join(case_dir, "out_points_with_bboxes.txt")
    if not (os.path.exists(bbox_file) and os.path.exists(query_file)):
        logger.warning(
            f"[{case_name}] Bbox/Query file not found, skipping bbox variant."
        )
        continue
    cmd = f"{bboxes_exec} {bbox_file} {input_file} {query_file} {output_file} {case.num_dims} {max_points} {case.rtree_variant}"
    run_command(cmd, case_name=case_name)

logger.info("Pipeline finished successfully!")
print("\nPipeline finished. Check pipeline.log for details.")
