## Required
* Python3 (`numpy`; `matplotlib`; `scikit-learn`; `pandas`; `tqdm`)
* C++ (g++...)
* LibSpatialIndex (https://github.com/libspatialindex/libspatialindex)


## Usage
### 1. Installation of dependencies
Make sure that you have all dependencies installed.

You have to be able build a C++ project using the library `LibSpatialIndex` (https://libspatialindex.org/en/latest/#download).

---

### 2. Build the R-Tree binaries
Unfortunately, no proper R-Tree solution for Python exists.
Thats why I had to make use of a C++ implementation.
You need to build two simple C++ files, to receive two binaries for your OS.

#### Build rtree_bboxesvariant
Example for MacOS is located under `code/src/rtree_stuff/rtree_bboxesvariant/build_macos.sh`.

#### Build rtree_pointsvariant
Example for MacOS is located under `code/src/rtree_stuff/rtree_pointsvariant/build_macos.sh`.


---
### 3. code/src/multigen_cached.py
Call the Python3 script `code/src/multigen_cached.py`.
Here is what it does:

#### 0) Defines a big grid of test cases

It builds a combinatorial set of `TestCase`s over:

* `num_clusters`: `[1, 2, 5]`
* `num_points` (total points per dataset): `[5000, 10000, 20000, 50000, 100000, 200000]`
* `distribution`: `["gaussian", "uniform"]`
* `noise_scale` (only for gaussian): `[0.1, 0.3, 0.5, 0.7]`
* `num_dims`: `[2, 5, 10, 20]`
* `rtree_variant`: `[0, 1, 2]` → `["Linear", "Quadratic", "RStar"]`
* `min_max_points_split`: `[(0.005, 0.01)]` (relative to "points per original cluster")

Because uniform forces `noise_scale=0.0` (single value), while gaussian uses 4 noise scales, this ends up creating **1080 test cases** total, see the thesis text.

It then recreates a clean output folder:

* deletes `test_cases/` if it exists
* creates `test_cases/` fresh
* creates/keeps `distribution_cache/` for reuse across runs


#### 1) Generates (and caches) the synthetic datasets

For each test case:

1. Creates `test_cases/<case_name>/`

2. Computes `points_per_cluster = num_points // num_clusters`

3. Writes `test_cases/<case_name>/info.txt` with all parameters, plus:

   * absolute min/max split sizes:

     * `min_abs = rel_min * points_per_cluster`
     * `max_abs = rel_max * points_per_cluster`

4. Produces the dataset file `points.txt`, using caching:

   * It builds a cache key like
     `distribution_cache/c{clusters}_p{points_per_cluster}_d-{distribution}_n{noise}_dim{dims}.txt`
   * If that cache file exists: **copy it** to `test_cases/<case>/points.txt`
   * Else: **generate it** by calling one of these scripts:

     * `generator_gaussian.py` with `--noise_scale ...`
     * `generator_uniform.py` with `--spacing 0`
   * Then copies the newly created cached file into the case folder as `points.txt`

All generator command output goes into `pipeline.log`


#### 2) Generates query points for each dataset

For each case that has a `points.txt`, it creates:

* `test_cases/<case>/queries.txt`

by randomly sampling **1000 existing points** from the dataset and saving them.


#### 3) Runs K-Means clustering on each dataset

For each case:

* Loads `points.txt` into a pandas DataFrame
* Computes the number of KMeans clusters as:

```python
k = int(num_points // (num_points * max_rel))
```

Then it fits:

* `KMeans(n_clusters=k, random_state=42, n_init=10)`

and stores labels + centers in memory.


#### 4) Splits clusters with the custom splitting algorithm

It calls my custom function:

* `split_cluster(X=data, labels=labels, max_size=max_points, min_points=min_points, seed=42)`

Result:

* `labels_split` and `centers_split` per case

This step is meant to enforce that final clusters lie within a target size range as described in the thesis


#### 5) Computes bounding boxes of the split clusters

For each case, it computes axis-aligned bounding boxes for each split cluster:

* `lo = min per dimension`
* `hi = max per dimension`

and writes them to:

* `test_cases/<case>/bboxes_split.txt`

Format per line:

```
<label> <lo_0> <lo_1> ... <lo_d> <hi_0> <hi_1> ... <hi_d>
```


#### 6) Runs the two C++ R-Tree binaries (batch build + query)

Finally it runs two external binaries (absolute paths in the script, you might have to adapt them):

* **Point variant** (`rtree_pointsvariant`):

  * inputs: `points.txt`, `queries.txt`
  * output: `out_points_only.txt`

Command args:

```
rtree_pointsvariant <points.txt> <queries.txt> <out_points_only.txt> <num_dims> <max_points> <rtree_variant>
```

* **BBox variant** (`rtree_bboxesvariant`):

  * inputs: `bboxes_split.txt`, `points.txt`, `queries.txt`
  * output: `out_points_with_bboxes.txt`

Command args:

```
rtree_bboxesvariant <bboxes_split.txt> <points.txt> <queries.txt> <out_points_with_bboxes.txt> <num_dims> <max_points> <rtree_variant>
```

#### What you get on disk per case

Each `test_cases/caseN/` ends up containing (assuming nothing failed):

* `info.txt` parameters + min/max sizes
* `points.txt` generated dataset (copied from cache)
* `queries.txt` 1000 sampled query points
* `bboxes_split.txt` bounding boxes from split clusters
* `out_points_only.txt` results of point-only R-Tree run
* `out_points_with_bboxes.txt` results of bbox R-Tree run

And one global log:

* `pipeline.log`

---

### 4. Analyze the data
You can now analyze each test case individually (`test_cases/caseN/`) using standard data analysis methods.
Or you can also analyze the whole dataset (`test_cases`) to compare both indices in a bulk.
