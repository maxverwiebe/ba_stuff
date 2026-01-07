#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <spatialindex/SpatialIndex.h>
#include <spatialindex/tools/Tools.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SpatialIndex;
namespace SIR = SpatialIndex::RTree;

using Coord = std::vector<double>;

struct BBox {
  uint64_t id;
  Coord lo, hi;
};

struct NodeCountVisitor : public IVisitor {
  uint64_t nodes = 0, leaves = 0;
  bool hasHit = false;
  void visitNode(const INode &n) override {
    ++nodes;
    if (n.isLeaf())
      ++leaves;
  }
  void visitData(const IData &) override { hasHit = true; }
  void visitData(std::vector<const IData *> &v) override {
    if (!v.empty())
      hasHit = true;
  }
};

std::vector<BBox> read_bboxes(const std::string &path, uint32_t dim) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot open bbox file " + path);
  std::vector<BBox> v;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream iss(line);
    BBox b;
    b.lo.resize(dim);
    b.hi.resize(dim);
    if (!(iss >> b.id))
      throw std::runtime_error("Malformed bbox id: " + line);
    for (uint32_t d = 0; d < dim; ++d)
      if (!(iss >> b.lo[d]))
        throw std::runtime_error("Malformed bbox lo: " + line);
    for (uint32_t d = 0; d < dim; ++d)
      if (!(iss >> b.hi[d]))
        throw std::runtime_error("Malformed bbox hi: " + line);
    for (uint32_t d = 0; d < dim; ++d)
      if (b.lo[d] > b.hi[d])
        std::swap(b.lo[d], b.hi[d]);
    v.push_back(std::move(b));
  }
  return v;
}

std::vector<Coord> read_points(const std::string &path, uint32_t dim) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot open point file " + path);
  std::vector<Coord> v;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    for (char &c : line)
      if (c == ',')
        c = ' ';
    std::istringstream iss(line);
    Coord pt(dim);
    for (uint32_t d = 0; d < dim; ++d)
      if (!(iss >> pt[d]))
        throw std::runtime_error("Malformed point: " + line);
    v.push_back(std::move(pt));
  }
  return v;
}

static inline bool contains(const BBox &b, const Coord &p) {
  for (size_t d = 0; d < p.size(); ++d)
    if (p[d] < b.lo[d] || p[d] > b.hi[d])
      return false;
  return true;
}

static inline double sqr(double x) { return x * x; }

// MINDIST(q, R) for Point->MBR
static double mindist2(const Coord &q, const BBox &b) {
  double sum = 0.0;
  for (size_t d = 0; d < q.size(); ++d) {
    if (q[d] < b.lo[d])
      sum += sqr(b.lo[d] - q[d]);
    else if (q[d] > b.hi[d])
      sum += sqr(q[d] - b.hi[d]);
    else
      sum += 0.0;
  }
  return sum;
}

// Morton/Z-Order Key (2D only). For d != 2: lexicographical.
static uint64_t morton2D(double x, double y, double xmin, double xmax,
                         double ymin, double ymax, unsigned bits = 20) {
  auto clamp01 = [](double v) { return v < 0 ? 0.0 : (v > 1 ? 1.0 : v); };
  double nx = (xmax > xmin) ? (x - xmin) / (xmax - xmin) : 0.0;
  double ny = (ymax > ymin) ? (y - ymin) / (ymax - ymin) : 0.0;
  nx = clamp01(nx);
  ny = clamp01(ny);
  uint64_t ix = static_cast<uint64_t>(nx * ((1ULL << bits) - 1));
  uint64_t iy = static_cast<uint64_t>(ny * ((1ULL << bits) - 1));

  auto part1by1 = [](uint64_t n) {
    n &= 0x00000000ffffffffULL;
    n = (n | (n << 16)) & 0x0000ffff0000ffffULL;
    n = (n | (n << 8)) & 0x00ff00ff00ff00ffULL;
    n = (n | (n << 4)) & 0x0f0f0f0f0f0f0f0fULL;
    n = (n | (n << 2)) & 0x3333333333333333ULL;
    n = (n | (n << 1)) & 0x5555555555555555ULL;
    return n;
  };
  return (part1by1(iy) << 1) | part1by1(ix);
}

struct PointRec {
  uint64_t id;
  uint64_t cluster; // BBox.id
  Coord x;
  uint64_t key; // ordering key (Morton or lexicographic)
};

// Assigns a cluster to each point (containment, min MINDIST in case of
// ambiguity) and sorts by (cluster, key).
static std::vector<PointRec> assign_and_order(const std::vector<Coord> &points,
                                              const std::vector<BBox> &boxes,
                                              uint32_t dim) {
  // global extents for Morton scaling (2D)
  double xmin = std::numeric_limits<double>::infinity();
  double ymin = std::numeric_limits<double>::infinity();
  double xmax = -xmin, ymax = -ymin;
  if (dim >= 2) {
    for (const auto &p : points) {
      xmin = std::min(xmin, p[0]);
      xmax = std::max(xmax, p[0]);
      ymin = std::min(ymin, p[1]);
      ymax = std::max(ymax, p[1]);
    }
  }

  std::vector<PointRec> recs;
  recs.reserve(points.size());

  for (size_t i = 0; i < points.size(); ++i) {
    const auto &p = points[i];
    // 1) candidates by containment
    std::vector<size_t> cand;
    cand.reserve(boxes.size());
    for (size_t b = 0; b < boxes.size(); ++b)
      if (contains(boxes[b], p))
        cand.push_back(b);

    size_t best = std::numeric_limits<size_t>::max();
    double bestDist2 = std::numeric_limits<double>::infinity();

    if (cand.empty()) {
      // 2) no exact containment: use MBR with minimum MINDIST
      for (size_t b = 0; b < boxes.size(); ++b) {
        double d2 = mindist2(p, boxes[b]);
        if (d2 < bestDist2) {
          bestDist2 = d2;
          best = b;
        }
      }
    } else if (cand.size() == 1) {
      best = cand[0];
    } else {
      // 3) multiple hits: select smallest MINDIST (smallest inclusion)
      for (size_t b : cand) {
        double d2 = mindist2(p, boxes[b]);
        if (d2 < bestDist2) {
          bestDist2 = d2;
          best = b;
        }
      }
    }

    PointRec r;
    r.id = static_cast<uint64_t>(i + 1);
    r.cluster = boxes[best].id;
    r.x = p;
    if (dim >= 2) {
      r.key = morton2D(p[0], p[1], xmin, xmax, ymin, ymax);
    } else {
      // Fallback: 1D or d>2 -> simple lex key via first attribute
      r.key = 0;
    }
    recs.push_back(std::move(r));
  }

  // sort: (cluster_id, key)
  std::sort(recs.begin(), recs.end(), [](const PointRec &a, const PointRec &b) {
    if (a.cluster != b.cluster)
      return a.cluster < b.cluster;
    if (a.key != b.key)
      return a.key < b.key;
    return a.id < b.id;
  });
  return recs;
}

int main(int argc, char *argv[]) {
  try {
    auto t0 = std::chrono::steady_clock::now();

    if (argc != 8) {
      std::cerr
          << "Usage: " << argv[0]
          << " bboxes.txt data_points.txt query_points.txt out.txt DIMENSIONS "
             "MAX_CHILDREN VARIANT(0=LINEAR,1=QUADRATIC,2=RSTAR)\n";
      return 1;
    }

    const char *bboxPath = argv[1];
    const char *dataPath = argv[2];
    const char *queryPath = argv[3];
    const char *outPath = argv[4];
    uint32_t dimension = static_cast<uint32_t>(std::stoul(argv[5]));
    uint32_t maxChildren = static_cast<uint32_t>(std::stoul(argv[6]));
    uint32_t variantType = static_cast<uint32_t>(std::stoul(argv[7]));

    SIR::RTreeVariant variant;
    switch (variantType) {
    case 0:
      variant = SIR::RV_LINEAR;
      break;
    case 1:
      variant = SIR::RV_QUADRATIC;
      break;
    case 2:
      variant = SIR::RV_RSTAR;
      break;
    default:
      std::cerr
          << "Invalid variant type. Use 0=LINEAR, 1=QUADRATIC, or 2=RSTAR\n";
      return 1;
    }

    auto bboxes = read_bboxes(bboxPath, dimension);
    auto dataPoints = read_points(dataPath, dimension);
    auto queryPoints = read_points(queryPath, dimension);

    // 1) cluster assignment & ordering
    auto recs = assign_and_order(dataPoints, bboxes, dimension);

    // 2) Incremental build: only points, no MBRs as data objects!!!!!
    std::unique_ptr<IStorageManager> sm(
        StorageManager::createNewMemoryStorageManager());
    id_type indexId = 1;
    double fillFactor = 0.5;
    uint32_t indexCap = maxChildren;
    uint32_t leafCap = maxChildren;

    std::unique_ptr<ISpatialIndex> tree(SIR::createNewRTree(
        *sm, fillFactor, indexCap, leafCap, dimension, variant, indexId));

    // insert points in cluster order
    for (const auto &r : recs) {
      Region reg(r.x.data(), r.x.data(), dimension);
      tree->insertData(0, nullptr, reg, static_cast<id_type>(r.id));
    }

    // 3) statssss
    std::unique_ptr<IStatistics> stats;
    {
      IStatistics *raw = nullptr;
      tree->getStatistics(&raw);
      stats.reset(raw);
    }
    std::cout << "R-Tree: " << stats->getNumberOfNodes() << " Nodes ("
              << stats->getNumberOfData() << " Data objects)\n";

    // 4) queries: kNN k=1
    std::ofstream out(outPath);
    if (!out)
      throw std::runtime_error(std::string("Cannot open output file ") +
                               outPath);

    for (const auto &qp : queryPoints) {
      NodeCountVisitor vis;
      Region q(qp.data(), qp.data(), dimension);
      tree->nearestNeighborQuery(1, q, vis);
      out << (vis.hasHit ? static_cast<int>(vis.nodes) : -1) << '\n';
    }

    auto t1 = std::chrono::steady_clock::now();
    double build_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "BUILD_INDEX_MS: " << build_ms << "\n";
    std::cout.flush();
  } catch (Tools::Exception &e) {
    std::cerr << "SpatialIndex error: " << e.what() << '\n';
    return 3;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
    return 2;
  }
  return 0;
}
