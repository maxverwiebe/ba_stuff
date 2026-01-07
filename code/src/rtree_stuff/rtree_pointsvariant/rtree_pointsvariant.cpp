#include <algorithm>
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
    Coord p(dim);
    for (uint32_t d = 0; d < dim; ++d)
      if (!(iss >> p[d]))
        throw std::runtime_error("Malformed point: " + line);
    v.push_back(std::move(p));
  }
  return v;
}

struct RTreeBundle {
  std::unique_ptr<IStorageManager> sm;
  std::unique_ptr<ISpatialIndex> tree;
};

RTreeBundle build_tree_from_points(const std::vector<Coord> &pts,
                                   double fillFactor, uint32_t maxChildren,
                                   uint32_t dimension,
                                   SIR::RTreeVariant variant) {
  if (fillFactor <= 0.0 || fillFactor > 1.0)
    throw std::invalid_argument("fillFactor must be in (0,1].");
  if (maxChildren < 4)
    throw std::invalid_argument("maxChildren must be >= 4.");
  if (dimension == 0)
    throw std::invalid_argument("dimension must be >= 1.");

  for (const auto &p : pts)
    if (p.size() != dimension)
      throw std::invalid_argument("Point dimension mismatch.");

  RTreeBundle bundle;
  bundle.sm.reset(StorageManager::createNewMemoryStorageManager());

  id_type idxId = 1;
  bundle.tree.reset(SIR::createNewRTree(*bundle.sm, fillFactor, maxChildren,
                                        maxChildren, dimension, variant,
                                        idxId));

  for (size_t i = 0; i < pts.size(); ++i) {
    Region r(pts[i].data(), pts[i].data(), dimension);
    bundle.tree->insertData(0, nullptr, r, static_cast<id_type>(i + 1));
  }
  return bundle;
}

int main(int argc, char *argv[]) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " data_points.txt query_points.txt out.txt DIMENSIONS "
                 "MAX_CHILDREN VARIANT(0=LINEAR,1=QUADRATIC,2=RSTAR)\n";
    return 1;
  }

  try {
    uint32_t dimension = static_cast<uint32_t>(std::stoul(argv[4]));
    uint32_t maxChildren = static_cast<uint32_t>(std::stoul(argv[5]));

    auto dataPts = read_points(argv[1], dimension);
    auto queries = read_points(argv[2], dimension);

    uint32_t variantType = static_cast<uint32_t>(std::stoul(argv[6]));
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

    auto bundle =
        build_tree_from_points(dataPts, 0.5, maxChildren, dimension, variant);
    auto &tree = *bundle.tree;

    std::unique_ptr<IStatistics> stats;
    {
      IStatistics *raw = nullptr;
      tree.getStatistics(&raw);
      stats.reset(raw);
    }

    std::cout << "R-Tree: " << stats->getNumberOfNodes() << " Nodes ("
              << stats->getNumberOfData() << " Data objects)\n";

    std::cout << tree << '\n';

    std::ofstream out(argv[3]);
    if (!out)
      throw std::runtime_error("Cannot open output file " +
                               std::string(argv[3]));

    for (const auto &q : queries) {
      NodeCountVisitor vis;
      Region rq(q.data(), q.data(), dimension);
      tree.nearestNeighborQuery(1, rq, vis);
      out << (vis.hasHit ? static_cast<int>(vis.nodes) : -1) << '\n';
    }
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 2;
  }
  return 0;
}