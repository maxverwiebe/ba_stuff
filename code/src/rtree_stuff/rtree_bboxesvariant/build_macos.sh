g++ -std=c++17 rtree_bboxesvariant.cpp \
       -I$(brew --prefix spatialindex)/include \
       -L$(brew --prefix spatialindex)/lib -lspatialindex \
       -o rtree_bboxesvariant