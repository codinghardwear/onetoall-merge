#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <omp.h>
#include <chrono>
#include <thread>
#include <algorithm>

#include "../hnsw/hnswlib/hnswlib.h"
#include "util/timer.h"
#include "util/multi_thread.h"
#include "util/graph.h"

using std::cout;
using std::endl;

// Global c
int c;

// Construction parameters
int dim = 128;   int max_elements = 1000000;
int M = 21;     int ef_construction = 200;      int ef_search = 10;

hnswlib::L2SpaceI space(dim);

Timer timer;

int main() {

    std::string hnsw_path = "com_hnsw_1M_t80_M24.bin";
    hnswlib::HierarchicalNSW<int>* complete_hnsw = new hnswlib::HierarchicalNSW<int>(&space, hnsw_path);        
    cout << "Complete hnsw loaded" << endl;

    int c_edge_count = 0;
    DistanceGraph graph_U0c(max_elements, 2 * M);
    for (int id = 0; id < max_elements; id++) {
        hnswlib::linklistsizeint* data = complete_hnsw->get_linklist0(id);
        int size = complete_hnsw->getListCount(data);
        c_edge_count += size;
        // hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
        // for (int j = 0; j < size; j++) {
        //     int neighbor_id = datal[j];
        //     float dist = complete_hnsw->fstdistfunc_(complete_hnsw->getDataByInternalId(id), complete_hnsw->getDataByInternalId(neighbor_id), complete_hnsw->dist_func_param_);
        //     graph_U0c.addNeighbor(id, neighbor_id, -dist);
        // }
    }
    cout << "Complete hnsw has " << c_edge_count << " edges" << endl;
    cout << "The ratio is " << (double)c_edge_count / max_elements << endl;    

    return 0;
}
