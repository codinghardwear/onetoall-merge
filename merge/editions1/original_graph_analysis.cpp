#include <omp.h>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include "matplotlibcpp.h"

#include "../hnsw/hnswlib/hnswlib.h"
#include "./util/timer.h"
#include "./util/multi_thread.h"
#include "./util/graph.h"

using std::cout;
using std::endl;
namespace plt = matplotlibcpp;

// Global c
int c;

// Construction parameters
int dim = 128;   int max_elements = 100000;
int M = 16;     int ef_construction = 200;      int ef_search = 10;

hnswlib::L2Space space(dim);

Timer timer;

int main() {

    std::string hnsw_path = "complete_hnsw_100k.bin";
    hnswlib::HierarchicalNSW<float>* complete_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);        
    cout << "Complete hnsw loaded" << endl;

    std::vector<int> neighbor_count(33);
    for (int i = 0; i < 33; i++)
        neighbor_count[i] = i;
    std::vector<int> node_number(2 * M + 1);
    DistanceGraph graph_U(max_elements, 2 * M);
    for (int id = 0; id < max_elements; id++) {
        hnswlib::linklistsizeint* data = complete_hnsw->get_linklist0(id);
        int size = complete_hnsw->getListCount(data);
        node_number[size]++;
        hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
        for (int j = 0; j < size; j++) {
            int neighbor_id = datal[j];
            float dist = complete_hnsw->fstdistfunc_(complete_hnsw->getDataByInternalId(id), complete_hnsw->getDataByInternalId(neighbor_id), complete_hnsw->dist_func_param_);
            graph_U0.addNeighbor(id, neighbor_id, dist);
        }
    }

    plt::plot(neighbor_count, node_number);
    plt::title("Neighbor Count Distribution");
    plt::xlabel("neighbor count");
    plt::ylabel("node number");
    plt::show();

    return 0;
}
