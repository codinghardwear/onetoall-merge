#include <omp.h>
#include <chrono>
#include <thread>
#include <algorithm>

#include "../hnsw/hnswlib/hnswlib.h"
#include "./util/timer.h"
#include "./util/multi_thread.h"
#include "./util/graph.h"

using std::cout;
using std::endl;

// Global c
int c;

// Construction parameters
int dim = 128;   int max_elements = 100000;
int M = 16;     int ef_construction = 200;      int ef_search = 200;

hnswlib::L2Space space(dim);

Timer timer;

int main() {

    int hnsw1_elements = max_elements / 2;    
    int hnsw2_elements = max_elements / 2;
    int cur_elements = hnsw1_elements + hnsw2_elements;

    std::string hnsw_path = "complete_hnsw_100k.bin";
    hnswlib::HierarchicalNSW<float>* complete_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);        
    cout << "Complete hnsw loaded" << endl;

    int c_edge_count = 0;
    DistanceGraph graph_U0c(cur_elements, 2 * M);
    for (int id = 0; id < cur_elements; id++) {
        hnswlib::linklistsizeint* data = complete_hnsw->get_linklist0(id);
        int size = complete_hnsw->getListCount(data);
        c_edge_count += size;
        hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
        for (int j = 0; j < size; j++) {
            int neighbor_id = datal[j];
            float dist = complete_hnsw->fstdistfunc_(complete_hnsw->getDataByInternalId(id), complete_hnsw->getDataByInternalId(neighbor_id), complete_hnsw->dist_func_param_);
            graph_U0c.addNeighbor(id, neighbor_id, -dist);
        }
    }
    cout << "Complete hnsw has " << c_edge_count << " edges" << endl;

    int cross_edge_count = 0;
    for (int id = 0; id < hnsw1_elements; id++) {
        auto pq = graph_U0c.getNeighbors(id);
        while (!pq.empty()) {
            if (pq.top().first >= hnsw1_elements)
                cross_edge_count++;
            pq.pop();
        }
    }
    for (int id = hnsw1_elements; id < cur_elements; id++) {
        auto pq = graph_U0c.getNeighbors(id);
        while (!pq.empty()) {
            if (pq.top().first < hnsw1_elements)
                cross_edge_count++;
            pq.pop();
        }
    }
    cout << "Complete hnsw has " << c_edge_count << " edges" << endl;
    cout << "Complete hnsw has " << cross_edge_count << " cross edges" << endl;
    cout << "The ratio of cross edges / total edges is " << (float)cross_edge_count / c_edge_count << endl;

    return 0;
}
