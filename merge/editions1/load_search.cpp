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
int M = 16;     int ef_construction = 200;      int ef_search = 10;

hnswlib::L2Space space(dim);
Timer timer;


int main() {

    int hnsw1_elements = max_elements / 2;    
    int hnsw2_elements = max_elements / 2;
    int cur_elements = hnsw1_elements + hnsw2_elements;

    cout << max_elements << " elements with " << dim << " dimensions" << endl;
    cout << endl;

    // hnswlib::HierarchicalNSW<float>* hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, hnsw1_elements, M, ef_construction);
    // hnswlib::HierarchicalNSW<float>* hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, hnsw2_elements, M, ef_construction);

    // std::string hnsw_path1 = "sub_hnsw1_50k.bin";
    // std::string hnsw_path2 = "sub_hnsw2_50k.bin";

    // hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path1);        cout << "graph1" << endl;
    // hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path2);        cout << "graph2" << endl;
    // cout << endl;

    hnswlib::HierarchicalNSW<float>* hnsw = new hnswlib::HierarchicalNSW<float>(&space, cur_elements, M, ef_construction);

    std::string hnsw_path = "complete_hnsw_100k.bin";

    hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);       
    cout << "Complete graph loaded" << endl;
    cout << endl;

    std::uniform_real_distribution<> distrib_real;
    std::mt19937 rng;  rng.seed(0);
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);  
    }

    hnsw->ef_ = ef_search;
    cout << "ef_search is " << ef_search << endl;

    int correct = 0;
    timer.reset();

    std::vector<hnswlib::labeltype> record(cur_elements);
    omp_set_num_threads(4);
    #pragma omp parallel for
    for (int i = 0; i < cur_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        record[i] = label;
    }
    for (int i = 0; i < cur_elements; i++) {
        if (record[i] == i) {
            correct++;
        }
    }
    cout << "Complete graph search time: " << timer.getElapsedTimeSeconds() << " seconds \t";
    float recall = (float)correct / max_elements;
    std::cout << "Recall of Complete graph: " << recall << "\n";
    cout << endl;

    // delete[] data;
    // delete hnsw1;
    // delete hnsw2;
    // delete hnsw;

    return 0;
}