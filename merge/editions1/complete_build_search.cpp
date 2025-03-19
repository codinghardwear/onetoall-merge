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
int dim = 128;   int max_elements = 30000;
int M = 16;     int ef_construction = 200;      int ef_search = 10;

hnswlib::L2Space space(dim);

Timer timer;

int main() {

    cout << endl;

    hnswlib::HierarchicalNSW<float>* hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    hnsw->ef_ = ef_search;
    cout << "-----COMPLETE GRAPH-----" << endl;
    cout << "30k elements,   128 dim,   16 M,   200 ef_construction,   10 ef_search." << endl;
    cout << endl;

    std::uniform_real_distribution<> distrib_real;
    std::mt19937 rng;  rng.seed(0);
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);  
    }

    int construction_threads = 4;      int search_threads = 4;

    timer.reset();

    ParallelFor(0, max_elements, construction_threads, [&](size_t row, size_t threadId) {
        hnsw->addPoint((void*)(data + dim * row), row);
    });
    
    // #pragma omp parallel for
    // for (int i = 0; i < max_elements; i++) {
    //     hnsw->addPoint(data + i * dim, i);
    // }

    cout << "Construct using " << construction_threads << " threads: " << timer.getElapsedTimeSeconds() << " seconds." << endl;

    hnsw->saveIndex("complete_hnsw_30k.bin");
    cout << "Save successfully." << endl;

    timer.reset();

    std::vector<hnswlib::labeltype> neighbors(max_elements);
    ParallelFor(0, max_elements, search_threads, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw->searchKnn(data + dim * row, 1);
        hnswlib::labeltype label = result.top().second;
        neighbors[row] = label;
    });
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        hnswlib::labeltype label = neighbors[i];
        if (label == i) correct++;
    }

    cout << "Search using " << search_threads << " threads: " << timer.getElapsedTimeSeconds() << " seconds ";
    float recall = (float)correct / max_elements;
    std::cout << "with recall: " << recall << ".\n";
    cout << endl;

    delete[] data;
    delete hnsw;

    return 0;
}