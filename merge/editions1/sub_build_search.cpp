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
int dim = 128;   int max_elements = 25000;
int M = 16;     int ef_construction = 200;      int ef_search = 10;

hnswlib::L2Space space(dim);

Timer timer;

int main() {

    cout << endl;

    int hnsw1_elements = max_elements / 2;    int hnsw2_elements = max_elements / 2;
    hnswlib::HierarchicalNSW<float>* hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, hnsw1_elements, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, hnsw2_elements, M, ef_construction);
    hnsw1->ef_ = ef_search;
    hnsw2->ef_ = ef_search;
    cout << "-----SUB GRAPHS-----" << endl;
    cout << "50k elements,   128 dim,   16 M,   200 ef_construction,   200 ef_search." << endl;
    cout << endl;

    std::uniform_real_distribution<> distrib_real;
    std::mt19937 rng;  rng.seed(0);
    float* data = new float[dim * max_elements];
    if (data == nullptr)   cout << "No memory for data." << endl;
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);  
    }
    float* data1 = new float[dim * hnsw1_elements];
    if (data1 == nullptr)   cout << "No memory for data1." << endl;
    for (int i = 0; i < dim * hnsw1_elements; i++) {
        data1[i] = data[i]; 
    }
    float* data2 = new float[dim * hnsw2_elements];
    if (data2 == nullptr)   cout << "No memory for data2." << endl;
    for (int i = dim * hnsw1_elements; i < dim * max_elements; i++) {
        data2[i - dim * hnsw1_elements] = data[i];
    }

    int construction_threads = 4;      int search_threads = 4;

    timer.reset();

    ParallelFor(0, hnsw1_elements, construction_threads, [&](size_t row, size_t threadId) {
        hnsw1->addPoint((void*)(data1 + dim * row), row);
    });
    ParallelFor(0, hnsw2_elements, construction_threads, [&](size_t row, size_t threadId) {
        hnsw2->addPoint((void*)(data2 + dim * row), row);
    });

    // #pragma omp parallel
    // {
    //     #pragma omp for
    //     for (int i = 0; i < hnsw1_elements; i++) {
    //         hnsw1->addPoint(data1 + i * dim, i);
    //     }

    //     #pragma omp for
    //     for (int i = 0; i < hnsw2_elements; i++) {
    //         hnsw2->addPoint(data2 + i * dim, i);
    //     }
    // }

    cout << "Construct using " << construction_threads << " threads: " << timer.getElapsedTimeSeconds() << " seconds." << endl;

    // hnsw1->saveIndex("sub_hnsw1_50k_t80.bin");
    // hnsw2->saveIndex("sub_hnsw2_50k_t80.bin");
    // cout << "Save successfully." << endl;

    timer.reset();

    std::vector<hnswlib::labeltype> neighbors(hnsw1_elements);
    ParallelFor(0, hnsw1_elements, search_threads, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw1->searchKnn(data1 + dim * row, 1);
        hnswlib::labeltype label = result.top().second;
        neighbors[row] = label;
    });
    float correct = 0;
    for (int i = 0; i <hnsw1_elements; i++) {
        hnswlib::labeltype label = neighbors[i];
        if (label == i) correct++;
    }

    cout << "Search using " << search_threads << " threads: " << timer.getElapsedTimeSeconds() << " seconds ";
    float recall = (float)correct / hnsw1_elements;
    std::cout << "with recall: " << recall << ".\n";
    cout << endl;

    delete[] data;
    delete[] data1;
    delete[] data2;
    delete hnsw1;
    delete hnsw2;

    return 0;
}