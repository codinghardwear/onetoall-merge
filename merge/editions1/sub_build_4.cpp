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
int M = 16;     int ef_construction = 200;      int ef_search = 10;

hnswlib::L2Space space(dim);

Timer timer;

int main() {

    cout << endl;

    hnswlib::HierarchicalNSW<float>* hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    cout << "-----SUB GRAPHS-----" << endl;
    cout << "1B elements,   128 dim,   16 M,   200 ef_construction,   10 ef_search." << endl;
    cout << endl;

    std::uniform_real_distribution<> distrib_real;
    std::mt19937 rng;  rng.seed(1);
    float* data1 = new float[dim * max_elements];
    if (data1 == nullptr)   cout << "No memory for data1." << endl;
    for (int i = 0; i < dim * max_elements; i++) {
        data1[i] = distrib_real(rng);  
    }


    int construction_threads = 4;      int search_threads = 4;

    timer.reset();

    #pragma omp parallel for
    for (int i = 0; i < max_elements; i++) {
        hnsw1->addPoint(data1 + i * dim, i);
    }

    cout << "Construct using " << construction_threads << " threads: " << timer.getElapsedTimeSeconds() << " seconds." << endl;

    hnsw1->saveIndex("sub_hnsw2_10k_s2.bin");
    // hnsw2->saveIndex("sub_hnsw2_12.5k_4p.bin");
    // hnsw1->saveIndex("sub_hnsw3_12.5k_4p.bin");
    // hnsw2->saveIndex("sub_hnsw4_12.5k_4p.bin");
    cout << "Save successfully." << endl;

    timer.reset();

    std::vector<hnswlib::labeltype> neighbors(max_elements);
    ParallelFor(0, max_elements, search_threads, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw1->searchKnn(data1 + dim * row, 1);
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

    cout << hnsw1->cur_element_count << endl;
    // cout << hnsw2->cur_element_count << endl;
    // cout << hnsw3->cur_element_count << endl;
    // cout << hnsw4->cur_element_count << endl;

    // delete[] data;
    // delete[] data1;
    // delete[] data2;
    delete hnsw1;
    delete hnsw2;
    delete hnsw3;
    delete hnsw4;
    
    delete[] data1;

    return 0;
}