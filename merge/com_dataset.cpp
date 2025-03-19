#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <omp.h>
#include <chrono>
#include <thread>
#include <algorithm>
#include <random>
#include <cstdint>

#include "../hnsw/hnswlib/hnswlib.h"
#include "util/timer.h"
#include "util/multi_thread.h"
#include "util/graph.h"

int c = 0;

// Construction parameters
int dim = 96;   int num_base = 10000000;
int M = 16;     int ef_construction = 200;

hnswlib::L2SpaceI space(dim);

Timer timer;

int main() {

    std::cout << "This is DEEP 10M" << std::endl;

    // std::string hnsw_path = "DP_1M_96_16t.bin";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<float> vectors(num_base * dim);
    for(int i = 0; i < num_base * dim; ++i) {
        vectors[i] = dist(gen);
    }

    std::cout << "Building HNSW index..." << std::endl;
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* hnsw = new hnswlib::HierarchicalNSW<float>(&space, num_base, M, ef_construction);

    Timer timer;        
    timer.reset();
    int construction_threads = 16;
    omp_set_num_threads(construction_threads);
    #pragma omp parallel for
    for (size_t i = 0; i < num_base; i++) {
        hnsw->addPoint((void *)(vectors.data() + i * dim), i);
    }
    std::cout << "Construct using " << construction_threads << " threads: " << timer.getElapsedTimeSeconds() << " seconds." << std::endl;

    // std::cout << "Saving HNSW index to disk..." << std::endl;
    // hnsw->saveIndex(hnsw_path);

    delete hnsw;

    return 0;
}