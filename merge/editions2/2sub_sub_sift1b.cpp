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

std::vector<unsigned char> load_bvecs(const std::string& path, size_t num_vectors, size_t dim) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Unable to open file: " + path);
    }

    std::vector<unsigned char> data(num_vectors * dim);
    for (size_t i = 0; i < num_vectors; i++) {
        int d;
        input.read((char *)(&d), sizeof(int));
        if (d != dim) {
            std::cerr << "Expected dimension: " << dim << ", found dimension: " << d << " at vector " << i << std::endl;
            throw std::runtime_error("Dimension mismatch in bvecs file.");
        }
        input.read((char *)(data.data() + i * dim), dim);
    }

    return data;
}

std::vector<std::vector<unsigned int>> load_ivecs(const std::string& path, size_t num_queries) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Unable to open file: " + path);
    }

    std::vector<std::vector<unsigned int>> groundtruth(num_queries, std::vector<unsigned int>(1000));
    for (size_t i = 0; i < num_queries; i++) {
        int t;
        input.read((char *)(&t), sizeof(int));
        if (t != 1000) {
            throw std::runtime_error("Groundtruth format error.");
        }
        input.read((char *)(groundtruth[i].data()), 1000 * sizeof(unsigned int));
    }

    return groundtruth;
}

float test_topk(hnswlib::HierarchicalNSW<int>* hnsw, const std::vector<unsigned char>& queries,
                 const std::vector<std::vector<unsigned int>>& groundtruth, size_t num_queries, size_t dim, size_t k, size_t search_threads) {
    size_t correct = 0;
    size_t total = num_queries * k;

    omp_set_num_threads(search_threads);
    #pragma omp parallel for
    for (size_t i = 0; i < num_queries; i++) {
        auto result = hnsw->searchKnn(queries.data() + i * dim, k);
        std::unordered_set<unsigned int> gt;

        for (int j = 0; j < k; j++) {
            gt.insert(groundtruth[i][j]);
        }

        while (!result.empty()) {
            if (gt.find(result.top().second) != gt.end()) {
                correct++;
            }
            result.pop();
        }
    }

    return static_cast<float>(correct) / total;
}

int main() {
    const size_t dim = 128;
    const size_t num_base = 1000000;  
    const size_t num_queries = 10000;           

    const size_t M = 16;
    const size_t ef_construction = 200;

    const size_t k = 10;

    std::string base_path = "/media/raid5/myt/bigann_base.bvecs";     
    std::string query_path = "/media/raid5/myt/bigann_query.bvecs";   
    std::string groundtruth_path = "/media/raid5/myt/gnd/idx_1M.ivecs";  
    std::string hnsw1_path = "2sub1_hnsw_1M_t80_f.bin"; 
    std::string hnsw2_path = "2sub2_hnsw_1M_t80_f.bin";

    cout << "Saving path is " << hnsw1_path << endl;

    std::cout << "Loading base vectors..." << std::endl;
    auto base_vectors = load_bvecs(base_path, num_base, dim);

    std::cout << "Building HNSW index..." << std::endl;
    int share = num_base / 2;
    hnswlib::L2SpaceI space(dim);
    hnswlib::HierarchicalNSW<int>* hnsw1 = new hnswlib::HierarchicalNSW<int>(&space, share, M, ef_construction);
    hnswlib::HierarchicalNSW<int>* hnsw2 = new hnswlib::HierarchicalNSW<int>(&space, share, M, ef_construction);

    Timer timer;        
    timer.reset();
    int construction_threads = 80;
    omp_set_num_threads(construction_threads);
    #pragma omp parallel
    {
        #pragma omp for
        for (size_t i = 0; i < share; i++) {
            hnsw1->addPoint((void *)(base_vectors.data() + i * dim), i);
        }
        #pragma omp for
        for (size_t i = share; i < num_base; i++) {
            hnsw2->addPoint((void *)(base_vectors.data() + i * dim), i);
        }
    }
    std::cout << "Construct using " << construction_threads << " threads: " << timer.getElapsedTimeSeconds() << " seconds." << std::endl;

    std::cout << "Saving HNSW index to disk..." << std::endl;
    hnsw1->saveIndex(hnsw1_path);
    hnsw2->saveIndex(hnsw2_path);

    // std::cout << "Loading query vectors..." << std::endl;
    // auto queries = load_bvecs(query_path, num_queries, dim);

    // std::cout << "Loading groundtruth..." << std::endl;
    // auto groundtruth = load_ivecs(groundtruth_path, num_queries);

    // timer.reset();
    // int search_threads = 80;
    // std::cout << "Testing Top-10 Recall..." << std::endl;
    // float recall = test_topk(hnsw, queries, groundtruth, num_queries, dim, k, search_threads);
    // std::cout << "Top-10 Recall: " << recall << std::endl;
    // std::cout << "Search using " << search_threads << " threads: " << timer.getElapsedTimeSeconds() << " seconds." << std::endl;
    delete hnsw1;
    delete hnsw2;

    return 0;
}