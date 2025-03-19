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
    const size_t num_base = 10000000;  
    const size_t num_queries = 10000;           

    const size_t M = 16;
    const size_t ef_construction = 200;
    const size_t k = 10;
    const size_t num_subsets = 10;

    std::string base_path = "/media/raid5/myt/bigann_base.bvecs";     
    std::string query_path = "/media/raid5/myt/bigann_query.bvecs";   
    std::string groundtruth_path = "/media/raid5/myt/gnd/idx_10M.ivecs";  

    std::vector<std::string> hnsw_paths;
    for (size_t i = 1; i <= num_subsets; i++) {
        hnsw_paths.push_back("sub" + std::to_string(i) + "_10M_1t.bin");
    }

    std::cout << "Loading base vectors..." << std::endl;
    auto base_vectors = load_bvecs(base_path, num_base, dim);

    std::cout << "Building HNSW index..." << std::endl;
    size_t share = num_base / num_subsets;

    hnswlib::L2SpaceI space(dim);
    std::vector<hnswlib::HierarchicalNSW<int>*> hnsw_indexes;

    for (size_t i = 0; i < num_subsets; i++) {
        size_t subset_size = share;
        hnsw_indexes.push_back(new hnswlib::HierarchicalNSW<int>(&space, subset_size, M, ef_construction));
    }

    Timer timer;
    timer.reset();

    size_t start_idx = 0;
    for (size_t i = 0; i < num_subsets; i++) {
        size_t subset_size = share;
        for (size_t j = 0; j < subset_size; j++) {
            hnsw_indexes[i]->addPoint((void *)(base_vectors.data() + (start_idx + j) * dim), start_idx + j);
        }
        start_idx += subset_size;
    }

    std::cout << "Construct using 1 thread: " << timer.getElapsedTimeSeconds() << " seconds." << std::endl;

    std::cout << "Saving HNSW index to disk..." << std::endl;
    for (size_t i = 0; i < num_subsets; i++) {
        hnsw_indexes[i]->saveIndex(hnsw_paths[i]);
        delete hnsw_indexes[i];
    }

    return 0;
}