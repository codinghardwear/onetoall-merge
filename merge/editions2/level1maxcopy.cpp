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
#include "./util/timer.h"
#include "./util/multi_thread.h"
#include "./util/graph.h"

using std::cout;
using std::endl;

// Global c
int c;

// Construction parameters
int dim = 128;   int max_elements = 1000000;
int M = 16;     int ef_construction = 200;      int ef_search = 10;

hnswlib::L2SpaceI space(dim);

Timer timer;

int graph_id(int node_id, int hnsw1_elements, int hnsw2_elements) {
    if (node_id >= 0 && node_id < hnsw1_elements)
        return 1;
    else if (node_id >= hnsw1_elements && node_id < (hnsw1_elements + hnsw2_elements))
        return 2;
    else
        return 3;
}

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
    size_t total = k;

    // omp_set_num_threads(search_threads);
    // #pragma omp parallel for
    for (size_t i = 0; i < 1; i++) {
        auto query_path = hnsw->searchKnnTopPath(queries.data() + i * dim, k);
        cout << "size is " << query_path.size() << endl;
        for (int i = 0; i < query_path.size(); i++) {
            cout << query_path[i] << "  ";
        }
        cout << endl;
    }

    return static_cast<float>(correct) / total;
}

int main() {
    
    omp_set_num_threads(80);

    hnswlib::HierarchicalNSW<int>* merged_hnsw = new hnswlib::HierarchicalNSW<int>(&space, max_elements, M, ef_construction);
    merged_hnsw->cur_element_count = max_elements;

    std::string com_hnsw_path = "com_hnsw_1M_t80.bin";
    hnswlib::HierarchicalNSW<int>* com_hnsw = new hnswlib::HierarchicalNSW<int>(&space, com_hnsw_path);

    #pragma omp parallel for
    for (int id = 0; id < max_elements; id++) {
        int level = com_hnsw->element_levels_[id];
        merged_hnsw->element_levels_[id] = level;
        if (level == 0)
            merged_hnsw->linkLists_[id] = nullptr;
        else {
            int size = merged_hnsw->size_links_per_element_ * level;
            merged_hnsw->linkLists_[id] = (char *) malloc(size);
            if (merged_hnsw->linkLists_[id] != nullptr) {
                memcpy(merged_hnsw->linkLists_[id], com_hnsw->linkLists_[id], size);
            }
            else
                cout << "no enough space for " << id << " " << endl;
        }
    }

    merged_hnsw->maxlevel_ = com_hnsw->maxlevel_;
    merged_hnsw->enterpoint_node_ = com_hnsw->enterpoint_node_;

    merged_hnsw->ef_ = ef_search;

    const size_t num_queries = 10000;       
    const size_t k = 10;
    std::string query_path = "/media/raid5/myt/bigann_query.bvecs";   
    std::string groundtruth_path = "/media/raid5/myt/gnd/idx_1M.ivecs";  

    std::cout << "Loading query vectors..." << std::endl;
    cout << endl;
    auto queries = load_bvecs(query_path, num_queries, dim);

    std::cout << "Loading groundtruth..." << std::endl;
    cout << endl;
    auto groundtruth = load_ivecs(groundtruth_path, num_queries);

    timer.reset();
    int search_threads = 1;
    float recall = test_topk(com_hnsw, queries, groundtruth, num_queries, dim, k, search_threads);
    recall = test_topk(merged_hnsw, queries, groundtruth, num_queries, dim, k, search_threads);

    return 0;
}