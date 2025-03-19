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

int c = 0;

// Construction parameters
int dim = 128;   int max_elements = 100000000;
int M = 30;     int ef_construction = 200;      int ef_search = 250;

hnswlib::L2SpaceI space(dim);

Timer timer;

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

    // omp_set_num_threads(search_threads);
    // #pragma omp parallel for
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

    // std::string sub_hnsw1_path = "sub1_hnsw_10M_t80_f.bin";
    // hnswlib::HierarchicalNSW<int>* sub_hnsw1 = new hnswlib::HierarchicalNSW<int>(&space, sub_hnsw1_path);        
    // cout << "Sub hnsw1 loaded \t";
    // std::string sub_hnsw2_path = "sub2_hnsw_10M_t80_f.bin";
    // hnswlib::HierarchicalNSW<int>* sub_hnsw2 = new hnswlib::HierarchicalNSW<int>(&space, sub_hnsw2_path);        
    // cout << "Sub hnsw2 loaded \t";
    // std::string sub_hnsw3_path = "sub3_hnsw_10M_t80_f.bin";
    // hnswlib::HierarchicalNSW<int>* sub_hnsw3 = new hnswlib::HierarchicalNSW<int>(&space, sub_hnsw3_path);        
    // cout << "Sub hnsw3 loaded" << endl;

    // int hnsw1_elements = sub_hnsw1->cur_element_count;      
    // int hnsw2_elements = sub_hnsw2->cur_element_count;
    // int hnsw3_elements = sub_hnsw3->cur_element_count;

    // cout << "sub graph elements: " << hnsw1_elements << " " << hnsw2_elements << " " << hnsw3_elements << endl;
    // cout << endl;

    std::string com_hnsw_path = "com_M30_100M_80t.bin"; 
    hnswlib::HierarchicalNSW<int>* com_hnsw = new hnswlib::HierarchicalNSW<int>(&space, com_hnsw_path);        
    com_hnsw->ef_ = ef_search;
    // cout << "Com hnsw loaded" << endl;

    const size_t num_queries = 10000;       
    const size_t k = 10;
    std::string query_path = "/media/raid5/myt/bigann_query.bvecs";   
    std::string groundtruth_path = "/media/raid5/myt/gnd/idx_100M.ivecs";  

    // std::cout << "Loading query vectors..." << std::endl;
    auto queries = load_bvecs(query_path, num_queries, dim);

    // std::cout << "Loading groundtruth..." << std::endl;
    auto groundtruth = load_ivecs(groundtruth_path, num_queries);

    for (ef_search = 350; ef_search <= 700; ef_search+=10) {
        com_hnsw->ef_ = ef_search;
        timer.reset();
        int search_threads = 1;
        // std::cout << "Testing Top-10 Recall..." << std::endl;
        float recall = test_topk(com_hnsw, queries, groundtruth, num_queries, dim, k, search_threads);
        // std::cout << "Top-10 Recall: " << recall << std::endl;
        std::cout << "Search ef: " << ef_search << "   ";
        std::cout << "Top-10 Recall: " << recall << "   ";
        std::cout << "Search time: " << (timer.getElapsedTimeSeconds() * 1000 / num_queries) << " seconds." << std::endl;  
    }

    // timer.reset();
    // DistanceGraph graph_U0(max_elements, 2 * M);
    // #pragma omp parallel
    // {
    //     #pragma omp for
    //     for (int id = 0; id < max_elements; id++) {
    //         hnswlib::linklistsizeint* data = com_hnsw->get_linklist0(id);
    //         int size = com_hnsw->getListCount(data);
    //         hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //         for (int j = 0; j < size; j++) {
    //             int neighbor_id = datal[j];
    //             float dist = com_hnsw->fstdistfunc_(com_hnsw->getDataByInternalId(id), com_hnsw->getDataByInternalId(neighbor_id), com_hnsw->dist_func_param_);
    //             graph_U0.addNeighbor(id, neighbor_id, dist);
    //         }
    //     }
    // }
    // cout << "Level 0 graph_U0 time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

    // graph_U0.printGraph(hnsw1_elements - 3, hnsw1_elements);
    // graph_U0.printGraph((hnsw1_elements + hnsw2_elements) - 3, (hnsw1_elements + hnsw2_elements));
    // graph_U0.printGraph((hnsw1_elements + hnsw2_elements + hnsw3_elements) - 3, (hnsw1_elements + hnsw2_elements + hnsw3_elements));

    // timer.reset();
    // DistanceGraph graph_U1(max_elements, M);
    // // #pragma omp parallel for
    // for (int id = hnsw1_elements; id < (hnsw1_elements + hnsw2_elements); id++) {
    //     if (com_hnsw->element_levels_[id] != 0 && com_hnsw->linkLists_[id] != nullptr) {
    //         hnswlib::linklistsizeint* data = com_hnsw->get_linklist(id, 1);
    //         if (data == nullptr) {
    //             cout << id << endl;
    //             return 0;
    //         }
    //         int size = com_hnsw->getListCount(data);
    //         hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //         for (int j = 0; j < size; j++) {
    //             int neighbor_id = datal[j];
    //             float dist = com_hnsw->fstdistfunc_(com_hnsw->getDataByInternalId(id), com_hnsw->getDataByInternalId(neighbor_id), com_hnsw->dist_func_param_);
    //             graph_U1.addNeighbor(id - hnsw1_elements, neighbor_id, 0);
    //         }
    //     }
    // }

    // graph_U1.printGraph(0, 10);

    return 0;
}