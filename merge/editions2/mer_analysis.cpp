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
int dim = 128;   int max_elements = 1000000;
int M = 16;     int ef_construction = 200;      int ef_search = 60;

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

    // std::string sub_hnsw1_path = "sub1_hnsw_10M_t80_f.bin";
    // hnswlib::HierarchicalNSW<int>* sub_hnsw1 = new hnswlib::HierarchicalNSW<int>(&space, sub_hnsw1_path);        
    // cout << "Sub hnsw1 loaded \t";
    // std::string sub_hnsw2_path = "sub2_hnsw_10M_t80_f.bin";
    // hnswlib::HierarchicalNSW<int>* sub_hnsw2 = new hnswlib::HierarchicalNSW<int>(&space, sub_hnsw2_path);        
    // cout << "Sub hnsw2 loaded \t";
    // std::string sub_hnsw3_path = "sub3_hnsw_10M_t80_f.bin";
    // hnswlib::HierarchicalNSW<int>* sub_hnsw3 = new hnswlib::HierarchicalNSW<int>(&space, sub_hnsw3_path);        
    // cout << "Sub hnsw3 loaded" << endl;
    // cout << endl;

    // int hnsw1_elements = sub_hnsw1->cur_element_count;      
    // int hnsw2_elements = sub_hnsw2->cur_element_count;
    // int hnsw3_elements = sub_hnsw3->cur_element_count;

    // cout << "sub graph elements: " << hnsw1_elements << " " << hnsw2_elements << " " << hnsw3_elements << endl;
    // cout << endl;

    // cout << "enter point: " << sub_hnsw1->enterpoint_node_ << " " << sub_hnsw2->enterpoint_node_ << " " << sub_hnsw3->enterpoint_node_ << " " << endl;

    // std::string mer_hnsw_path = "2sub_mer_hnsw_1M_t80_level1maxsearch.bin";
    std::string mer_hnsw_path = "mer_hnsw_1M_t80.add2loop1ori1maxnew.bin";
    hnswlib::HierarchicalNSW<int>* merged_hnsw = new hnswlib::HierarchicalNSW<int>(&space, mer_hnsw_path);
    std::string com_hnsw_path = "com_hnsw_1M_t80.bin";
    hnswlib::HierarchicalNSW<int>* com_hnsw = new hnswlib::HierarchicalNSW<int>(&space, com_hnsw_path);        

    // timer.reset();
    // DistanceGraph graph_U0(max_elements, 2 * M);
        
    //     #pragma omp parallel for
    //     for (int id = 0; id < max_elements; id++) {
    //         hnswlib::linklistsizeint* data = merged_hnsw->get_linklist0(id);
    //         int size = merged_hnsw->getListCount(data);
    //         hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //         for (int j = 0; j < size; j++) {
    //             int neighbor_id = datal[j];
    //             float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(neighbor_id), merged_hnsw->dist_func_param_);
    //             graph_U0.addNeighbor(id, neighbor_id, dist);
    //         }
    //     }

    const size_t num_queries = 10000;       
    const size_t k = 10;
    std::string query_path = "/media/raid5/myt/bigann_query.bvecs";   
    std::string groundtruth_path = "/media/raid5/myt/gnd/idx_1M.ivecs";  

    std::cout << "Loading query vectors..." << std::endl;
    auto queries = load_bvecs(query_path, num_queries, dim);

    std::cout << "Loading groundtruth..." << std::endl;
    auto groundtruth = load_ivecs(groundtruth_path, num_queries);

    // for (int qid = 0; qid < num_queries; qid++) {
    //     // for (int i = 0; i < k; i++) {
    //     //     int id = groundtruth[qid][i];
    //     //     for (int j = 0; j < k-1; j++) {
    //     //         graph_U0.getNeighbors(id).pop();
    //     //     }
    //     // }
    //     for (int i = 0; i < k; i++) {
    //         int id = groundtruth[qid][i];
    //         for (int j = i + 1; j < k; j++) {
    //             int nid = groundtruth[qid][j];
    //             float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(nid), merged_hnsw->dist_func_param_);
    //             graph_U0.addNeighbor(id, nid, j);
    //             graph_U0.addNeighbor(nid, id, j);
    //         }
    //     }
    // }

    // timer.reset();
    // #pragma omp parallel for
    // for (int id = 0; id < max_elements; id++)
    // {
    //     int size = graph_U0.getNeighbors(id).size();
    //     hnswlib::linklistsizeint* data = merged_hnsw->get_linklist0(id);
    //     merged_hnsw->setListCount(data, size);
    //     hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //     std::vector<int> neighbors(size);

    //     for (int idx = size - 1; idx >= 0; idx--) {
    //         datal[idx] = graph_U0.getNeighbors(id).top().first;
    //         graph_U0.getNeighbors(id).pop();
    //     }
    // }

    #pragma omp parallel for
    for (int id = 0; id < max_elements; id++)
    {
        hnswlib::linklistsizeint* data = merged_hnsw->get_linklist0(id);
        merged_hnsw->setListCount(data, 24);
    }

    // Compare search time and recall

    timer.reset();
    int search_threads = 1;
    cout << "ef search is " << ef_search << endl; 
    std::cout << "Testing Top-10 Recall..." << std::endl;
    merged_hnsw->ef_ = ef_search;
    float recall = test_topk(merged_hnsw, queries, groundtruth, num_queries, dim, k, search_threads);
    std::cout << "Top-10 Recall: " << recall << std::endl;
    std::cout << "Search using " << search_threads << " threads: " << timer.getElapsedTimeSeconds() << " seconds." << std::endl;

    // for (int id = 0; id < max_elements; id++) {
    //     std::cout << "Node " << id << " External Label: " << *(merged_hnsw->getExternalLabeLp(id)) << std::endl;
    // }

    // std::vector<int> level1_nodes;
    // for (int id = 0; id < max_elements; id++) {
    //     if (merged_hnsw->element_levels_[id] != 0) {
    //         level1_nodes.push_back(id);
    //     }
    // }

    // cout << "level 1 has " << level1_nodes.size() << " nodes" << endl;

    // timer.reset();
    // DistanceGraph graph_U1(max_elements, M);
    // // #pragma omp parallel for
    // for (int id : level1_nodes) {
    //     if (merged_hnsw->linkLists_[id] == nullptr) {
    //         cout << id << "does not allocate memory" << endl;
    //     }
    //     else {
    //         hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, 1);
    //         if (data == nullptr) {
    //             cout << id << endl;
    //             return 0;
    //         }
    //         int size = merged_hnsw->getListCount(data);
    //         hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //         for (int j = 0; j < size; j++) {
    //             int neighbor_id = datal[j];
    //             float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(neighbor_id), merged_hnsw->dist_func_param_);
    //             graph_U1.addNeighbor(id, neighbor_id, dist);
    //         }
    //     }
    // }

    // graph_U1.printGraph(level1_nodes[0], level1_nodes[0]);
    // graph_U1.printGraph(level1_nodes[level1_nodes.size()-1], level1_nodes[level1_nodes.size()-1]);

    // std::vector<int> level4_nodes;

    // for (int id = 0; id < max_elements; id++) {
    //     if (merged_hnsw->element_levels_[id] == 4) {
    //         level4_nodes.push_back(id);
    //     }
    // }

    // cout << "level 4 has " << level4_nodes.size() << " nodes" << endl;

    // timer.reset();
    // DistanceGraph graph_U4(max_elements, M);
    // // #pragma omp parallel for
    // for (int id : level4_nodes) {
    //     if (merged_hnsw->linkLists_[id] == nullptr) {
    //         cout << id << "does not allocate memory" << endl;
    //     }
    //     else {
    //         hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, 1);
    //         if (data == nullptr) {
    //             cout << id << endl;
    //             return 0;
    //         }
    //         int size = merged_hnsw->getListCount(data);
    //         hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //         for (int j = 0; j < size; j++) {
    //             int neighbor_id = datal[j];
    //             float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(neighbor_id), merged_hnsw->dist_func_param_);
    //             graph_U4.addNeighbor(id, neighbor_id, dist);
    //         }
    //     }
    // }

    // graph_U4.printGraph(level4_nodes[0], level4_nodes[0]);
    // graph_U4.printGraph(level4_nodes[level4_nodes.size()-1], level4_nodes[level4_nodes.size()-1]);

    return 0;
}