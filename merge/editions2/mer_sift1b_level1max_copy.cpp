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
int M = 16;     int ef_construction = 200;      int ef_search = 200;

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

void getNeighborsByHeuristicRevised(
    hnswlib::HierarchicalNSW<int>* hnsw,
    std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, compare> &top_candidates, const size_t M) {
    
    if (top_candidates.size() < M) {
        return;
    }

    std::priority_queue<std::pair<int, float>> queue_closest;
    std::vector<std::pair<int, float>> return_list;
    
    while (top_candidates.size() > 0) {
        queue_closest.emplace(top_candidates.top().first, -top_candidates.top().second);
        top_candidates.pop();
    }

    while (queue_closest.size()) {
        if (return_list.size() >= M) {
            break;
        }
        std::pair<int, float> curent_pair = queue_closest.top();
        float dist_to_query = -curent_pair.second;
        queue_closest.pop();
        bool good = true;

        for (std::pair<int, float> second_pair : return_list) {
            float curdist = hnsw->fstdistfunc_(hnsw->getDataByInternalId(second_pair.first), hnsw->getDataByInternalId(curent_pair.first), hnsw->dist_func_param_);
            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }
        if (good) {
                return_list.push_back(curent_pair);
        }
    }

    for (std::pair<int, float> current_pair : return_list) {
        top_candidates.emplace(current_pair.first, -current_pair.second);
    }
}

int main() {
    
    omp_set_num_threads(80);

    std::string mer_hnsw_path = "mer_hnsw_1M_t80.4.bin";
    hnswlib::HierarchicalNSW<int>* merged_hnsw = new hnswlib::HierarchicalNSW<int>(&space, mer_hnsw_path);

    std::string com_hnsw_path = "com_hnsw_1M_t80.bin";
    hnswlib::HierarchicalNSW<int>* com_hnsw = new hnswlib::HierarchicalNSW<int>(&space, com_hnsw_path);       

    timer.reset();
    DistanceGraph graph_U0(max_elements, 2 * M);
        
        #pragma omp parallel for
        for (int id = 0; id < max_elements; id++) {
            hnswlib::linklistsizeint* data = merged_hnsw->get_linklist0(id);
            int size = merged_hnsw->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
            for (int j = 0; j < size; j++) {
                int neighbor_id = datal[j];
                float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(neighbor_id), merged_hnsw->dist_func_param_);
                graph_U0.addNeighbor(id, neighbor_id, dist);
            }
        }

    // cout << "Level 0 graph_U0 time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

    // #pragma omp parallel for
    // for (int id = 0; id < max_elements; id++) {
    //     getNeighborsByHeuristicRevised(merged_hnsw, graph_U0.getNeighbors(id), 21);
    // }

    // Compare search time and recall
    const size_t num_queries = 10000;       
    const size_t k = 10;
    std::string query_path = "/media/raid5/myt/bigann_query.bvecs";   
    std::string groundtruth_path = "/media/raid5/myt/gnd/idx_1M.ivecs";  

    std::cout << "Loading query vectors..." << std::endl;
    auto queries = load_bvecs(query_path, num_queries, dim);

    std::cout << "Loading groundtruth..." << std::endl;
    auto groundtruth = load_ivecs(groundtruth_path, num_queries);

    timer.reset();

    for (int qid = 0; qid < num_queries; qid++) {
        // for (int i = 0; i < k; i++) {
        //     int id = groundtruth[qid][i];
        //     for (int j = 0; j < k-1; j++) {
        //         graph_U0.getNeighbors(id).pop();
        //     }
        // }
        for (int i = 0; i < k; i++) {
            int id = groundtruth[qid][i];
            for (int j = i + 1; j < k; j++) {
                int nid = groundtruth[qid][j];
                float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(nid), merged_hnsw->dist_func_param_);
                graph_U0.addNeighbor(id, nid, dist);
                graph_U0.addNeighbor(nid, id, dist);
            }
        }
    }
    
    // cout << "Level 0 neighbors modifying: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    // cout << endl;

    // graph_U0.printGraph(614443, 614443);


    timer.reset();
    #pragma omp parallel for
    for (int id = 0; id < max_elements; id++)
    {
        int size = graph_U0.getNeighbors(id).size();
        hnswlib::linklistsizeint* data = merged_hnsw->get_linklist0(id);
        merged_hnsw->setListCount(data, size);
        hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
        std::vector<int> neighbors(size);

        for (int idx = size - 1; idx >= 0; idx--) {
            datal[idx] = graph_U0.getNeighbors(id).top().first;
            graph_U0.getNeighbors(id).pop();
        }
    }
    // cout << "Level 0 neighbors rewriting time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    // cout << endl;

    // #pragma omp parallel for
    // for (int id = 0; id < max_elements; id++) {
    //     if (merged_hnsw->linkLists_[id] != nullptr) {
    //         free(merged_hnsw->linkLists_[id]);
    //         merged_hnsw->linkLists_[id] = nullptr;
    //         merged_hnsw->element_levels_[id] = 0;
    //     }
    // }

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

    merged_hnsw->ef_ = ef_search;

    timer.reset();
    int search_threads = 1;
    std::cout << "Testing Top-10 Recall..." << std::endl;
    float recall = test_topk(merged_hnsw, queries, groundtruth, num_queries, dim, k, search_threads);
    std::cout << "Top-10 Recall: " << recall << std::endl;
    cout << "Search ef is " << ef_search << endl;
    std::cout << "Search using " << search_threads << " threads: " << timer.getElapsedTimeSeconds() << " seconds." << std::endl;

    delete merged_hnsw;

    return 0;
}