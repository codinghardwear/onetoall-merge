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
int M = 16;     int ef_construction = 200;      int ef_search = 110;

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

// float test_topk(hnswlib::HierarchicalNSW<int>* hnsw, const std::vector<unsigned char>& queries,
//                 const std::vector<std::vector<unsigned int>>& groundtruth, size_t num_queries, size_t dim, size_t k, size_t search_threads) {
//     size_t correct = 0;
//     size_t total = k;

//     // omp_set_num_threads(search_threads);
//     // #pragma omp parallel for
//     for (size_t i = 0; i < 1; i++) {
//         auto [result, query_path] = hnsw->searchKnnPath(queries.data() + i * dim, k);
//         cout << result.size() << " " << query_path.size() << endl;
//         for (int i = 0; i < query_path.size(); i++) {
//             cout << query_path[i] << "  ";
//         }
//         cout << endl;
//         cout << "size is " << query_path.size() << endl;
//         cout << endl;

//         std::unordered_set<unsigned int> gt;

//         for (int j = 0; j < k; j++) {
//             cout << groundtruth[i][j] << "  ";
//             gt.insert(groundtruth[i][j]);
//         }
//         cout << endl;

//         while (!result.empty()) {
//             if (gt.find(result.top().second) != gt.end()) {
//                 correct++;
//             }
//             cout << result.top().second << " with " << result.top().first << "  ";
//             result.pop();
//         }
//         cout << endl;
//     }

//     return static_cast<float>(correct) / total;
// }

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

    std::string sub_hnsw1_path = "sub1_hnsw_1M_t80_f.bin";
    hnswlib::HierarchicalNSW<int>* sub_hnsw1 = new hnswlib::HierarchicalNSW<int>(&space, sub_hnsw1_path);        
    cout << "Sub hnsw1 loaded \t";
    std::string sub_hnsw2_path = "sub2_hnsw_1M_t80_f.bin";
    hnswlib::HierarchicalNSW<int>* sub_hnsw2 = new hnswlib::HierarchicalNSW<int>(&space, sub_hnsw2_path);        
    cout << "Sub hnsw2 loaded \t";
    std::string sub_hnsw3_path = "sub3_hnsw_1M_t80_f.bin";
    hnswlib::HierarchicalNSW<int>* sub_hnsw3 = new hnswlib::HierarchicalNSW<int>(&space, sub_hnsw3_path);        
    cout << "Sub hnsw3 loaded" << endl;

    int hnsw1_elements = sub_hnsw1->cur_element_count;      
    int hnsw2_elements = sub_hnsw2->cur_element_count;
    int hnsw3_elements = sub_hnsw3->cur_element_count;

    cout << "sub graph elements: " << hnsw1_elements << " " << hnsw2_elements << " " << hnsw3_elements << endl;
    cout << endl;

    std::string mer_hnsw_path = "mer_hnsw_1M_t80.add2loop1.bin";
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

    // for (int qid = 0; qid < num_queries; qid++) {
    //     for (int i = 0; i < k; i++) {
    //         int id = groundtruth[qid][i];
    //         for (int j = 0; j < k-1; j++) {
    //             graph_U0.getNeighbors(id).pop();
    //         }
    //     }
    //     for (int i = 0; i < k; i++) {
    //         int id = groundtruth[qid][i];
    //         for (int j = i + 1; j < k; j++) {
    //             int nid = groundtruth[qid][j];
    //             float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(nid), merged_hnsw->dist_func_param_);
    //             graph_U0.addNeighbor(id, nid, dist);
    //             graph_U0.addNeighbor(nid, id, dist);
    //         }
    //     }
    // }
    
    // cout << "Level 0 neighbors modifying: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    // cout << endl;

    // graph_U0.printGraph(614443, 614443);

    // #pragma omp parallel for
    // for (int id = 0; id < max_elements; id++) {
    //     getNeighborsByHeuristicRevised(merged_hnsw, graph_U0.getNeighbors(id), 24);
    // }

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
                graph_U0.addNeighbor(id, nid, j);
                graph_U0.addNeighbor(nid, id, j);
            }
        }
    }


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

    timer.reset();
    #pragma omp parallel for
    for (int id = 0; id < max_elements; id++)
    {
        hnswlib::linklistsizeint* data = merged_hnsw->get_linklist0(id);
        merged_hnsw->setListCount(data, 24);
    }
    cout << "Level 0 neighbors rewriting time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    cout << endl;

    // #pragma omp parallel for
    // for (int id = 0; id < max_elements; id++) {
    //     getNeighborsByHeuristicRevised(merged_hnsw, graph_U0.getNeighbors(id), 21);
    // }

    // std::vector<std::vector<int>> nodes_level(merged_hnsw->maxlevel_ + 1);
    // for (int id = 0; id < max_elements; id++) {
    //     if (merged_hnsw->linkLists_[id] != nullptr) {
    //         int level = merged_hnsw->element_levels_[id];
    //         for (int i = level; i >= 1; i--) {
    //             nodes_level[i].push_back(id);
    //         }
    //     }
    // }

    // std::vector<std::unordered_set<int>> nodes_set(merged_hnsw->maxlevel_ + 1);
    // for (int level = 1; level <= merged_hnsw->maxlevel_; level++) {
    //     for (int i = 0; i < nodes_level[level].size(); i++) {
    //         nodes_set[level].insert(nodes_level[level][i]);
    //     }
    // }

    // for (int level = 1; level <= merged_hnsw->maxlevel_; level++) {

    //     timer.reset();
    //     DistanceGraph graph_Ul(max_elements, M);

    //     for (int i = 0; i < nodes_level[level].size(); i++) {
    //         int id = nodes_level[level][i];

    //         hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, level);
    //         int size = merged_hnsw->getListCount(data);
    //         hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //         for (int j = 0; j < size; j++) {
    //             float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(datal[j]), merged_hnsw->dist_func_param_);
    //             graph_Ul.addNeighbor(id, datal[j], dist);
    //         }

    //     }

    //     #pragma omp parallel for
    //     for (int i = 0; i < nodes_level[level].size(); i++) {
    //         int id = nodes_level[level][i];
    //             for (int j = 0; j < nodes_level[level].size(); j++) {
    //                 int nid = nodes_level[level][j];
    //                 if (graph_id(id, hnsw1_elements, hnsw2_elements) != graph_id(nid, hnsw1_elements, hnsw2_elements)) {
    //                     float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(nid), merged_hnsw->dist_func_param_);
    //                     graph_Ul.addNeighbor(id, nid, dist);
    //                 }
    //             }
    //     }

    //     cout << "Level " << level << " graph U1 and adding time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
        
    //     timer.reset();

    //     for (int i = 0; i < nodes_level[level].size(); i++) {
    //         int id = nodes_level[level][i];
    //         // getNeighborsByHeuristicRevised(merged_hnsw, graph_Ul.getNeighbors(i), M);
    //         auto neighbors = graph_Ul.getNeighbors(id);
    //         int size = neighbors.size();
    //         hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, level);
    //         merged_hnsw->setListCount(data, size);
    //         hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //         for (int idx = size - 1; idx >= 0; idx--) {
    //             datal[idx] = neighbors.top().first;
    //             neighbors.pop();
    //         }
    //     }

    //     cout << "Level " << level << " neighors resetting time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    // }

    std::vector<std::vector<int>> nodes_level(com_hnsw->maxlevel_ + 1);
    for (int id = 0; id < max_elements; id++) {
        if (com_hnsw->linkLists_[id] != nullptr) {
            int level = com_hnsw->element_levels_[id];
            for (int i = level; i >= 1; i--) {
                nodes_level[i].push_back(id);
            }
        }
    }

    std::vector<std::unordered_set<int>> nodes_set(com_hnsw->maxlevel_ + 1);
    for (int level = 1; level <= com_hnsw->maxlevel_; level++) {
        for (int i = 0; i < nodes_level[level].size(); i++) {
            nodes_set[level].insert(nodes_level[level][i]);
        }
    }

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
    merged_hnsw->enterpoint_node_ = com_hnsw->enterpoint_node_;

    timer.reset();
    int search_threads = 1;
    std::cout << "Testing Top-10 Recall..." << std::endl;
    float recall = test_topk(merged_hnsw, queries, groundtruth, num_queries, dim, k, search_threads);
    std::cout << "Top-10 Recall: " << recall << std::endl;
    cout << "Search ef is " << ef_search << endl;
    std::cout << "Search using " << search_threads << " threads: " << timer.getElapsedTimeSeconds() << " seconds." << std::endl;

    std::string save_hnsw_path = "mer_hnsw_1M_t80.modify.bin";
    merged_hnsw->saveIndex(save_hnsw_path);

    delete merged_hnsw;

    return 0;
}