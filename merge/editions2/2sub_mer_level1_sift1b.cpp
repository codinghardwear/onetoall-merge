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

        // cout << i << " done " << endl;
    }

    return static_cast<float>(correct) / total;
}

int main() {

    omp_set_num_threads(80);

    std::string sub_hnsw1_path = "2sub1_hnsw_1M_t80_f.bin";
    hnswlib::HierarchicalNSW<int>* sub_hnsw1 = new hnswlib::HierarchicalNSW<int>(&space, sub_hnsw1_path);        
    cout << "Sub hnsw1 loaded \t";
    std::string sub_hnsw2_path = "2sub2_hnsw_1M_t80_f.bin";
    hnswlib::HierarchicalNSW<int>* sub_hnsw2 = new hnswlib::HierarchicalNSW<int>(&space, sub_hnsw2_path);        
    cout << "Sub hnsw2 loaded \t";

    int hnsw1_elements = sub_hnsw1->cur_element_count;      
    int hnsw2_elements = sub_hnsw2->cur_element_count;

    cout << "sub graph elements: " << hnsw1_elements << " " << hnsw2_elements << endl;

    std::string hnsw_path = "2sub_mer_hnsw_1M_t80.bin";
    hnswlib::HierarchicalNSW<int>* merged_hnsw = new hnswlib::HierarchicalNSW<int>(&space, hnsw_path);       
    cout << "Merge hnsw loaded \t";
    cout << merged_hnsw->cur_element_count;
    merged_hnsw->enterpoint_node_ = sub_hnsw1->enterpoint_node_;
    cout << endl;

    // Level 1 to Level max
    std::vector<std::vector<int>> nodes_level(merged_hnsw->maxlevel_ + 1);
    for (int id = 0; id < max_elements; id++) {
        if (merged_hnsw->linkLists_[id] != nullptr) {
            int level = merged_hnsw->element_levels_[id];
            for (int i = level; i >= 1; i--) {
                nodes_level[i].push_back(id);
            }
        }
    }

    std::vector<std::unordered_set<int>> nodes_set(merged_hnsw->maxlevel_ + 1);
    for (int level = 1; level <= merged_hnsw->maxlevel_; level++) {
        for (int i = 0; i < nodes_level[level].size(); i++) {
            nodes_set[level].insert(nodes_level[level][i]);
        }
    }

    // Level 1
    // No poping
    int add = 100;
    for (int level = 1; level <= merged_hnsw->maxlevel_; level++) {

        timer.reset();
        int num_add = 0;
        DistanceGraph graph_Ul(nodes_level[level].size(), M);

        for (int i = 0; i < nodes_level[level].size(); i++) {
            int id = nodes_level[level][i];

            hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, level);
            int size = merged_hnsw->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
            for (int j = 0; j < size; j++) {
                float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(datal[j]), merged_hnsw->dist_func_param_);
                graph_Ul.addNeighbor(i, datal[j], dist);
            }

        }

        for (int i = 0; i < nodes_level[level].size(); i++) {
            int id = nodes_level[level][i];
            if (id < hnsw1_elements) {
                // std::priority_queue<std::pair<int, hnswlib::labeltype>> result = sub_hnsw2->searchKnnTargetLayer(sub_hnsw1->getDataByInternalId(id), add, level);
                
                // if (result.empty()) cout << id << " empty " << endl;

                // while(result.size() != 0) {
                //     // if (graph_Ul.getNeighbors(i).size() == M)
                //     //     graph_Ul.getNeighbors(i).pop();
                //     if (nodes_set[level].find(result.top().second) == nodes_set[level].end()) {

                //     }
                //     else {
                //         graph_Ul.addNeighbor(i, result.top().second, result.top().first);
                //         num_add++;
                //     }
                //     // if (i == 0) cout << result.top().second << " ";
                //     // if (i == 0) graph_Ul.printGraph(0, 1);

                //     // if (i == nodes_level[level].size() - 1) graph_Ul.printGraph(nodes_level[level].size() - 2, nodes_level[level].size() - 1);

                //     result.pop();
                // }

                for (int j = 0; j < nodes_level[level].size(); j++) {
                    int nid = nodes_level[level][i];
                    float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(datal[j]), merged_hnsw->dist_func_param_);
                    graph_Ul.addNeighbor(i, nid, dist);
                }
            }
            else {
                std::priority_queue<std::pair<int, hnswlib::labeltype>> result = sub_hnsw1->searchKnnTargetLayer(sub_hnsw2->getDataByInternalId(id - hnsw1_elements), add, level);
                while(result.size() != 0) {
                    // if (graph_Ul.getNeighbors(i).size() == M)
                    //     graph_Ul.getNeighbors(i).pop();

                    if (nodes_set[level].find(result.top().second) == nodes_set[level].end()) {
                        
                    }
                    else {
                        graph_Ul.addNeighbor(i, result.top().second, result.top().first);
                        num_add++;
                    }
                    // if (i == nodes_level[level].size() - 1) cout << result.top().second << " ";
                    // if (i == 0) graph_Ul.printGraph(0, 1);

                    // if (i == nodes_level[level].size() - 1) graph_Ul.printGraph(nodes_level[level].size() - 2, nodes_level[level].size() - 1);

                    result.pop();
                }
            }
        }

        cout << "Level " << level << " has " << nodes_level[level].size() << " but add " << num_add << endl;
        cout << ((double)num_add / (nodes_level[level].size() * add)) << endl;
        cout << "Level " << level << " graph U1 and adding time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


        // timer.reset();

        // int iteration = 0;
        // do {
        //     std::vector<std::unordered_set<int>> U(graph_Ul.getElementsNum());
        //     for (int i = 0; i < graph_Ul.getElementsNum(); i++) {
        //         auto pq = graph_Ul.getNeighbors(i);
        //         while (!pq.empty()) {
        //             U[i].insert(pq.top().first);
        //             for (int j = 0; j < nodes_level[level].size(); j++) {
        //                 if (nodes_level[level][j] == pq.top().first) {
        //                     U[j].insert(nodes_level[level][i]);
        //                     break;
        //                 }
        //             }
        //             pq.pop();
        //         }
        //     }

        //     c = 0;
        //     for (int u = 0; u < graph_Ul.getElementsNum(); u++) {
        //         std::vector<int> neighbors(U[u].begin(), U[u].end());
        //         for (int i = 0; i < neighbors.size(); i++) {
        //             for (int j = i + 1; j < neighbors.size(); j++) {
        //                 int si = neighbors[i];
        //                 int sj = neighbors[j];

        //                 if ((si < hnsw1_elements && sj >= hnsw1_elements) || 
        //                     (si >= hnsw1_elements && sj < hnsw1_elements)) {
        //                     float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(si), merged_hnsw->getDataByInternalId(sj), merged_hnsw->dist_func_param_);

        //                     int si_idx = -1;     int sj_idx = -1;
        //                     for (int idx = 0; idx < nodes_level[level].size(); idx++) {
        //                         if (nodes_level[level][idx] == si)
        //                             si_idx = idx;
        //                         if (nodes_level[level][idx] == sj)
        //                             sj_idx = idx;
        //                     }
        //                     graph_Ul.updateNN(si_idx, sj, dist);
        //                     graph_Ul.updateNN(sj_idx, si, dist);
        //                 }
        //             }
        //         }
        //     }

        //     iteration++;

        // } while (c != 0 && iteration < 0);

        // float loopTime = timer.getElapsedTimeSeconds();
        // cout << "Level " << level << " looping time: " << loopTime << " seconds \t";
        // cout << iteration << " iterations with " << (float)loopTime / iteration << " average time" << endl;

        
        timer.reset();

        for (int i = 0; i < nodes_level[level].size(); i++) {
            int id = nodes_level[level][i];
            // getNeighborsByHeuristicRevised(merged_hnsw, graph_Ul.getNeighbors(i), M);
            auto neighbors = graph_Ul.getNeighbors(i);
            int size = neighbors.size();
            hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, level);
            merged_hnsw->setListCount(data, size);
            hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
            for (int idx = size - 1; idx >= 0; idx--) {
                datal[idx] = neighbors.top().first;
                neighbors.pop();
            }
        }

        cout << "Level " << level << " neighors resetting time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    }

    merged_hnsw->ef_ = ef_search;
    merged_hnsw->enterpoint_node_ = sub_hnsw2->enterpoint_node_ + hnsw1_elements;

    // Compare search time and recall
    const size_t num_queries = 10000;       
    const size_t k = 10;
    std::string query_path = "/media/raid5/myt/bigann_query.bvecs";   
    std::string groundtruth_path = "/media/raid5/myt/gnd/idx_1M.ivecs";  

    std::cout << "Loading query vectors..." << std::endl;
    auto queries = load_bvecs(query_path, num_queries, dim);

    std::cout << "Loading groundtruth..." << std::endl;
    auto groundtruth = load_ivecs(groundtruth_path, num_queries);

    // std::string mer_hnsw_path = "2sub_mer_hnsw_1M_t80.bin";
    // merged_hnsw->saveIndex(mer_hnsw_path);

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