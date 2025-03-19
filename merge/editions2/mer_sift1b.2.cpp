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

    hnswlib::HierarchicalNSW<int>* merged_hnsw = new hnswlib::HierarchicalNSW<int>(&space, max_elements, M, ef_construction);
    merged_hnsw->cur_element_count = max_elements;
    merged_hnsw->maxlevel_ = sub_hnsw2->maxlevel_;
    merged_hnsw->enterpoint_node_ = sub_hnsw2->enterpoint_node_;
    // cout << sub_hnsw2->enterpoint_node_ << endl;
    // cout << merged_hnsw->enterpoint_node_ << endl;
    merged_hnsw->ef_ = ef_search;

    cout << "-----MERGED GRAPH-----" << endl;
    cout << "1M elements,   128 dim,   16 M,   200 ef_construction,   10 ef_search." << endl;
    cout << endl;

    timer.reset();
    for (int id = 0; id < max_elements; id++) {
        // Set data and label
        memset(merged_hnsw->data_level0_memory_ + id * merged_hnsw->size_data_per_element_ + merged_hnsw->offsetLevel0_, 0, merged_hnsw->size_data_per_element_);

        if (id < hnsw1_elements) {
            memcpy(merged_hnsw->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
            memcpy(merged_hnsw->getDataByInternalId(id), sub_hnsw1->getDataByInternalId(id), merged_hnsw->data_size_);
        }
        else if (id >= hnsw1_elements && id < (hnsw1_elements + hnsw2_elements)) {
            void* src_data = sub_hnsw2->getDataByInternalId(id);
            if (src_data == nullptr) {
                std::cerr << "Invalid data pointer for id " << id << " in sub_hnsw1" << std::endl;
                return 0;
            }
            memcpy(merged_hnsw->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
            memcpy(merged_hnsw->getDataByInternalId(id), sub_hnsw2->getDataByInternalId(id), merged_hnsw->data_size_);
        }
        else {
            memcpy(merged_hnsw->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
            memcpy(merged_hnsw->getDataByInternalId(id), sub_hnsw3->getDataByInternalId(id), merged_hnsw->data_size_);
        }
    }
    cout << "Level 0 copying time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    timer.reset();
    DistanceGraph graph_U0(max_elements, 2 * M);
    #pragma omp parallel
    {
        #pragma omp for
        for (int id = 0; id < hnsw1_elements; id++) {
            hnswlib::linklistsizeint* data = sub_hnsw1->get_linklist0(id);
            int size = sub_hnsw1->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
            for (int j = 0; j < size; j++) {
                int neighbor_id = datal[j];
                float dist = sub_hnsw1->fstdistfunc_(sub_hnsw1->getDataByInternalId(id), sub_hnsw1->getDataByInternalId(neighbor_id), sub_hnsw1->dist_func_param_);
                graph_U0.addNeighbor(id, neighbor_id, dist);
            }
        }
        #pragma omp for
        for (int id = hnsw1_elements; id < (hnsw1_elements + hnsw2_elements); id++) {
            hnswlib::linklistsizeint* data = sub_hnsw2->get_linklist0(id);
            int size = sub_hnsw2->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
            for (int j = 0; j < size; j++) {
                int neighbor_id = datal[j];
                float dist = sub_hnsw2->fstdistfunc_(sub_hnsw2->getDataByInternalId(id), sub_hnsw2->getDataByInternalId(neighbor_id), sub_hnsw2->dist_func_param_);
                graph_U0.addNeighbor(id, neighbor_id, dist);
            }
        }   
        #pragma omp for
        for (int id = (hnsw1_elements + hnsw2_elements); id < max_elements; id++) {
            hnswlib::linklistsizeint* data = sub_hnsw3->get_linklist0(id);
            int size = sub_hnsw3->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
            for (int j = 0; j < size; j++) {
                int neighbor_id = datal[j];
                float dist = sub_hnsw3->fstdistfunc_(sub_hnsw3->getDataByInternalId(id), sub_hnsw3->getDataByInternalId(neighbor_id), sub_hnsw3->dist_func_param_);
                graph_U0.addNeighbor(id, neighbor_id, dist);
            }
        }   
    }
    cout << "Level 0 graph_U0 time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

    timer.reset();
    int add = 4;
    // #pragma omp parallel 
    {
        // #pragma omp for
        for (int id = hnsw1_elements; id < (hnsw1_elements + hnsw2_elements); id++) {
            std::priority_queue<std::pair<int, hnswlib::labeltype>> result1 = sub_hnsw1->searchKnn(sub_hnsw2->getDataByInternalId(id), add);
            std::priority_queue<std::pair<int, hnswlib::labeltype>> result3 = sub_hnsw3->searchKnn(sub_hnsw2->getDataByInternalId(id), add);
            std::vector<std::pair<int, hnswlib::labeltype>> vec1;
            std::vector<std::pair<int, hnswlib::labeltype>> vec3;
            while (!result1.empty()) {
                vec1.push_back(result1.top());
                result1.pop();
            }
            while (!result3.empty()) {
                vec3.push_back(result3.top());
                result3.pop();
            }

            for (int i = 0; i < add; i++) {
                hnswlib::tableint G1_node = vec1[i].second;
                hnswlib::tableint G3_node = vec3[i].second;
                graph_U0.addNeighbor(id, G1_node, vec1[i].first);
                graph_U0.addNeighbor(id, G3_node, vec3[i].first);
            }

            for (int i = 0; i < add; i++) {
                hnswlib::tableint G1_node = vec1[i].second;
                for (int j = 0; j < add; j++) {
                    hnswlib::tableint neighbor = vec3[j].second;
                    float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(G1_node), merged_hnsw->getDataByInternalId(neighbor), merged_hnsw->dist_func_param_);
                    graph_U0.addNeighbor(G1_node, neighbor, dist);
                }
            }

            for (int i = 0; i < add; i++) {
                hnswlib::tableint G3_node = vec3[i].second;
                for (int j = 0; j < add; j++) {
                    hnswlib::tableint neighbor = vec1[j].second;
                    float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(G3_node), merged_hnsw->getDataByInternalId(neighbor), merged_hnsw->dist_func_param_);
                    graph_U0.addNeighbor(G3_node, neighbor, dist);
                }
            }
        }
    }
    cout << "Level 0 adding time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    timer.reset();
    int iteration = 0;
    do {
        std::vector<std::unordered_set<int>> U(graph_U0.getElementsNum());
        // #pragma omp parallel for
        for (int i = 0; i < graph_U0.getElementsNum(); i++) {
            auto pq = graph_U0.getNeighborsCopy(i);
            while (!pq.empty()) {
                U[i].insert(pq.top().first);
                U[pq.top().first].insert(i);
                pq.pop();
            }
        }
        c = 0;
        // #pragma omp parallel for
        for (int u = 0; u < graph_U0.getElementsNum(); u++) {
            std::vector<int> neighbors(U[u].begin(), U[u].end());
            for (int i = 0; i < neighbors.size(); i++) {
                // #pragma omp parallel for
                for (int j = i + 1; j < neighbors.size(); j++) {
                    int si = neighbors[i];
                    int sj = neighbors[j];
                    // if ((graph_id(si) != 2 && graph_id(sj) == 2) || (graph_id(si) == 2 && graph_id(sj) != 2)) {
                    if (graph_id(si, hnsw1_elements, hnsw2_elements) != graph_id(sj, hnsw1_elements, hnsw2_elements)) {
                        float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(si), merged_hnsw->getDataByInternalId(sj), merged_hnsw->dist_func_param_);
                        graph_U0.updateNN(si, sj, dist);
                        graph_U0.updateNN(sj, si, dist);
                    }
                }
            }
        }
        iteration++;
    } while (c != 0 && iteration < 0);
    float loopTime = timer.getElapsedTimeSeconds();
    cout << "Level 0 looping time: " << loopTime << " seconds \t";
    cout << iteration << " iterations with " << (float)loopTime / iteration << " average time" << endl;


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
    cout << "Level 0 neighbors rewriting time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    cout << endl;


    timer.reset();
    #pragma omp parallel 
    {
        #pragma omp for
        for (int id = 0; id < max_elements; id++) {
            int level = sub_hnsw1->element_levels_[id];
            merged_hnsw->element_levels_[id] = level;
            if (level == 0)
                merged_hnsw->linkLists_[id] = nullptr;
            else {
                int size = merged_hnsw->size_links_per_element_ * level;
                merged_hnsw->linkLists_[id] = (char *) malloc(size);
                if (merged_hnsw->linkLists_[id] != nullptr) {
                    memcpy(merged_hnsw->linkLists_[id], sub_hnsw1->linkLists_[id], size);
                }
                else
                    cout << "no enough space for " << id << " ";
            }
        }
    }

    cout << "Level 1 to level max copying time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    cout << endl;

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
    int search_threads = 1;
    std::cout << "Testing Top-10 Recall..." << std::endl;
    float recall = test_topk(merged_hnsw, queries, groundtruth, num_queries, dim, k, search_threads);
    std::cout << "Top-10 Recall: " << recall << std::endl;
    std::cout << "Search using " << search_threads << " threads: " << timer.getElapsedTimeSeconds() << " seconds." << std::endl;

    delete sub_hnsw1;
    delete sub_hnsw2;
    delete sub_hnsw3;
    delete merged_hnsw;

    return 0;
}