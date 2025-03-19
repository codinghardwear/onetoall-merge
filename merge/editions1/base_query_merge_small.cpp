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
int dim = 128;   int max_elements = 50000;
int M = 16;     int ef_construction = 200;      int ef_search = 200;

hnswlib::L2Space space(dim);

Timer timer;

int main() {

    int hnsw1_elements = max_elements / 2;    int hnsw2_elements = max_elements / 2;
    int cur_elements = hnsw1_elements + hnsw2_elements;

    hnswlib::HierarchicalNSW<float>* hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, hnsw1_elements, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, hnsw2_elements, M, ef_construction);

    std::string hnsw_path1 = "sub_hnsw1_25k_t1.bin";
    std::string hnsw_path2 = "sub_hnsw2_25k_t1.bin";

    hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path1);        cout << "graph1 \t";
    hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path2);        cout << "graph2" << endl;       cout << endl;

    std::uniform_real_distribution<> distrib_real;
    std::mt19937 rng;  rng.seed(0);
    float* data = new float[dim * cur_elements];
    for (size_t i = 0; i < dim * cur_elements; i++) {
        data[i] = distrib_real(rng);  
    }

    hnswlib::HierarchicalNSW<float>* merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    merged_hnsw->cur_element_count = cur_elements;
    merged_hnsw->maxlevel_ = hnsw1->maxlevel_;
    merged_hnsw->enterpoint_node_ = hnsw1->enterpoint_node_;
    merged_hnsw->ef_ = ef_search;

    cout << "-----MERGED GRAPH-----" << endl;
    cout << "50k elements,   128 dim,   16 M,   200 ef_construction,   200 ef_search." << endl;
    cout << endl;

    timer.reset();

    for (size_t id = 0; id < cur_elements; id++) {
            // Set data and label
        memset(merged_hnsw->data_level0_memory_ + id * merged_hnsw->size_data_per_element_ + merged_hnsw->offsetLevel0_, 0, merged_hnsw->size_data_per_element_);

        if (id < hnsw1_elements) {
            memcpy(merged_hnsw->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
            memcpy(merged_hnsw->getDataByInternalId(id), hnsw1->getDataByInternalId(id), merged_hnsw->data_size_);
        }
        else {
            memcpy(merged_hnsw->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
            memcpy(merged_hnsw->getDataByInternalId(id), hnsw2->getDataByInternalId(id - hnsw1_elements), merged_hnsw->data_size_);
        }
    }

    cout << "Level 0 copying time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    timer.reset();

    #pragma omp parallel 
    {
        #pragma omp for
        for (size_t id = 0; id < hnsw1_elements; id++)
        {
            merged_hnsw->element_levels_[id] = hnsw1->element_levels_[id];
            if (merged_hnsw->element_levels_[id] == 0)
                merged_hnsw->linkLists_[id] = nullptr;
            else
            {
                size_t size = merged_hnsw->size_links_per_element_ * merged_hnsw->element_levels_[id];
                merged_hnsw->linkLists_[id] = (char*) malloc(size);

                if (merged_hnsw->linkLists_[id] != nullptr) {
                    memset(merged_hnsw->linkLists_[id], 0, size);
                    memcpy(merged_hnsw->linkLists_[id], hnsw1->linkLists_[id], size);
                }
                else {
                    cout << "no enough space for " << id << " ";
                }
            }
        }

        #pragma omp for
        for (int id = 0; id < hnsw2_elements; id++)
        {
            int level = hnsw2->element_levels_[id];
            merged_hnsw->element_levels_[id + hnsw1_elements] = level;
            if (level == 0)
                merged_hnsw->linkLists_[id + hnsw1_elements] = nullptr;
            else
            {
                int size = merged_hnsw->size_links_per_element_ * level;
                merged_hnsw->linkLists_[id + hnsw1_elements] = (char*) malloc(size);

                if (merged_hnsw->linkLists_[id] != nullptr) {
                    memset(merged_hnsw->linkLists_[id + hnsw1_elements], 0, size);
                    memcpy(merged_hnsw->linkLists_[id + hnsw1_elements], hnsw2->linkLists_[id], size);
                    for (int i = 1; i <= level; i++) {
                        hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id + hnsw1_elements, i);
                        int size = merged_hnsw->getListCount(data);
                        hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
                        for (int j = 0; j < size; j++) {
                            datal[j] += hnsw1_elements;
                        }
                    }     
                }
                else {
                    cout << "no enough space for " << id << " ";
                }
            }
        }
    }

    cout << "Level 1 to level max copying time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    cout << endl;


    std::vector<std::vector<int>> nodes_level(merged_hnsw->maxlevel_ + 1);
    for (int id = 0; id < cur_elements; id++) {
        if (merged_hnsw->linkLists_[id] != nullptr) {
            int level = merged_hnsw->element_levels_[id];
            for (int i = level; i >= 1; i--) {
                nodes_level[i].push_back(id);
            }
        }
    }

    std::unordered_set<int> level1_nodes;
    for (int i = 0; i < nodes_level[1].size(); i++) {
        level1_nodes.insert(nodes_level[1][i]);
    }


    timer.reset();

    DistanceGraph graph_U0(cur_elements, 2 * M);

    #pragma omp parallel
    {
        #pragma omp for
        for (int id = 0; id < hnsw1_elements; id++)
        {
            hnswlib::linklistsizeint* data = hnsw1->get_linklist0(id);
            int size = hnsw1->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);

            for (int j = 0; j < size; j++) 
            {
                int neighbor_id = datal[j];
                float dist = hnsw1->fstdistfunc_(hnsw1->getDataByInternalId(id), hnsw1->getDataByInternalId(neighbor_id), hnsw1->dist_func_param_);
                graph_U0.addNeighbor(id, neighbor_id, dist);
            }
        }

        #pragma omp for
        for (int id = 0; id < hnsw2_elements; id++)
        {
            hnswlib::linklistsizeint* data = hnsw2->get_linklist0(id);
            int size = hnsw2->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);

            for (int j = 0; j < size; j++) 
            {
                int neighbor_id = datal[j];
                float dist = hnsw2->fstdistfunc_(hnsw2->getDataByInternalId(id), hnsw2->getDataByInternalId(neighbor_id), hnsw2->dist_func_param_);
                graph_U0.addNeighbor(id + hnsw1_elements, neighbor_id + hnsw1_elements, dist);
            }
        }   
    }

    cout << "Level 0 graph_U0 time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    int line = 32;   int pop = 8;   int add = 8;
    cout << "Level 0 parameters: Above " << line << ", pop " << pop << " and add " << add << endl;

    timer.reset();

    #pragma omp parallel for
    for (int id = 0; id < cur_elements; id++) {
        if (graph_U0.getNeighbors(id).size() > line) {
            for (int i = 0; i < pop; i++) {
                graph_U0.getNeighbors(id).pop();
            }
        }
    }

    cout << "Level 0 poping time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    timer.reset();

    #pragma omp parallel 
    {
        #pragma omp for
        for (size_t id = 0; id < hnsw1_elements; id++) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw2->searchKnn(data + id * dim, add);
            while(result.size() != 0) {
                graph_U0.addNeighbor(id, result.top().second + hnsw1_elements, result.top().first);
                result.pop();
            }
        }

        #pragma omp for
        for (size_t id = hnsw1_elements; id < cur_elements; id++) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw1->searchKnn(data + id * dim, add);
            while(result.size() != 0) {
                graph_U0.addNeighbor(id, result.top().second, result.top().first);
                result.pop();
            }
        }
    }

    cout << "Level 0 adding time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

    // timer.reset();

    // DistanceGraph graph_U1(nodes_level[1].size(), M);

    // for (int i = 0; i < nodes_level[1].size(); i++) {
    //     int id = nodes_level[1][i];

    //     hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, 1);
    //     int size = merged_hnsw->getListCount(data);
    //     hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //     for (int j = 0; j < size; j++) {
    //         float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(datal[j]), merged_hnsw->dist_func_param_);
    //         graph_U1.addNeighbor(i, datal[j], dist);
    //     }
    // }

    // for (int i = 0; i < nodes_level[1].size(); i++) {
    //     int id = nodes_level[1][i];
    //     auto pq = graph_U0.getNeighborsCopy(id);
    //     for (int j = 0; j < add; j++) {
    //         int n_id = pq.top().first;
    //         if (level1_nodes.find(n_id) != level1_nodes.end()) {
    //             float dist = pq.top().second;
    //             if (pq.size() == M) {
    //                 graph_U1.getNeighbors(i).pop();
    //                 graph_U1.getNeighbors(i).pop();
    //             }
    //             graph_U1.addNeighbor(i, n_id, dist);
    //         }
    //         pq.pop();
    //     }
    // }

    // for (int i = 0; i < nodes_level[1].size(); i++) {
    //     int id = nodes_level[1][i];

    //     auto neighbors = graph_U1.getNeighbors(i);
    //     int size = neighbors.size();
    //     hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, 1);
    //     merged_hnsw->setListCount(data, size);
    //     hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //     for (int idx = 0; idx < size; idx++) {
    //         datal[idx] = neighbors.top().first;
    //         neighbors.pop();
    //     }
    // }

    // cout << "Level 1 construction time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

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
        // cout << "1" << endl;
        
        c = 0;
        // #pragma omp parallel for
        for (size_t u = 0; u < graph_U0.getElementsNum(); u++) 
        {
            std::vector<size_t> neighbors(U[u].begin(), U[u].end());
            for (size_t i = 0; i < neighbors.size(); i++) 
            {
                for (size_t j = i + 1; j < neighbors.size(); j++) 
                {
                    size_t si = neighbors[i];
                    size_t sj = neighbors[j];
                    if ((si < hnsw1_elements && sj >= hnsw1_elements) || 
                        (si >= hnsw1_elements && sj < hnsw1_elements))
                    {
                        float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(si), merged_hnsw->getDataByInternalId(sj), merged_hnsw->dist_func_param_);

                        graph_U0.updateNN(si, sj, dist);
                        graph_U0.updateNN(sj, si, dist);
                    }
                }
            }
        }
        iteration++;
        // cout << "2" << endl;

    } while (c != 0 && iteration < 0);

    float loopTime = timer.getElapsedTimeSeconds();
    cout << "Level 0 looping time: " << loopTime << " seconds \t";
    cout << iteration << " iterations with " << (float)loopTime / iteration << " average time" << endl;

    
    for (int id = 0; id < cur_elements; id++) {
        while (graph_U0.getNeighbors(id).size() > 32) {
            graph_U0.getNeighbors(id).pop();
        }
    }


    // Level 0 neighbors loading
    timer.reset();

    #pragma omp parallel for
    for (int id = 0; id < cur_elements; id++)
    {
        // getNeighborsByHeuristicRevised(merged_hnsw, graph_U0.getNeighbors(id), M * 2);
        auto neighbors = graph_U0.getNeighbors(id);
        int size = neighbors.size();
        hnswlib::linklistsizeint* data = merged_hnsw->get_linklist0(id);
        merged_hnsw->setListCount(data, size);
        hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
        for (int idx = 0; idx < size; idx++) {
            datal[idx] = neighbors.top().first;
            neighbors.pop();
        }
    }

    cout << "Level 0 neighbors resetting time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    cout << endl;



    // Compare search time and recall
    int correct = 0;
    timer.reset();

    std::vector<hnswlib::labeltype> record(cur_elements);

    #pragma omp parallel for
    for (int i = 0; i < cur_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = merged_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        record[i] = label;
    }

    for (int i = 0; i < cur_elements; i++) {
        if (record[i] == i) {
            correct++;
        }
    }

    cout << "Merged graph search time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    float recall = (float)correct / cur_elements;
    std::cout << "Recall of Merged graph: " << recall << "\n";
    cout << endl;

}