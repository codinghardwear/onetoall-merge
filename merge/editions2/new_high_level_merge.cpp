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

Timer timer;

hnswlib::L2Space space(dim);

void getNeighborsByHeuristicRevised(
    hnswlib::HierarchicalNSW<float>* hnsw,
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

    int hnsw1_elements = max_elements / 2;    int hnsw2_elements = max_elements / 2;
    int cur_elements = hnsw1_elements + hnsw2_elements;

    cout << cur_elements << " elements with " << dim << " dimensions" << endl;
    cout << endl;

    hnswlib::HierarchicalNSW<float>* hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, hnsw1_elements, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, hnsw2_elements, M, ef_construction);

    std::string hnsw_path1 = "sub_hnsw1_25k.bin";
    std::string hnsw_path2 = "sub_hnsw2_25k.bin";

    hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path1);        cout << "graph1" << endl;
    hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path2);        cout << "graph2" << endl;
    cout << endl;

    hnswlib::HierarchicalNSW<float>* merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, cur_elements, M, ef_construction);

    std::string hnsw_path = "merged_hnsw_50k_2.bin";

    merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);       cout << "graph" << endl;
    cout << endl;

    std::uniform_real_distribution<> distrib_real;
    std::mt19937 rng;  rng.seed(0);
    float* data = new float[dim * cur_elements];
    for (size_t i = 0; i < dim * cur_elements; i++) {
        data[i] = distrib_real(rng);  
    }

    // cout << merged_hnsw->element_levels_[5545] << endl;

    // Level 1 to Level max
    std::vector<std::vector<int>> nodes_level(merged_hnsw->maxlevel_ + 1);
    for (int id = 0; id < cur_elements; id++) {
        if (merged_hnsw->linkLists_[id] != nullptr) {
            int level = merged_hnsw->element_levels_[id];
            for (int i = level; i >= 1; i--) {
                nodes_level[i].push_back(id);
            }
        }
    }

    // Level 1
    // No poping
    int add = 2;
    for (int level = 1; level <= merged_hnsw->maxlevel_; level++) {

        timer.reset();
        
        DistanceGraph graph_Ul(nodes_level[level].size(), M);

        for (int i = 0; i < nodes_level[level].size(); i++) {
            int id = nodes_level[level][i];

            hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, level);
            int size = merged_hnsw->getListCount(data);
            // if (size == 0)  continue;
            hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
            for (int j = 0; j < size; j++) {
                float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(datal[j]), merged_hnsw->dist_func_param_);
                graph_Ul.addNeighbor(i, datal[j], dist);
            }
        }

        for (int i = 0; i < nodes_level[level].size(); i++) {
            int id = nodes_level[level][i];
            if (id < hnsw1_elements) {
                std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw2->searchKnnTargetLayer(data + id * dim, add, level);
                while(result.size() != 0) {
                    graph_Ul.addNeighbor(i, result.top().second + hnsw1_elements, result.top().first);
                    result.pop();
                }
            }
            else {
                std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw1->searchKnnTargetLayer(data + id * dim, add, level);
                while(result.size() != 0) {
                    graph_Ul.addNeighbor(i, result.top().second, result.top().first);
                    result.pop();
                }
            }
        }

        cout << "Level " << level << " graph U1 and adding time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


        timer.reset();

        int iteration = 0;
        do {
            std::vector<std::unordered_set<int>> U(graph_Ul.getElementsNum());
            for (int i = 0; i < graph_Ul.getElementsNum(); i++) {
                auto pq = graph_Ul.getNeighbors(i);
                while (!pq.empty()) {
                    U[i].insert(pq.top().first);
                    for (int j = 0; j < nodes_level[level].size(); j++) {
                        if (nodes_level[level][j] == pq.top().first) {
                            U[j].insert(nodes_level[level][i]);
                            break;
                        }
                    }
                    pq.pop();
                }
            }

            c = 0;
            for (int u = 0; u < graph_Ul.getElementsNum(); u++) {
                std::vector<int> neighbors(U[u].begin(), U[u].end());
                for (int i = 0; i < neighbors.size(); i++) {
                    for (int j = i + 1; j < neighbors.size(); j++) {
                        int si = neighbors[i];
                        int sj = neighbors[j];

                        if ((si < hnsw1_elements && sj >= hnsw1_elements) || 
                            (si >= hnsw1_elements && sj < hnsw1_elements)) {
                            float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(si), merged_hnsw->getDataByInternalId(sj), merged_hnsw->dist_func_param_);

                            int si_idx = -1;     int sj_idx = -1;
                            for (int idx = 0; idx < nodes_level[level].size(); idx++) {
                                if (nodes_level[level][idx] == si)
                                    si_idx = idx;
                                if (nodes_level[level][idx] == sj)
                                    sj_idx = idx;
                            }
                            graph_Ul.updateNN(si_idx, sj, dist);
                            graph_Ul.updateNN(sj_idx, si, dist);
                        }
                    }
                }
            }

            iteration++;

        } while (c != 0 && iteration < 0);

        float loopTime = timer.getElapsedTimeSeconds();
        cout << "Level " << level << " looping time: " << loopTime << " seconds \t";
        cout << iteration << " iterations with " << (float)loopTime / iteration << " average time" << endl;

        
        timer.reset();

        for (int i = 0; i < nodes_level[level].size(); i++) {
            int id = nodes_level[level][i];
            getNeighborsByHeuristicRevised(merged_hnsw, graph_Ul.getNeighbors(i), M);
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

    cout << endl;
    merged_hnsw->ef_ = ef_search;
    cout << "The ef search is " << ef_search << endl;

    // Compare search time and recall
    timer.reset();
    int correct = 0;
    std::vector<hnswlib::labeltype> record(cur_elements);
    #pragma omp parallel for
    for (int i = 0; i < cur_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = merged_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        record[i] = label;
    }
    for (int i = 0; i < cur_elements; i++) {
        if (record[i] == i)
            correct++;
    }
    cout << "Merged graph search time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    float recall = (float)correct / cur_elements;
    std::cout << "Recall of Merged graph: " << recall << "\n";
    cout << endl;

    delete merged_hnsw;
}