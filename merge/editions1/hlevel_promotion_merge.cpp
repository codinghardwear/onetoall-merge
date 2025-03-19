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

    hnswlib::HierarchicalNSW<float>* sub_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, hnsw1_elements, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* sub_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, hnsw2_elements, M, ef_construction);

    std::string hnsw_path1 = "sub_hnsw1_25k.bin";
    std::string hnsw_path2 = "sub_hnsw2_25k.bin";

    sub_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path1);        cout << "graph1" << endl;
    sub_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path2);        cout << "graph2" << endl;
    cout << endl;

    hnswlib::HierarchicalNSW<float>* merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, cur_elements, M, ef_construction);

    std::string hnsw_path = "merged_hnsw_50k_2.bin";

    merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);       cout << "graph" << endl;
    cout << endl;

    std::uniform_real_distribution<> distrib_real;
    std::mt19937 rng;  rng.seed(0);
    float* data = new float[dim * cur_elements];
    for (int i = 0; i < dim * cur_elements; i++) {
        data[i] = distrib_real(rng);  
    }
    cout << "1";

    std::unordered_set<int> level1_nodes;
    for (int id = 0; id < cur_elements; id++) {
        if (merged_hnsw->element_levels_[id] != 0 && merged_hnsw->linkLists_[id] != nullptr) {
            level1_nodes.insert(id);
        }
    }
    cout << "1";

    for (int id = 0; id < hnsw1_elements; id++) {
        if (merged_hnsw->element_levels_[id] != 0 && merged_hnsw->linkLists_[id] != nullptr) {
            DistanceGraph graph_U1(1, M);
            // std::unordered_set<int> neighors;
            hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, 1);
            int size = merged_hnsw->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
            cout << "1";
            for (int j = 0; j < size; j++) {
                int neighbor_id = datal[j];
                // neighbors.insert(neighbor_id);
                float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(neighbor_id), merged_hnsw->dist_func_param_);
                graph_U1.addNeighbor(0, neighbor_id, dist);
            }
            cout << "2";
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = sub_hnsw2->searchKnn(data + id * dim, 1);
            cout << "1";
            float dist = result.top().first;
            hnswlib::labeltype hnsw2_node = result.top().second + hnsw1_elements;
            // if (neighbors.find(hnsw2_node) == neighbors.end())
            if (size == M) {
                if (graph_U1.getNeighbors(0).top().second < dist) {
                    graph_U1.getNeighbors(0).pop();
                    graph_U1.addNeighbor(0, hnsw2_node, dist);
                } 
            }
            graph_U1.addNeighbor(0, hnsw2_node, dist);

            cout << "1";
            size = graph_U1.getNeighbors(0).size();
            for (int idx = size - 1; idx >= 0; idx--) {
                datal[idx] = graph_U1.getNeighbors(0).top().first;
                graph_U1.getNeighbors(0).pop();
            }
            cout << "1";

            if (level1_nodes.find(hnsw2_node) == level1_nodes.end()) {
                std::priority_queue<std::pair<float, hnswlib::labeltype>> result = sub_hnsw2->searchKnn(data + hnsw2_node * dim, 15);
                merged_hnsw->element_levels_[hnsw2_node] = 1;
                merged_hnsw->linkLists_[hnsw2_node] = (char *) malloc(merged_hnsw->size_links_per_element_);
                hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(hnsw2_node, 1);
                merged_hnsw->setListCount(data, 16);
                hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
                datal[0] = id;
                for (int i = 1; i < 16; i++) {
                    datal[i] = result.top().second;
                    result.pop();
                }
            }
        }  
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