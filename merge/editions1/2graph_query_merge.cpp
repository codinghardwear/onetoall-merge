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
int dim = 128;   int max_elements = 20000;
int M = 16;     int ef_construction = 200;      int ef_search = 10;

hnswlib::L2Space space(dim);

Timer timer;

int main() {

    int hnsw1_elements = max_elements / 2;    
    int hnsw2_elements = max_elements / 2;
    int cur_elements = hnsw1_elements + hnsw2_elements;

    std::uniform_real_distribution<> distrib_real;
    std::mt19937 rng;  rng.seed(1);
    float* data1 = new float[hnsw1_elements * dim];
    for (int i = 0; i < hnsw1_elements * dim; i++) {
        data1[i] = distrib_real(rng); 
    }
    rng.seed(2);
    float* data2 = new float[hnsw2_elements * dim];
    for (int i = 0; i < hnsw2_elements * dim; i++) {
        data2[i] = distrib_real(rng); 
    }
    omp_set_num_threads(4);

    std::string sub_hnsw1_path = "sub_hnsw1_10k_s1.bin";
    hnswlib::HierarchicalNSW<float>* sub_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, sub_hnsw1_path);        
    cout << "Sub hnsw1 loaded \t";
    std::string sub_hnsw2_path = "sub_hnsw2_10k_s2.bin";
    hnswlib::HierarchicalNSW<float>* sub_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, sub_hnsw2_path);        
    cout << "Sub hnsw2 loaded" << endl;       cout << endl;

    hnswlib::HierarchicalNSW<float>* merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    merged_hnsw->cur_element_count = cur_elements;
    merged_hnsw->maxlevel_ = sub_hnsw1->maxlevel_;
    merged_hnsw->enterpoint_node_ = sub_hnsw1->enterpoint_node_;
    merged_hnsw->ef_ = ef_search;

    cout << "-----MERGED GRAPH-----" << endl;
    cout << "25k elements,   128 dim,   16 M,   200 ef_construction,   10 ef_search." << endl;
    cout << endl;

    timer.reset();
    for (int id = 0; id < cur_elements; id++) {
        // Set data and label
        memset(merged_hnsw->data_level0_memory_ + id * merged_hnsw->size_data_per_element_ + merged_hnsw->offsetLevel0_, 0, merged_hnsw->size_data_per_element_);

        if (id < hnsw1_elements) {
            memcpy(merged_hnsw->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
            memcpy(merged_hnsw->getDataByInternalId(id), sub_hnsw1->getDataByInternalId(id), merged_hnsw->data_size_);
        }
        else {
            memcpy(merged_hnsw->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
            memcpy(merged_hnsw->getDataByInternalId(id), sub_hnsw2->getDataByInternalId(id - hnsw1_elements), merged_hnsw->data_size_);
        }
    }
    cout << "Level 0 copying time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    timer.reset();
    DistanceGraph graph_U0(cur_elements, 2 * M);
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
        for (int id = 0; id < hnsw2_elements; id++) {
            hnswlib::linklistsizeint* data = sub_hnsw2->get_linklist0(id);
            int size = sub_hnsw2->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
            for (int j = 0; j < size; j++) {
                int neighbor_id = datal[j];
                float dist = sub_hnsw2->fstdistfunc_(sub_hnsw2->getDataByInternalId(id), sub_hnsw2->getDataByInternalId(neighbor_id), sub_hnsw2->dist_func_param_);
                graph_U0.addNeighbor(id + hnsw1_elements, neighbor_id + hnsw1_elements, dist);
            }
        }   
    }
    cout << "Level 0 graph_U0 time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    timer.reset();
    int add = 8;
    #pragma omp parallel 
    {
        #pragma omp for
        for (int id = 0; id < hnsw1_elements; id++) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = sub_hnsw2->searchKnn(sub_hnsw1->getDataByInternalId(id), add);
            while(result.size() != 0) {
                if (graph_U0.getNeighbors(id).size() == 2 * M)
                    graph_U0.getNeighbors(id).pop();
                graph_U0.addNeighbor(id, result.top().second + hnsw1_elements, result.top().first);
                result.pop();
            }
        }
        #pragma omp for
        for (int id = hnsw1_elements; id < cur_elements; id++) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = sub_hnsw1->searchKnn(sub_hnsw2->getDataByInternalId(id - hnsw1_elements), add);
            while(result.size() != 0) {
                if (graph_U0.getNeighbors(id).size() == 2 * M)
                    graph_U0.getNeighbors(id).pop();
                graph_U0.addNeighbor(id, result.top().second, result.top().first);
                result.pop();
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
                    if ((si < hnsw1_elements && sj >= hnsw1_elements) || (si >= hnsw1_elements && sj < hnsw1_elements)) {
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
    for (int id = 0; id < cur_elements; id++)
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
        for (int id = 0; id < hnsw1_elements; id++) {
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

        #pragma omp for
        for (int id = 0; id < hnsw2_elements; id++)
        {
            int level = sub_hnsw2->element_levels_[id];
            merged_hnsw->element_levels_[id + hnsw1_elements] = level;
            if (level == 0)
                merged_hnsw->linkLists_[id + hnsw1_elements] = nullptr;
            else
            {
                int size = merged_hnsw->size_links_per_element_ * level;
                merged_hnsw->linkLists_[id + hnsw1_elements] = (char *) malloc(size);

                if (merged_hnsw->linkLists_[id] != nullptr) {
                    memcpy(merged_hnsw->linkLists_[id + hnsw1_elements], sub_hnsw2->linkLists_[id], size);
                    for (int i = 1; i <= level; i++) {
                        hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id + hnsw1_elements, i);
                        int size = merged_hnsw->getListCount(data);
                        hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
                        for (int j = 0; j < size; j++) {
                            datal[j] += hnsw1_elements;
                        }
                    }     
                }
                else
                    cout << "no enough space for " << id << " ";
            }
        }
    }

    cout << "Level 1 to level max copying time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    cout << endl;

    timer.reset();
    int correct = 0;
    std::vector<hnswlib::labeltype> record(cur_elements);
    #pragma omp parallel
    {
        #pragma omp for 
        for (int i = 0; i < hnsw1_elements; i++) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = merged_hnsw->searchKnn(data1 + i * dim, 1);
            hnswlib::labeltype label = result.top().second;
            record[i] = label;
        }
        #pragma omp for 
        for (int i = 0; i < hnsw2_elements; i++) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = merged_hnsw->searchKnn(data2 + i * dim, 1);
            hnswlib::labeltype label = result.top().second;
            record[i + hnsw1_elements] = label;
        }
    }
    for (int i = 0; i < 2 * hnsw1_elements; i++) {
        if (record[i] == i)
            correct++;
    }
    cout << "Merged graph search time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    float recall = (float)correct / cur_elements;
    std::cout << "Recall of Merged graph: " << recall << "\n";
    cout << endl;

    // merged_hnsw->saveIndex("merged_hnsw2_25k_2p.bin");
    // cout << "Saved successfully!" << endl;

    // delete[] data;
    // delete sub_hnsw1;
    // delete sub_hnsw2;
    // delete complete_hnsw;
    // delete merged_hnsw;

    return 0;
}