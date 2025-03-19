#include "../hnsw/hnswlib/hnswlib.h"
#include <omp.h>
#include <chrono>
#include <thread>
#include <algorithm>

#include "timer.h"
#include "graph.h"
#include "multi_thread.h"

using std::cout;
using std::endl;

// Global c
int c;

// Construction parameters
int dim = 128;   int maxElementsNum = 100000;
int M = 16;     int ef_construction = 200;

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

    int graph1ElementsNum = maxElementsNum / 2;    int graph2ElementsNum = maxElementsNum / 2;
    int curElementsNum = graph1ElementsNum + graph2ElementsNum;

    cout << curElementsNum << " elements with " << dim << " dimensions" << endl;
    cout << endl;

    hnswlib::HierarchicalNSW<float>* hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, graph1ElementsNum, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, graph2ElementsNum, M, ef_construction);

    std::string hnsw_path1 = "sub_hnsw1_50k.bin";
    std::string hnsw_path2 = "sub_hnsw2_50k.bin";

    hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path1);        cout << "graph1" << endl;
    hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path2);        cout << "graph2" << endl;

    std::uniform_real_distribution<> distrib_real;
    std::mt19937 rng;  rng.seed(0);
    float* data = new float[dim * curElementsNum];
    for (size_t i = 0; i < dim * curElementsNum; i++) {
        data[i] = distrib_real(rng);  
    }

    hnswlib::HierarchicalNSW<float>* merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, maxElementsNum, M, ef_construction);
    merged_hnsw->cur_element_count = curElementsNum;
    merged_hnsw->maxlevel_ = hnsw1->maxlevel_;
    merged_hnsw->enterpoint_node_ = hnsw1->enterpoint_node_;

    timer.reset();

    for (size_t id = 0; id < curElementsNum; id++) {
            // Set data and label
        memset(merged_hnsw->data_level0_memory_ + id * merged_hnsw->size_data_per_element_ + merged_hnsw->offsetLevel0_, 0, merged_hnsw->size_data_per_element_);

        if (id < graph1ElementsNum) {
            memcpy(merged_hnsw->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
            memcpy(merged_hnsw->getDataByInternalId(id), hnsw1->getDataByInternalId(id), merged_hnsw->data_size_);
        }
        else {
            memcpy(merged_hnsw->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
            memcpy(merged_hnsw->getDataByInternalId(id), hnsw2->getDataByInternalId(id - graph1ElementsNum), merged_hnsw->data_size_);
        }
    }

    cout << "Level 0 copying time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    timer.reset();

    DistanceGraph graph_U0(curElementsNum, 2 * M);

    #pragma omp parallel for
    for (int id = 0; id < graph1ElementsNum; id++)
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

    #pragma omp parallel for
    for (int id = 0; id < graph2ElementsNum; id++)
    {
        hnswlib::linklistsizeint* data = hnsw2->get_linklist0(id);
        int size = hnsw2->getListCount(data);
        hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);

        for (int j = 0; j < size; j++) 
        {
            int neighbor_id = datal[j];
            float dist = hnsw2->fstdistfunc_(hnsw2->getDataByInternalId(id), hnsw2->getDataByInternalId(neighbor_id), hnsw2->dist_func_param_);
            graph_U0.addNeighbor(id + graph1ElementsNum, neighbor_id + graph1ElementsNum, dist);
        }
    }

    cout << "Level 0 graph_U0 time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    int line = 32;   int pop = 8;   int add = 8;
    cout << "Level 0 parameters: Above " << line << ", pop " << pop << " and add " << add << endl;

    timer.reset();

    for (int id = 0; id < curElementsNum; id++) {
        if (graph_U0.getNeighbors(id).size() > line) {
            for (int i = 0; i < pop; i++) {
                graph_U0.getNeighbors(id).pop();
            }
        }
    }

    cout << "Level 0 poping time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    timer.reset();

    hnsw1->ef_ = 200;
    hnsw2->ef_ = 200;

    #pragma omp parallel for
    for (size_t id = 0; id < graph1ElementsNum; id++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw2->searchKnn(data + id * dim, add);
        while(result.size() != 0) {
            graph_U0.addNeighbor(id, result.top().second + graph1ElementsNum, result.top().first);
            result.pop();
        }
    }

    #pragma omp parallel for
    for (size_t id = graph1ElementsNum; id < curElementsNum; id++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw1->searchKnn(data + id * dim, add);
        while(result.size() != 0) {
            graph_U0.addNeighbor(id, result.top().second, result.top().first);
            result.pop();
        }
    }

    cout << "Level 0 adding time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    // Level 0 neighbors loading
    timer.reset();

    for (int id = 0; id < curElementsNum; id++)
    {
        if (id == 0)    graph_U0.printGraph(0, 0);
        getNeighborsByHeuristicRevised(merged_hnsw, graph_U0.getNeighbors(id), M * 2);
        if (id == 0)    graph_U0.printGraph(0, 0);
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

    // Level 1 to level max
    timer.reset();

    for (size_t id = 0; id < graph1ElementsNum; id++)
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
                cout << "no enough space";
            }
        }
    }

    for (int id = 0; id < graph2ElementsNum; id++)
    {
        int level = hnsw2->element_levels_[id];
        merged_hnsw->element_levels_[id + graph1ElementsNum] = level;
        if (level == 0)
            merged_hnsw->linkLists_[id + graph1ElementsNum] = nullptr;
        else
        {
            int size = merged_hnsw->size_links_per_element_ * level;
            merged_hnsw->linkLists_[id + graph1ElementsNum] = (char*) malloc(size);

            if (merged_hnsw->linkLists_[id] != nullptr) {
                memset(merged_hnsw->linkLists_[id + graph1ElementsNum], 0, size);
                memcpy(merged_hnsw->linkLists_[id + graph1ElementsNum], hnsw2->linkLists_[id], size);
                for (int i = 1; i <= level; i++) {
                    hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id + graph1ElementsNum, i);
                    int size = merged_hnsw->getListCount(data);
                    hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
                    for (int j = 0; j < size; j++) {
                        datal[j] += graph1ElementsNum;
                    }
                }     
            }
            else {
                cout << "no enough space";
            }
        }
    }

    cout << "Level 1 to level max copying time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    cout << endl;

    merged_hnsw->ef_ = 150;
    int correct = 0;
    timer.reset();

    for (int i = 0; i < curElementsNum; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = merged_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) {
            correct++;
        }
    }

    cout << "Merged graph search time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    float recall = (float)correct / curElementsNum;
    std::cout << "Recall of merged graph: " << recall << "\n";
    cout << endl;

    // merged_hnsw->saveIndex("merged_hnsw_20k.bin");

    return 0;
}
