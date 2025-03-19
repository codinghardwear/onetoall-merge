#include "../hnsw/hnswlib/hnswlib.h"

#include <chrono>
#include <thread>
#include <algorithm>

#include "timer.h"
#include "multi_thread.h"
#include "graph.h"

using std::cout;
using std::endl;

// Global c
int c;

// Construction parameters
int dim = 128;   int maxElementsNum = 100000;
int M = 16;     int ef_construction = 200;

hnswlib::L2Space space(dim);

int main() {

    int graph1ElementsNum = maxElementsNum/2;    int graph2ElementsNum = maxElementsNum/2;
    int curElementsNum = graph1ElementsNum + graph2ElementsNum;

    cout << curElementsNum << " elements with " << dim << " dimensions" << endl;
    cout << endl;

    hnswlib::HierarchicalNSW<float>* hnsw = new hnswlib::HierarchicalNSW<float>(&space, maxElementsNum, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, graph1ElementsNum, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, graph2ElementsNum, M, ef_construction);

    std::string hnsw_path1 = "hnsw1_5w.bin";
    std::string hnsw_path2 = "hnsw2_5w.bin";

    cout << "graph1" << endl;
    hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path1);
    cout << "graph2" << endl;
    hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path2);

    std::uniform_real_distribution<> distrib_real;
    std::mt19937 rng;  rng.seed(0);
    float* data = new float[dim * curElementsNum];
    for (size_t i = 0; i < dim * curElementsNum; i++) {
        data[i] = distrib_real(rng);  
    }
    // float* data1 = new float[dim * graph1ElementsNum];
    // for (size_t i = 0; i < dim * graph1ElementsNum; i++) {
    //     data1[i] = data[i]; 
    // }
    // float* data2 = new float[dim * graph2ElementsNum];
    // for (size_t i = dim * graph1ElementsNum; i < dim * curElementsNum; i++) {
    //     data2[i - dim * graph1ElementsNum] = data[i];
    // }

    // for (size_t i = 0; i < graph1ElementsNum; i++) {
    //     hnsw1->addPoint(data1 + i * dim, i);
    // }
    // for (size_t i = 0; i < graph2ElementsNum; i++) {
    //     hnsw2->addPoint(data2 + i * dim, i);
    // }

    hnswlib::HierarchicalNSW<float>* merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, maxElementsNum, M, ef_construction);
    merged_hnsw->cur_element_count = curElementsNum;
    merged_hnsw->maxlevel_ = hnsw1->maxlevel_;
    merged_hnsw->enterpoint_node_ = hnsw1->enterpoint_node_;

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

    Timer timer;
    timer.reset();

    DistanceGraph graph_U(curElementsNum, 2 * M);

    for (size_t id = 0; id < graph1ElementsNum; id++)
    {
        int* data = (int*) hnsw1->get_linklist0(id);
        int size = hnsw1->getListCount((hnswlib::linklistsizeint*) data);

        for (size_t j = 1; j <= size; j++) 
        {
            int neighbor_id = *(data + j);
            float dist = hnsw1->fstdistfunc_(hnsw1->getDataByInternalId(id), hnsw1->getDataByInternalId(neighbor_id), hnsw1->dist_func_param_);
            graph_U.addNeighbor(id, neighbor_id, dist);
        }
    }

    for (size_t id = 0; id < graph2ElementsNum; id++)
    {
        int* data = (int*) hnsw2->get_linklist0(id);
        int size = hnsw2->getListCount((hnswlib::linklistsizeint*) data);

        for (size_t j = 1; j <= size; j++) 
        {
            int neighbor_id = *(data + j);
            float dist = hnsw2->fstdistfunc_(hnsw2->getDataByInternalId(id), hnsw2->getDataByInternalId(neighbor_id), hnsw2->dist_func_param_);
            graph_U.addNeighbor(id + graph1ElementsNum, neighbor_id + graph1ElementsNum, dist);
        }
    }

    cout << "Graph U time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

    // graph_U.printGraph(0, 9);
    // graph_U.printGraph(99990, 100000);
    // graph_U.printGraph();

    // cout << hnsw1->getExternalLabel(100) << endl;

    int line = 32;
    int add = 8;
    cout << "cutoff line is " << line << " and add " << add << endl;

    timer.reset();

    for (size_t id = 0; id < curElementsNum; id++) {
        while (graph_U.getNeighbors(id).size() > line) {
            graph_U.getNeighbors(id).pop();
        }
    }

    cout << "Poping time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

    timer.reset();

    for (size_t id = 0; id < graph1ElementsNum; id++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw2->searchKnn(data + id * dim, add);
        while(result.size() != 0) {
            graph_U.addNeighbor(id, result.top().second + graph1ElementsNum, result.top().first);
            result.pop();
        }
    }

    for (size_t id = graph1ElementsNum; id < curElementsNum; id++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw1->searchKnn(data + id * dim, add);
        while(result.size() != 0) {
            graph_U.addNeighbor(id, result.top().second, result.top().first);
            result.pop();
        }
    }

    cout << "Adding time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

    // graph_U.printGraph(100, 100);
    // float dist = hnsw2->fstdistfunc_(hnsw1->getDataByInternalId(100), hnsw2->getDataByInternalId(27733), hnsw2->dist_func_param_);
    // cout << dist << endl;
    // dist = hnsw2->fstdistfunc_(hnsw1->getDataByInternalId(100), hnsw2->getDataByInternalId(45296), hnsw2->dist_func_param_);
    // cout << dist << endl;

    timer.reset();

    int iteration = 0;
    do {
        std::vector<std::unordered_set<size_t>> U(graph_U.getElementsNum());
        for (size_t i = 0; i < graph_U.getElementsNum(); i++) {
            auto pq = graph_U.getNeighbors(i);
            while (!pq.empty()) {
                U[i].insert(pq.top().first);
                U[pq.top().first].insert(i);
                pq.pop();
            }
        }

        std::vector<std::vector<size_t>> U_(graph_U.getElementsNum());
        for (int i = 0; i < U.size(); i++) {
            U_[i] = std::vector<size_t>(U[i].begin(), U[i].end());
        }
        
        c = 0;
        for (size_t u = 0; u < graph_U.getElementsNum(); u++) {
            // for (size_t i = 0; i < U_[u].size(); i++) {
            //     size_t one_hop = U_[u][i];
            //     for (size_t j = 0; j < U_[one_hop].size(); j++) {
            //         size_t v1 = u;
            //         size_t v2 = U_[one_hop][j];

            //         float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(v1), merged_hnsw->getDataByInternalId(v2), merged_hnsw->dist_func_param_);
            //         graph_U.updateNN(v1, v2, dist);
            //         graph_U.updateNN(v2, v1, dist);
            //     }
            // }
            for (size_t v : U_[u]) {
                for (size_t w : U_[v]) {
                    if (u != w) {
                        float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(u), merged_hnsw->getDataByInternalId(w), merged_hnsw->dist_func_param_);
                        graph_U.updateNN(u, w, dist);
                    }
                }
            }
        }
        iteration++;

    } while (c != 0 && iteration < 0);

    float loopTime = timer.getElapsedTimeSeconds();
    cout << "Looping time: " << loopTime << " seconds" << endl;
    cout << iteration << " iterations with " << (float)loopTime / iteration << " average time" << endl;
    cout << endl;

    // timer.reset();

    // hnswlib::HierarchicalNSW<float>* merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, maxElementsNum, M, ef_construction);
    // merged_hnsw->cur_element_count = curElementsNum;
    // merged_hnsw->maxlevel_ = hnsw1->maxlevel_;
    // merged_hnsw->enterpoint_node_ = hnsw1->enterpoint_node_;

    // Level 0
    for (size_t id = 0; id < curElementsNum; id++)
    {
        // // Set data and label
        // memset(merged_hnsw->data_level0_memory_ + id * merged_hnsw->size_data_per_element_ + merged_hnsw->offsetLevel0_, 0, merged_hnsw->size_data_per_element_);

        // if (id < graph1ElementsNum) {
        //     memcpy(merged_hnsw->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
        //     memcpy(merged_hnsw->getDataByInternalId(id), hnsw1->getDataByInternalId(id), merged_hnsw->data_size_);
        // }
        // else {
        //     memcpy(merged_hnsw->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
        //     memcpy(merged_hnsw->getDataByInternalId(id), hnsw2->getDataByInternalId(id - graph1ElementsNum), merged_hnsw->data_size_);
        // }

        // Set neighbors
        auto neighbors = graph_U.getNeighbors(id);
        size_t neighborSize = neighbors.size();
        hnswlib::linklistsizeint* neighborSizePtr = merged_hnsw->get_linklist0(id);
        merged_hnsw->setListCount(neighborSizePtr, neighborSize);
        hnswlib::tableint* neighborIdPtr = (hnswlib::tableint*) (neighborSizePtr + 1);
        for (size_t idx = 0; idx < neighborSize; idx++) {
            neighborIdPtr[idx] = neighbors.top().first;
            neighbors.pop();
        }

        // std::vector<int> neighbors;
    }

    // Other levels
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
            // else {
            //     cout << "no enough space";
            // }
        }
    }

    for (size_t id = 0; id < graph2ElementsNum; id++)
    {
        merged_hnsw->element_levels_[id + graph1ElementsNum] = hnsw2->element_levels_[id];
        if (merged_hnsw->element_levels_[id + graph1ElementsNum] == 0)
            merged_hnsw->linkLists_[id + graph1ElementsNum] = nullptr;
        else
        {
            size_t size = merged_hnsw->size_links_per_element_ * merged_hnsw->element_levels_[id + graph1ElementsNum];
            merged_hnsw->linkLists_[id + graph1ElementsNum] = (char*) malloc(size);

            if (merged_hnsw->linkLists_[id] != nullptr) {
                memset(merged_hnsw->linkLists_[id + graph1ElementsNum], 0, size);
                memcpy(merged_hnsw->linkLists_[id + graph1ElementsNum], hnsw2->linkLists_[id], size);     
            }
            // else {
            //     cout << "no enough space";
            // }

        }
    }

    // cout << "Saving time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

    // Compare search time and recall
    int correct = 0;
    timer.reset();
    for (int i = 0; i < curElementsNum; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = merged_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    cout << "Original search time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    float recall = (float)correct / curElementsNum;
    std::cout << "Recall of original graph: " << recall << "\n";
    cout << endl;

}