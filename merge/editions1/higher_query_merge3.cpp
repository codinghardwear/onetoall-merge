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

Timer timer;

hnswlib::L2Space space(dim);

int main() {

    int graph1ElementsNum = maxElementsNum / 2;    int graph2ElementsNum = maxElementsNum / 2;
    int curElementsNum = graph1ElementsNum + graph2ElementsNum;

    cout << curElementsNum << " elements with " << dim << " dimensions" << endl;
    cout << endl;

    hnswlib::HierarchicalNSW<float>* hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, graph1ElementsNum, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, graph2ElementsNum, M, ef_construction);

    std::string hnsw_path1 = "sub_hnsw1_50k.bin";
    std::string hnsw_path2 = "sub_hnsw2_50k.bin";

    cout << "graph1" << endl;
    hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path1);
    cout << "graph2" << endl;
    hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path2);
    cout << endl;

    std::uniform_real_distribution<> distrib_real;
    std::mt19937 rng;  rng.seed(0);
    float* data = new float[dim * curElementsNum];
    for (size_t i = 0; i < dim * curElementsNum; i++) {
        data[i] = distrib_real(rng);  
    }

    hnswlib::HierarchicalNSW<float>* merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, maxElementsNum, M, ef_construction);
    merged_hnsw->cur_element_count = curElementsNum;
    merged_hnsw->maxlevel_ = hnsw1->maxlevel_;
    merged_hnsw->enterpoint_node_ = hnsw1->enterpoint_node_ ;

    timer.reset();

    for (int id = 0; id < curElementsNum; id++) {
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

    cout << "Level 0 graph U0 time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


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

    for (int id = 0; id < graph1ElementsNum; id++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw2->searchKnn(data + id * dim, add);
        while(result.size() != 0) {
            graph_U0.addNeighbor(id, result.top().second + graph1ElementsNum, result.top().first);
            result.pop();
        }
    }

    for (int id = graph1ElementsNum; id < curElementsNum; id++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw1->searchKnn(data + id * dim, add);
        while(result.size() != 0) {
            graph_U0.addNeighbor(id, result.top().second, result.top().first);
            result.pop();
        }
    }

    cout << "Level 0 adding time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

    timer.reset();

    int iteration = 0;
    do {
        std::vector<std::unordered_set<int>> U(graph_U0.getElementsNum());
        for (int i = 0; i < graph_U0.getElementsNum(); i++) {
            auto pq = graph_U0.getNeighborsCopy(i);
            while (!pq.empty()) {
                U[i].insert(pq.top().first);
                U[pq.top().first].insert(i);
                pq.pop();
            }
        }
        
        c = 0;
        for (int u = 0; u < graph_U0.getElementsNum(); u++) 
        {
            std::vector<int> neighbors(U[u].begin(), U[u].end());
            for (int i = 0; i < neighbors.size(); i++) 
            {
                for (int j = i + 1; j < neighbors.size(); j++) 
                {
                    int si = neighbors[i];
                    int sj = neighbors[j];
                    if ((si < graph1ElementsNum && sj >= graph1ElementsNum) || 
                        (si >= graph1ElementsNum && sj < graph1ElementsNum))
                    {
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
    cout << endl;

    // Level 0 neighbors loading
    timer.reset();

    for (int id = 0; id < curElementsNum; id++)
    {
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


    timer.reset();

    for (int id = 0; id < graph1ElementsNum; id++)
    {
        merged_hnsw->element_levels_[id] = hnsw1->element_levels_[id];
        if (merged_hnsw->element_levels_[id] == 0)
            merged_hnsw->linkLists_[id] = nullptr;
        else
        {
            int size = merged_hnsw->size_links_per_element_ * merged_hnsw->element_levels_[id];
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

    merged_hnsw->saveIndex("merged_hnsw_100k.bin");

    return 0;

    // Level 1 to Level max
    std::vector<std::vector<int>> nodes_level(merged_hnsw->maxlevel_ + 1);
    for (int id = 0; id < curElementsNum; id++) {
        if (merged_hnsw->linkLists_[id] != nullptr) {
            int level = merged_hnsw->element_levels_[id];
            nodes_level[level].push_back(id);
        }
    }

    // Level 1
    // No poping
    timer.reset();

    DistanceGraph graph_U1(nodes_level[1].size(), M);

    for (int i = 0; i < nodes_level[1].size(); i++) {
        int id = nodes_level[1][i];

        hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, 1);
        int size = merged_hnsw->getListCount(data);
        hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
        for (int j = 0; j < size; j++) {
            float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(datal[j]), merged_hnsw->dist_func_param_);
            graph_U1.addNeighbor(i, datal[j], dist);
        }

        if (id < graph1ElementsNum) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw2->searchKnnTargetLayer(data + id * dim, 8, 1);
            while(result.size() != 0) {
                graph_U1.addNeighbor(i, result.top().second + graph1ElementsNum, result.top().first);
                result.pop();
            }
        }
        else {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw1->searchKnnTargetLayer(data + id * dim, 8, 1);
            while(result.size() != 0) {
                graph_U1.addNeighbor(i, result.top().second, result.top().first);
                result.pop();
            }
        }
    }

    cout << "Level 1 graph U1 and adding time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    timer.reset();

    iteration = 0;
    do {
        std::vector<std::unordered_set<int>> U(graph_U1.getElementsNum());
        for (int i = 0; i < graph_U1.getElementsNum(); i++) {
            auto pq = graph_U1.getNeighbors(i);
            while (!pq.empty()) {
                U[i].insert(pq.top().first);
                U[pq.top().first].insert(nodes_level[1][i]);
                pq.pop();
            }
        }
        
        c = 0;
        for (int u = 0; u < graph_U1.getElementsNum(); u++) 
        {
            std::vector<int> neighbors(U[u].begin(), U[u].end());
            for (int i = 0; i < neighbors.size(); i++) 
            {
                for (int j = i + 1; j < neighbors.size(); j++) 
                {
                    int si = neighbors[i];
                    int sj = neighbors[j];
                    if ((si < graph1ElementsNum && sj >= graph1ElementsNum) || 
                        (si >= graph1ElementsNum && sj < graph1ElementsNum))
                    {
                        float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(si), merged_hnsw->getDataByInternalId(sj), merged_hnsw->dist_func_param_);

                        int si_idx = -1;     int sj_idx = -1;
                        for (int idx = 0; idx < nodes_level[1].size(); idx++) {
                            if (nodes_level[1][idx] == si) {
                                si_idx = idx;
                            }
                            if (nodes_level[1][idx] == sj) {
                                sj_idx = idx;
                            }
                        }
                        if (si_idx == -1 || sj_idx == -1) {
                            cout << "ERROR in finding index" << endl;
                        }

                        graph_U1.updateNN(si_idx, sj, dist);
                        graph_U1.updateNN(sj_idx, si, dist);
                    }
                }
            }
        }
        iteration++;

    } while (c != 0 && iteration < 0);

    loopTime = timer.getElapsedTimeSeconds();
    cout << "Level 1 looping time: " << loopTime << " seconds" << endl;
    cout << iteration << " iterations with " << (float)loopTime / iteration << " average time" << endl;
    cout << endl;


    timer.reset();

    for (int i = 0; i < nodes_level[1].size(); i++) {
        int id = nodes_level[1][i];

        auto neighbors = graph_U1.getNeighbors(i);
        int size = neighbors.size();
        hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, 1);
        merged_hnsw->setListCount(data, size);
        hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
        for (int idx = 0; idx < size; idx++) {
            datal[idx] = neighbors.top().first;
            neighbors.pop();
        }
    }

    cout << "Level 1 neighors resetting time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

    // merged_hnsw->saveIndex("merged_hnsw_100k.bin");

    // Compare search time and recall
    correct = 0;
    timer.reset();

    for (int i = 0; i < curElementsNum; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = merged_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) {
            correct++;
        }
        // else {
        //     merged_hnsw->enterpoint_node_ = hnsw2->enterpoint_node_ + graph1ElementsNum;
        //     result = merged_hnsw->searchKnn(data + i * dim, 1);
        //     label = result.top().second;
        //     if (label == i) {
        //         correct++;
        //     }
        //     merged_hnsw->enterpoint_node_ = hnsw1->enterpoint_node_;
        // }
    }

    cout << "Merged graph search time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    recall = (float)correct / curElementsNum;
    std::cout << "Recall of merged graph: " << recall << "\n";
    cout << endl;

    delete hnsw1;
    delete hnsw2;
    delete merged_hnsw;
}