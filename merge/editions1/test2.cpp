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

    int graph1ElementsNum = maxElementsNum / 2;    int graph2ElementsNum = maxElementsNum / 2;
    int curElementsNum = graph1ElementsNum + graph2ElementsNum;

    cout << curElementsNum << " elements with " << dim << " dimensions" << endl;
    cout << endl;

    hnswlib::HierarchicalNSW<float>* hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, graph1ElementsNum, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, graph2ElementsNum, M, ef_construction);

    std::string hnsw_path1 = "hnsw1_5w.bin";
    std::string hnsw_path2 = "hnsw2_5w.bin";

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

    // hnsw1->saveIndex("graph1_50k_M8.bin");
    // hnsw2->saveIndex("graph2_50k_M8.bin");

    hnswlib::HierarchicalNSW<float>* merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, maxElementsNum, M, ef_construction);
    merged_hnsw->cur_element_count = curElementsNum;
    merged_hnsw->maxlevel_ = hnsw1->maxlevel_;
    merged_hnsw->enterpoint_node_ = hnsw1->enterpoint_node_ ;
    merged_hnsw->dist_func_param_ = hnsw1->dist_func_param_;

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
        int level = hnsw2->element_levels_[id];
        merged_hnsw->element_levels_[id + graph1ElementsNum] = level;
        if (level == 0)
            merged_hnsw->linkLists_[id + graph1ElementsNum] = nullptr;
        else
        {
            size_t size = merged_hnsw->size_links_per_element_ * level;
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
            // else {
            //     cout << "no enough space";
            // }
        }
    }

    std::vector<int> level1_nodes;
    for (size_t id = 0; id < curElementsNum; id++) {
        if (merged_hnsw->linkLists_[id] != nullptr) {
            level1_nodes.push_back(id);
        }
    }

    // for (int i = 0; i < level1_nodes.size(); i++) {
    //     if (level1_nodes[i] == 3363) {
    //         cout << i << endl;
    //         break;
    //     }
    // }

    // cout << level1_nodes[218] << " " << level1_nodes[219] << endl;

    // hnswlib::linklistsizeint* data_1 = merged_hnsw->get_linklist(3367, 1);
    // int size_1 = merged_hnsw->getListCount(data_1);
    // hnswlib::tableint* datal_1 = (hnswlib::tableint*) (data_1 + 1);
    // for (int i = 0; i < size_1; i++) {
    //     cout << datal_1[i] << " ";
    //     float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(3367), merged_hnsw->getDataByInternalId(datal_1[i]), merged_hnsw->dist_func_param_);
    //     graph_U1.addNeighbor(0, datal_1[i], dist);
    // }

    // cout << endl;
    // cout << endl;

    // cout << level1_nodes.size() << endl;
    // cout << level1_nodes[0] << endl;
    // cout << level1_nodes[level1_nodes.size() - 1] << endl;

    // cout << hnsw1->element_levels_[11] << endl;

    DistanceGraph graph_U1(level1_nodes.size(), M);   int index = 0;

    for (size_t id : level1_nodes) {
        hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, 1);
        int size = merged_hnsw->getListCount(data);
        hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);

        for (int i = 0; i < size; i++) {
            float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(datal[i]), merged_hnsw->dist_func_param_);
            graph_U1.addNeighbor(index, datal[i], dist);
        }
        index++;
    }

    

    // int level3_node = 4373;

    // hnswlib::linklistsizeint* data_test = merged_hnsw->get_linklist(level3_node + graph1ElementsNum, 3);
    // int size = merged_hnsw->getListCount(data_test);
    // hnswlib::tableint* datal = (hnswlib::tableint*) (data_test + 1);
    // for (int i = 0; i < size; i++) {
    //     cout << datal[i] << " ";
    // }
    // cout << endl;

    // data_test = hnsw2->get_linklist(level3_node, 3);
    // size = hnsw2->getListCount(data_test);
    // datal = (hnswlib::tableint*) (data_test + 1);
    // for (int i = 0; i < size; i++) {
    //     cout << datal[i] << " ";
    // }
    // cout << endl;

    // size_ptr = hnsw2->get_linklist(level3_node, 1);
    // size = hnsw1->getListCount(size_ptr);
    // cout << size << endl;

    // for (int id = graph1ElementsNum; id < curElementsNum; id++) {
    //     int level = merged_hnsw->element_levels_[id];
    //     if (level == 3) {
    //         cout << id << endl;
    //         break;
    //     }
    // }
    // cout << hnsw2->element_levels_[4373] << endl;

    // std::unordered_map<int, int> table;

    // for (int id = graph1ElementsNum; id < curElementsNum; id++) {
    //     int level = merged_hnsw->element_levels_[id];
    //     if (table.find(level) == table.end()) {
    //         table.emplace(level, 1);
    //     }
    //     else {
    //         table[level] ++;
    //     }
    // }

    // for (auto iter = table.begin(); iter != table.end(); iter++) {
    //     cout << iter->first << ": " << iter->second << endl;
    // }

    // cout << endl;
    // table.clear();

    // for (int id = 0; id < graph2ElementsNum; id++) {
    //     int level = hnsw2->element_levels_[id];
    //     if (table.find(level) == table.end()) {
    //         table.emplace(level, 1);
    //     }
    //     else {
    //         table[level] ++;
    //     }
    // }

    // for (auto iter = table.begin(); iter != table.end(); iter++) {
    //     cout << iter->first << ": " << iter->second << endl;
    // }

    
    // Form level 0 neighbors

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


    int line = 12;   int pop = 8;   int add = 8;
    cout << "Above " << line << ", pop " << pop << " and add " << add << endl;

    timer.reset();

    for (size_t id = 0; id < curElementsNum; id++) {
        // while (graph_U.getNeighbors(id).size() > line) {
        //     graph_U.getNeighbors(id).pop();
        // }
        if (graph_U.getNeighbors(id).size() >= line) {
            for (int i = 0; i < pop; i++) {
                graph_U.getNeighbors(id).pop();
            }
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
        
        c = 0;
        for (size_t u = 0; u < graph_U.getElementsNum(); u++) 
        {
            std::vector<size_t> neighbors(U[u].begin(), U[u].end());
            for (size_t i = 0; i < neighbors.size(); i++) 
            {
                for (size_t j = i + 1; j < neighbors.size(); j++) 
                {
                    size_t si = neighbors[i];
                    size_t sj = neighbors[j];
                    if ((si < graph1ElementsNum && sj >= graph1ElementsNum) || 
                        (si >= graph1ElementsNum && sj < graph1ElementsNum))
                    {
                        float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(si), merged_hnsw->getDataByInternalId(sj), merged_hnsw->dist_func_param_);

                        graph_U.updateNN(si, sj, dist);
                        graph_U.updateNN(sj, si, dist);
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

    // Write level 0 neighbors 
    for (size_t id = 0; id < curElementsNum; id++)
    {
        auto neighbors = graph_U.getNeighbors(id);
        size_t neighborSize = neighbors.size();
        hnswlib::linklistsizeint* neighborSizePtr = merged_hnsw->get_linklist0(id);
        merged_hnsw->setListCount(neighborSizePtr, neighborSize);
        hnswlib::tableint* neighborIdPtr = (hnswlib::tableint*) (neighborSizePtr + 1);
        for (size_t idx = 0; idx < neighborSize; idx++) {
            neighborIdPtr[idx] = neighbors.top().first;
            neighbors.pop();
        }
    }

    // Level 0 neighbors formed


    // Form level 1 neighbors
    
    // std::vector<int> level1_nodes;
    // for (size_t id = 0; id < curElementsNum; id++) {
    //     if (merged_hnsw->linkLists_[id] != nullptr) {
    //         level1_nodes.push(id);
    //     }
    // }

    // cout << level1_nodes.size() << endl;

    
    // DistanceGraph graph_U1(level1_nodes.size(), M);
    // for (size_t id = 0; id < level1_nodes.size(); id++) {

    // }
    
    // Level 1 neighbors formed


    // Compare search time and recall
    int correct = 0;
    timer.reset();

    for (int i = 0; i < curElementsNum; i++) {

        float dist1 = merged_hnsw->fstdistfunc_((char*) (data + i * dim), merged_hnsw->getDataByInternalId(hnsw1->enterpoint_node_), merged_hnsw->dist_func_param_);
        float dist2 = merged_hnsw->fstdistfunc_((char*) (data + i * dim), merged_hnsw->getDataByInternalId(hnsw2->enterpoint_node_ + graph1ElementsNum), merged_hnsw->dist_func_param_);
        if (dist1 <= dist2) {
            merged_hnsw->enterpoint_node_ = hnsw1->enterpoint_node_;
        }
        else {
            merged_hnsw->enterpoint_node_ = hnsw2->enterpoint_node_ + graph1ElementsNum;
        }

        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = merged_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) {
            correct++;
        }
        else {
            if (merged_hnsw->enterpoint_node_ == hnsw1->enterpoint_node_) {
                merged_hnsw->enterpoint_node_ = hnsw2->enterpoint_node_ + graph1ElementsNum;
            }
            else {
                merged_hnsw->enterpoint_node_ = hnsw1->enterpoint_node_;
            }

            result = merged_hnsw->searchKnn(data + i * dim, 1);
            label = result.top().second;
            if (label == i) {
                correct++;
            }
        }
    }

    cout << "Original search time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    float recall = (float)correct / curElementsNum;
    std::cout << "Recall of original graph: " << recall << "\n";
    cout << endl;

    delete hnsw1;
    delete hnsw2;
    delete merged_hnsw;
}