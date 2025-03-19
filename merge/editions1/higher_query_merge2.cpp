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
    for (int i = 0; i < dim * curElementsNum; i++) {
        data[i] = distrib_real(rng);  
    }

    hnswlib::HierarchicalNSW<float>* merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, maxElementsNum, M, ef_construction);
    merged_hnsw->cur_element_count = curElementsNum;
    merged_hnsw->maxlevel_ = hnsw1->maxlevel_;
    merged_hnsw->enterpoint_node_ = hnsw1->enterpoint_node_ ;

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
            // else {
            //     cout << "no enough space";
            // }
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
            // else {
            //     cout << "no enough space";
            // }
        }
    }

    // cout << merged_hnsw->maxlevel_ << endl;


    Timer timer;
    timer.reset();

    DistanceGraph graph_U(curElementsNum, 2 * M);

    for (int id = 0; id < graph1ElementsNum; id++)
    {
        hnswlib::linklistsizeint* data = hnsw1->get_linklist0(id);
        int size = hnsw1->getListCount(data);
        hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);

        for (int j = 0; j < size; j++) 
        {
            int neighbor_id = datal[j];
            float dist = hnsw1->fstdistfunc_(hnsw1->getDataByInternalId(id), hnsw1->getDataByInternalId(neighbor_id), hnsw1->dist_func_param_);
            graph_U.addNeighbor(id, neighbor_id, dist);
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
            graph_U.addNeighbor(id + graph1ElementsNum, neighbor_id + graph1ElementsNum, dist);
        }
    }

    cout << "Graph U time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    int line = 33;   int pop = 8;   int add = 8;
    cout << "Above " << line << ", pop " << pop << " and add " << add << endl;

    timer.reset();

    for (int id = 0; id < curElementsNum; id++) {
        if (graph_U.getNeighbors(id).size() >= line) {
            for (int i = 0; i < pop; i++) {
                graph_U.getNeighbors(id).pop();
            }
        }
    }

    cout << "Poping time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

    timer.reset();

    for (int id = 0; id < graph1ElementsNum; id++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw2->searchKnnTargetLayer(data + id * dim, add, 0);
        while(result.size() != 0) {
            graph_U.addNeighbor(id, result.top().second + graph1ElementsNum, result.top().first);
            result.pop();
        }
    }

    for (int id = graph1ElementsNum; id < curElementsNum; id++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw1->searchKnnTargetLayer(data + id * dim, add, 0);
        while(result.size() != 0) {
            graph_U.addNeighbor(id, result.top().second, result.top().first);
            result.pop();
        }
    }

    cout << "Adding time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    timer.reset();

    int iteration = 0;
    do {
        std::vector<std::unordered_set<int>> U(graph_U.getElementsNum());
        for (int i = 0; i < graph_U.getElementsNum(); i++) {
            auto pq = graph_U.getNeighbors(i);
            while (!pq.empty()) {
                U[i].insert(pq.top().first);
                U[pq.top().first].insert(i);
                pq.pop();
            }
        }
        
        c = 0;
        for (int u = 0; u < graph_U.getElementsNum(); u++) 
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

    // Level 0 neighbors loading
    for (int id = 0; id < curElementsNum; id++)
    {
        auto neighbors = graph_U.getNeighbors(id);
        int size = neighbors.size();
        hnswlib::linklistsizeint* data = merged_hnsw->get_linklist0(id);
        merged_hnsw->setListCount(data, size);
        hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
        for (int idx = 0; idx < size; idx++) {
            datal[idx] = neighbors.top().first;
            neighbors.pop();
        }
    }


    // Compare search time and recall
    int correct = 0;
    timer.reset();

    for (int i = 0; i < curElementsNum; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = merged_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) {
            correct++;
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