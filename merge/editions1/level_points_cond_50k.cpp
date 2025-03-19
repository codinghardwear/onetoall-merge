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

    std::vector<std::vector<int>> hnsw1_nodes_level(hnsw1->maxlevel_ + 1);
    for (int id = 0; id < graph1ElementsNum; id++) {
        if (hnsw1->linkLists_[id] != nullptr) {
            int level = hnsw1->element_levels_[id];
            hnsw1_nodes_level[level].push_back(id);
        }
    }

    std::vector<std::vector<int>> hnsw2_nodes_level(hnsw2->maxlevel_ + 1);
    for (int id = 0; id < graph2ElementsNum; id++) {
        if (hnsw2->linkLists_[id] != nullptr) {
            int level = hnsw2->element_levels_[id];
            hnsw2_nodes_level[level].push_back(id);
        }
    }

    for (int i = 0; i < hnsw1_nodes_level[3].size(); i++) {
        cout << hnsw1_nodes_level[3][i] << " ";
    }
    cout << endl;

    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw1->searchKnnTargetLayer(data + 1 * dim, 16, 3);
    while (!result.empty()) {
        cout << result.top().second << "(dist: "<< result.top().first << ") ";
        result.pop();
    }
    cout << endl;


    delete hnsw1;
    delete hnsw2;

    return 0;
}