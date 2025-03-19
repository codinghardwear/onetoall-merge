#include "../hnsw/hnswlib/hnswlib.h"
#include "timer.h"
#include "graph.h"
#include "s_merge.h"

using std::cout;
using std::endl;

// Global c for updateNN()
int c = 0;

// Construction parameters
int dim = 128;   int totalElementsNum = 100;
int M = 16;     int ef_construction = 200;

// Merge parameters
int line = M * 1.25;     int cutoff = M * 0.75;

int main()
{
    float* data = generateData(totalElementsNum, 0);

    cout << totalElementsNum << " elements with " << dim << " dimensions" << endl;
    cout << endl;

    int graphAElementsNum = totalElementsNum / 2;     int graphBElementsNum = totalElementsNum / 2;

    // Split data
    float* data1 = new float[dim * graphAElementsNum];
    for (int i = 0; i < dim * graphAElementsNum; i++) 
        data1[i] = data[i];

    float* data2 = new float[dim * graphBElementsNum];
    for (int i = 0; i < dim * graphBElementsNum; i++) 
        data2[i] = data[i + dim * graphAElementsNum];
    
    // Build subgraph
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* hnsw1 = build_subgraph(&space, data1, graphAElementsNum);
    hnswlib::HierarchicalNSW<float>* hnsw2 = build_subgraph(&space, data1, graphBElementsNum);

    // Test build_subgraph()
    DistanceGraph hnsw1_graph(graphAElementsNum, 2 * M);
    for (size_t id = 0; id < graphAElementsNum; id++)
    {
        int* data = (int*) hnsw1->get_linklist0(id);
        size_t size = hnsw1->getListCount((hnswlib::linklistsizeint*) data);

        for (int j = 1; j <= size; j++) 
        {
            int neighbor_id = *(data + j);
            float dist = hnsw1->fstdistfunc_(hnsw1->getDataByInternalId(id), hnsw1->getDataByInternalId(neighbor_id), hnsw1->dist_func_param_);
            hnsw1_graph.addNeighbor(id, neighbor_id, dist);
        }
    }
    hnsw1_graph.printGraph(0, 4);

    // Test prepare_graphAB()
    auto [graph_p, graph_n] = prepare_graphAB(hnsw1);
    graph_p->printGraph(0, 4);
    graph_n->printGraph(0, 4);

    // cout << hnsw1->cur_element_count << endl;
    // cout << hnsw2->cur_element_count << endl;

    // cout << "cutoff line is " << line << " and cutoff is " << cutoff << endl;
    // cout << endl;

    // hnswlib::HierarchicalNSW<float>* merged_hnsw = s_merge(&space, hnsw1, hnsw2, data1, data2);

    // int correct = 0;
    // for (int i = 0; i < totalElementsNum; i++) {
    //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = merged_hnsw->searchKnn(data + i * dim, 1);
    //     hnswlib::labeltype label = result.top().second;
    //     if (label == i) correct++;
    // }
    // float recall = (float)correct / totalElementsNum;
    // std::cout << "Recall of merged graph: " << recall << "\n";
    // cout << endl;

    delete[] data;
    delete[] data1;
    delete[] data2;

    return 0;
}