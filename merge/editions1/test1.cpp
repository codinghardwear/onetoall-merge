#include "../hnsw/hnswlib/hnswlib.h"

#include <thread>

#include "graph.h"

// Global c for updateNN()
int c = 0;

// Construction parameters
int dim = 128;   int totalElementsNum = 100;
int M = 16;     int ef_construction = 200;

// Merge parameters
int line = M * 1.25;     int cutoff = M * 0.75;

int main() 
{
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* hnsw = new hnswlib::HierarchicalNSW<float>(&space, totalElementsNum, M, ef_construction);

    float* data = generateData(totalElementsNum, 0);

    for (int i = 0; i < totalElementsNum; i++)s
        hnsw->addPoint(data + i * dim, i);

    DistanceGraph hnsw_graph(totalElementsNum, 2*M);
    for (size_t id = 0; id < totalElementsNum; id++)
    {
        int* data = (int*) hnsw->get_linklist0(id);
        size_t size = hnsw->getListCount((hnswlib::linklistsizeint*) data);

        for (size_t j = 1; j <= size; j++) 
        {
            int neighbor_id = *(data + j);
            float dist = hnsw->fstdistfunc_(hnsw->getDataByInternalId(id), hnsw->getDataByInternalId(neighbor_id), hnsw->dist_func_param_);
            hnsw_graph.addNeighbor(id, neighbor_id, dist);
        }
    }
    hnsw_graph.printGraph();

    return 0;
}