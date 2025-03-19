#include "../hnsw/hnswlib/hnswlib.h"
#include <thread>

#include "timer.h"
#include "multi_thread.h"
#include "graph.h"

int c = 0;

int dim = 128;               int elementsNum = 100000;   
int M = 16;                  int ef_construction = 200; 
Timer timer;

void constructionTime(hnswlib::L2Space& space, int num_threads, float* data, int elementsNum)
{
    cout << "Oneoff Construction" << endl;
    cout << num_threads << " threads construction time:" << "\t";
    hnswlib::HierarchicalNSW<float>* mt_hnsw = new hnswlib::HierarchicalNSW<float>(&space, elementsNum, M, ef_construction);

    timer.reset();

    ParallelFor(0, elementsNum, num_threads, [&](size_t row, size_t threadId) {
        mt_hnsw->addPoint((void*)(data + dim * row), row);
    });

    cout << timer.getElapsedTimeSeconds() << " seconds" << endl;

    delete mt_hnsw;
}

void constructGraph(hnswlib::HierarchicalNSW<float>* graph, int num_threads, float* data, int elementsNum) {
    ParallelFor(0, elementsNum, num_threads, [&](size_t row, size_t threadId) {
        graph->addPoint((void*)(data + dim * row), row);
    });
}

void constructionTime(hnswlib::L2Space& space, int num_threads, float* data1, float* data2, int elementsNum1, int elementsNum2)
{
    cout << "Seperate Construction" << endl;
    cout << num_threads << " threads construction time:" << "\t";

    hnswlib::HierarchicalNSW<float>* mt_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, elementsNum1, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* mt_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, elementsNum2, M, ef_construction);

    timer.reset();

    std::thread thread1(constructGraph, mt_hnsw1, num_threads / 2, data1, elementsNum1);
    std::thread thread2(constructGraph, mt_hnsw2, num_threads / 2, data2, elementsNum2);
    thread1.join();
    thread2.join();

    cout << timer.getElapsedTimeSeconds() << " seconds" << endl;

    delete mt_hnsw1;
    delete mt_hnsw2;
}

float* generateData()
{
    std::mt19937 rng;   rng.seed(0);
    std::uniform_real_distribution<> distrib_real;
    cout << "random seed 0" << endl;
    
    float* data = new float[dim * elementsNum];
    for (int i = 0; i < dim * elementsNum; i++)
        data[i] = distrib_real(rng);
    
    return data;
}

int main() { 

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* mt_hnsw = new hnswlib::HierarchicalNSW<float>(&space, elementsNum, M, ef_construction);

    float* data = generateData();

    int elementsNum1 = elementsNum / 2;
    float* data1 = new float[dim * elementsNum1];
    for (int i = 0; i < elementsNum1; i++)
        data1[i] = data[i];

    int elementsNum2 = elementsNum / 2;
    float* data2 = new float[dim * elementsNum2];
    for (int i = 0; i < elementsNum2; i++)
        data2[i] = data[i + dim * elementsNum1];
        

    cout << elementsNum << " elements with " << dim << " dimensions" << endl;  
    // cout << std::thread::hardware_concurrency() << " cores" << endl;

    timer.reset();

    int thread = 12;
    cout << thread << " threads" << endl;
    ParallelFor(0, elementsNum, thread, [&](size_t row, size_t threadId) {
        mt_hnsw->addPoint((void*)(data + dim * row), row);
    });

    // for (size_t i = 0; i < elementsNum; i++) 
    //     mt_hnsw->addPoint(data + i * dim, i);

    cout << "Full graph construction time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    cout << endl;

    // DistanceGraph merged_graph(elementsNum, 2*M);
    // for (size_t id = 0; id < elementsNum; id++)
    // {
    //     int* data = (int*) mt_hnsw->get_linklist0(id);
    //     size_t size = mt_hnsw->getListCount((hnswlib::linklistsizeint*) data);

    //     for (size_t j = 1; j <= size; j++) 
    //     {
    //         int neighbor_id = *(data + j);
    //         float dist = mt_hnsw->fstdistfunc_(mt_hnsw->getDataByInternalId(id), mt_hnsw->getDataByInternalId(neighbor_id), mt_hnsw->dist_func_param_);
    //         merged_graph.addNeighbor(id, neighbor_id, dist);
    //     }
    // }
    // merged_graph.printGraph();
    // cout << endl;

    // constructionTime(space, 8, data, elementsNum);

    // constructionTime(space, 12, data1, elementsNum1);
    // constructionTime(space, 12, data2, elementsNum2);

    // constructionTime(space, 12, data1, data2, elementsNum1, elementsNum2);

    // Query the elements for themselves and measure recall
    timer.reset();

    std::vector<hnswlib::labeltype> neighbors(elementsNum);
    ParallelFor(0, elementsNum, thread, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = mt_hnsw->searchKnn(data + dim * row, 1);
        hnswlib::labeltype label = result.top().second;
        neighbors[row] = label;
    });

    float correct = 0;
    for (int i = 0; i < elementsNum; i++) {
        hnswlib::labeltype label = neighbors[i];
        if (label == i) correct++;
    }

    cout << "Search time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    cout << endl;

    float recall = correct / elementsNum;
    std::cout << "Recall: " << recall << "\n";

    delete[] data;
    delete[] data1;
    delete[] data2;
    delete mt_hnsw;
    return 0;
}